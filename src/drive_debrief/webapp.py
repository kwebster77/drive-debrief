"""One-page web app: upload GPS driver data OR paste a video link.

Runs on 0.0.0.0:8000 so it maps straight onto a Sparkles preview.

  * CSV / GPX upload -> the deterministic telematics debrief
  * video URL        -> Claude-vision traffic-context analysis
"""
from __future__ import annotations

import collections
import datetime
import html
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import asdict
from urllib.parse import parse_qsl

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .assessment import assess
from .events import Thresholds
from .history import append_run, build_entry, load_history
from .io import load_sensorlog_records, load_track, parse_ingest_text
from .pipeline import analyse_dataframe
from .progress import build_progress_html
from .report import build_report_html
from .vision import (
    MODEL,
    VisionUnavailable,
    analyse_video,
    analyse_video_file,
    build_frames_page_html,
    build_vision_report_html,
)

# Persistent data (survives restarts): progress history + stored videos.
DATA_DIR = os.getenv("DRIVE_DEBRIEF_DATA", "data")
MEDIA_DIR = os.path.join(DATA_DIR, "media")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
VIDEOS_INDEX = os.path.join(DATA_DIR, "videos.json")
os.makedirs(MEDIA_DIR, exist_ok=True)

app = FastAPI(title="drive-debrief")

# --- logging: to stdout (sandbox logs) and an in-memory ring for /logs ----
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("drive_debrief.web")
_RING: "collections.deque[str]" = collections.deque(maxlen=300)


class _RingHandler(logging.Handler):
    def emit(self, record):
        _RING.append(self.format(record))


_ring_handler = _RingHandler()
_ring_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
log.addHandler(_ring_handler)
log.setLevel(logging.INFO)

EXAMPLE_VIDEO = "https://www.youtube.com/watch?v=zBteu7mmQ3s"

# Accumulated SensorLog samples keyed by session id (phone auto-upload).
_TRIPS: dict = {}


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _save_history(label: str, summary, assessment) -> None:
    """Append a completed drive to the progress history."""
    result = {"summary": asdict(summary), "assessment": asdict(assessment)}
    try:
        entry = build_entry(label, _now_iso(), result)
        append_run(entry, HISTORY_PATH)
        log.info("history: saved '%s' (%d/%s)", label, summary.score, summary.grade)
    except Exception:
        log.exception("history: failed to save '%s'", label)


def _load_videos() -> list:
    if not os.path.exists(VIDEOS_INDEX):
        return []
    try:
        with open(VIDEOS_INDEX, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _store_video(data: bytes, suffix: str, label: str, result: dict, vid: str) -> dict:
    """Persist an uploaded/recorded video + its analysis; return the index entry."""
    filename = vid + (suffix or ".webm")
    with open(os.path.join(MEDIA_DIR, filename), "wb") as fh:
        fh.write(data)
    entry = {"id": vid, "label": label, "created": _now_iso(),
             "filename": filename, "n_frames": result.get("n_frames", 0),
             "analysis": result.get("analysis", {}),
             "frames": result.get("frames", []),
             "seconds_per_frame": result.get("seconds_per_frame", 4.0)}
    videos = _load_videos()
    videos.append(entry)
    with open(VIDEOS_INDEX, "w", encoding="utf-8") as fh:
        json.dump(videos, fh, indent=2)
    log.info("video: stored %s (%s, %d bytes)", vid, label, len(data))
    return entry

HOME = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>drive-debrief</title>
<style>
 :root{{--bg:#0a0c11;--card:#141a25;--card2:#0f131c;--line:#222b3a;--ink:#e9edf4;--dim:#93a0b4;--accent:#37b0ff;--accent2:#22d3a8}}
 *{{box-sizing:border-box}}
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;margin:0;color:var(--ink);min-height:100vh;
      background:radial-gradient(1100px 560px at 50% -12%,rgba(55,176,255,.16),transparent 62%),var(--bg)}}
 .wrap{{max-width:680px;margin:0 auto;padding:34px 20px 72px}}
 .topbar{{display:flex;justify-content:flex-end;gap:18px;font-size:13px;margin-bottom:22px}}
 .topbar a{{color:var(--dim);text-decoration:none}} .topbar a:hover{{color:var(--accent)}}
 .hero h1{{font-size:34px;line-height:1.08;letter-spacing:-.6px;margin:0 0 10px;
          background:linear-gradient(92deg,#ffffff,#a6dbff);-webkit-background-clip:text;background-clip:text;color:transparent}}
 .hero p{{color:var(--dim);font-size:15px;line-height:1.5;margin:0 0 26px;max-width:46ch}}
 .card{{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);border-radius:18px;
       padding:22px 22px 20px;margin-bottom:16px;box-shadow:inset 0 1px 0 rgba(255,255,255,.03),0 12px 34px rgba(0,0,0,.28);
       transition:transform .16s ease,border-color .16s ease}}
 .card:hover{{transform:translateY(-2px);border-color:#2f3b50}}
 .card h2{{font-size:17px;margin:8px 0 4px}} .card p{{color:var(--dim);font-size:13px;line-height:1.55;margin:0 0 14px}}
 input[type=file],input[type=text]{{width:100%;padding:12px 13px;border-radius:11px;border:1px solid #2a3444;
   background:#0b0f17;color:var(--ink);font-size:14px;margin-bottom:12px;outline:none;transition:border-color .15s,box-shadow .15s}}
 input[type=text]:focus{{border-color:var(--accent);box-shadow:0 0 0 3px rgba(55,176,255,.18)}}
 input[type=file]::file-selector-button{{background:#1b2431;color:var(--ink);border:0;border-radius:8px;padding:7px 12px;margin-right:10px;cursor:pointer}}
 button{{background:linear-gradient(180deg,var(--accent),#1f8fe6);color:#fff;border:0;border-radius:11px;
       padding:11px 18px;font-size:14px;font-weight:600;cursor:pointer;transition:filter .15s,transform .05s}}
 button:hover{{filter:brightness(1.08)}} button:active{{transform:translateY(1px)}}
 #recBtn{{background:#1b2431;border:1px solid #2e3a4e;filter:none}}
 .tag{{display:inline-flex;align-items:center;font-size:11px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;
      color:var(--accent);border:1px solid #234055;background:rgba(55,176,255,.08);border-radius:999px;padding:3px 10px;margin-bottom:10px}}
 .tag.g{{color:var(--accent2);border-color:#1f5045;background:rgba(34,211,168,.08)}}
 .tag.p{{color:#c99bff;border-color:#453163;background:rgba(160,110,255,.1)}}
 code{{background:#0b0f17;border:1px solid #222a38;padding:2px 7px;border-radius:6px;font-size:12px;color:#bcd6ff}}
 .foot{{color:#5b6572;font-size:12px;margin-top:26px;text-align:center;line-height:2}}
 .foot a{{text-decoration:none}}
</style></head><body><div class="wrap">
 <div class="topbar">
   <a href="/progress">Progress</a><a href="/videos">Videos</a><a href="/logs">Logs</a>
 </div>
 <div class="hero">
   <h1>🚗 drive-debrief</h1>
   <p>Turn a practice drive into a coaching debrief — share your GPS data, or a dashcam video for an AI review.</p>
 </div>

 <div class="card">
   <div class="tag g">GPS · deterministic</div>
   <h2>Upload your driver data</h2>
   <p><code>.csv</code>, <code>.gpx</code>, <code>.kml/.kmz</code>, or a <strong>Google Takeout</strong>
      location <code>.json</code> (phyphox / SensorLog / Strava / Google Timeline all work as-is).
      Google exports are multi-trip histories — we analyse your longest drive.</p>
   <form action="/analyze" method="post" enctype="multipart/form-data">
     <input type="file" name="file" accept=".csv,.gpx,.kml,.kmz,.json" required>
     <button type="submit">Analyse drive →</button>
   </form>
 </div>

 <div class="card">
   <div class="tag">Video · AI vision ({MODEL})</div>
   <h2>Analyse a dashcam video</h2>
   <p>Paste a link, upload a clip, or record now with your phone camera. Frames are read
      by Claude for traffic context — lights, lane position, following distance, hazards.</p>
   <form action="/analyze" method="post" enctype="multipart/form-data">
     <input type="text" name="url" value="{EXAMPLE_VIDEO}" placeholder="https://... (YouTube/dashcam)">
     <button type="submit">Analyse link →</button>
   </form>
   <form action="/analyze" method="post" enctype="multipart/form-data" style="margin-top:12px">
     <input type="file" name="video" accept="video/*" capture="environment" required>
     <button type="submit">Analyse uploaded clip →</button>
   </form>
   <div style="margin-top:12px">
     <button id="recBtn" type="button">● Record from camera</button>
     <span id="recStatus" style="color:#9aa4b2;font-size:12px;margin-left:8px"></span>
   </div>
 </div>

 <div class="card">
   <div class="tag p">Phone · auto-upload</div>
   <h2>Stream live from SensorLog (iOS)</h2>
   <p>In SensorLog, turn on <strong>HTTP</strong> upload and set the target URL to
      <code id="ingestUrl">…/ingest?session=myphone</code>. Log your drive, then open the trip:</p>
   <form onsubmit="location.href='/trip/'+(this.session.value||'myphone');return false;">
     <input type="text" name="session" value="myphone" placeholder="session name">
     <button type="submit">Open my live trip →</button>
   </form>
 </div>

 <div class="foot">Practice feedback, not a substitute for a qualified instructor.<br>
   <a href="/progress" style="color:#0091ff">progress</a> ·
   <a href="/videos" style="color:#0091ff">stored videos</a> ·
   <a href="/logs" style="color:#5b6572">activity log</a></div>

 <script>
   var _iu = document.getElementById('ingestUrl');
   if (_iu) _iu.textContent = location.origin + '/ingest?session=myphone';
   (function() {{
     var btn = document.getElementById('recBtn'), st = document.getElementById('recStatus');
     var rec, chunks = [], stream;
     if (!navigator.mediaDevices || !window.MediaRecorder) {{ btn.disabled = true; st.textContent = 'recording not supported here'; return; }}
     btn.onclick = async function() {{
       if (rec && rec.state === 'recording') {{ rec.stop(); return; }}
       try {{ stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'environment' }}, audio: false }}); }}
       catch (e) {{ st.textContent = 'camera blocked: ' + e.message; return; }}
       chunks = []; rec = new MediaRecorder(stream);
       rec.ondataavailable = function(e) {{ if (e.data.size) chunks.push(e.data); }};
       rec.onstop = async function() {{
         stream.getTracks().forEach(function(t) {{ t.stop(); }});
         st.textContent = 'uploading & analysing…';
         var fd = new FormData();
         fd.append('video', new Blob(chunks, {{ type: 'video/webm' }}), 'recording.webm');
         var r = await fetch('/analyze', {{ method: 'POST', body: fd }});
         document.open(); document.write(await r.text()); document.close();
       }};
       rec.start(); btn.textContent = '■ Stop & analyse';
       st.textContent = 'recording… mount your phone and drive safely';
     }};
   }})();
 </script>
</div></body></html>"""


def _error_page(title: str, message: str) -> HTMLResponse:
    body = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:-apple-system,sans-serif;background:#f4f5f7;color:#11181c;margin:0}}
.wrap{{max-width:640px;margin:0 auto;padding:48px 20px}}.box{{background:#fff;border:1px solid #e6e8eb;
border-radius:12px;padding:20px}}.box h1{{font-size:18px;margin:0 0 8px}}a{{color:#0091ff}}</style></head>
<body><div class="wrap"><div class="box"><h1>{title}</h1><p>{message}</p>
<p><a href="/">← back</a></p></div></div></body></html>"""
    return HTMLResponse(body, status_code=400)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(HOME)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "vision_key": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.get("/logs", response_class=HTMLResponse)
def logs() -> HTMLResponse:
    lines = "\n".join(_RING) or "(no activity yet — analyse a drive)"
    return HTMLResponse(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>activity log</title>"
        "<meta http-equiv='refresh' content='3'></head>"
        "<body style='background:#0b0d12;color:#cbd3e1;margin:0;font:13px/1.5 ui-monospace,Menlo,monospace'>"
        "<div style='padding:14px 18px'><a href='/' style='color:#0091ff'>← back</a> "
        "· auto-refreshes every 3s</div>"
        f"<pre style='padding:0 18px 24px;white-space:pre-wrap'>{html.escape(lines)}</pre>"
        "</body></html>"
    )


async def _analyse_video_upload(video: UploadFile, t0: float) -> HTMLResponse:
    suffix = os.path.splitext(video.filename)[1].lower() or ".webm"
    data = await video.read()
    log.info("Video upload: %s (%d bytes)", video.filename, len(data))
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    vid = uuid.uuid4().hex[:12]
    frames_dir = os.path.join(MEDIA_DIR, f"{vid}_frames")
    try:
        result = analyse_video_file(path, label=video.filename, frames_dir=frames_dir)
        entry = _store_video(data, suffix, video.filename or "recording", result, vid)
        log.info("Video-file ok: %d frames in %.2fs -> /video/%s",
                 result.get("n_frames", 0), time.time() - t0, entry["id"])
        return HTMLResponse(build_vision_report_html(
            result, video_url=f"/media/{entry['filename']}", frames_link=f"/frames/{vid}"))
    except VisionUnavailable as exc:
        log.warning("Video-file unavailable: %s", exc)
        return _error_page("Video analysis unavailable", str(exc))
    except Exception as exc:
        log.exception("Video-file failed after %.2fs", time.time() - t0)
        return _error_page("Video analysis failed", f"{type(exc).__name__}: {exc}")
    finally:
        os.unlink(path)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(None), video: UploadFile = File(None),
                  url: str = Form(None)) -> HTMLResponse:
    t0 = time.time()

    if video is not None and video.filename:
        return await _analyse_video_upload(video, t0)

    # GPS path — an uploaded file takes priority.
    if file is not None and file.filename:
        suffix = os.path.splitext(file.filename)[1].lower() or ".csv"
        data = await file.read()
        log.info("GPS upload: %s (%d bytes)", file.filename, len(data))
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            df = load_track(path)
            track, events, summary = analyse_dataframe(df, Thresholds())
            assessment = assess(events)
            report = build_report_html(track, events, summary, assessment,
                                       title=f"Debrief — {file.filename}")
            _save_history(file.filename, summary, assessment)
            log.info("GPS ok: %s -> %d/%s, %d events, '%s' in %.2fs",
                     file.filename, summary.score, summary.grade, len(events),
                     assessment.verdict, time.time() - t0)
            return HTMLResponse(report)
        except ValueError as exc:
            log.warning("GPS parse failed: %s: %s", file.filename, exc)
            return _error_page("Couldn't read that file", str(exc))
        finally:
            os.unlink(path)

    # Video path (AI vision).
    if url and url.strip():
        log.info("Video request: %s", url.strip())
        try:
            result = analyse_video(url.strip())
            log.info("Video ok: %d frames analysed in %.2fs", result.get("n_frames", 0), time.time() - t0)
            return HTMLResponse(build_vision_report_html(result))
        except VisionUnavailable as exc:
            log.warning("Video unavailable: %s", exc)
            return _error_page("Video analysis unavailable", str(exc))
        except Exception as exc:  # keep the demo alive; log the full trace
            log.exception("Video failed after %.2fs", time.time() - t0)
            return _error_page("Video analysis failed", f"{type(exc).__name__}: {exc}")

    log.info("Empty request")
    return _error_page("Nothing to analyse", "Upload a CSV/GPX/KML/JSON file or paste a video link.")


@app.get("/ingest")
async def ingest_get(request: Request, session: str = "default") -> dict:
    """SensorLog GET mode: each sample's fields arrive as query params."""
    row = {k: v for k, v in request.query_params.items() if k != "session"}
    if not row:
        return {"received": 0, "note": "GET with no data params — enable sensors incl. Location"}
    _TRIPS.setdefault(session, []).append(row)
    n = len(_TRIPS[session])
    if n <= 3 or n % 50 == 0:
        log.info("ingest[%s] GET: 1 row (total %d) keys=%s", session, n, list(row)[:6])
    return {"received": 1, "total": n, "report": f"/trip/{session}"}


@app.post("/ingest")
async def ingest(request: Request, session: str = "default") -> dict:
    """Receive streamed samples auto-uploaded from a phone (SensorLog HTTP POST).

    Accepts JSON (array/object), newline-JSON, CSV, or form-url-encoded bodies,
    accumulated per ``session``. Open ``/trip/<session>`` for the debrief.
    (For one-shot files, use the upload form on the home page.)
    """
    # Streamed body (SensorLog HTTP): tolerate JSON / NDJSON / CSV / form-url.
    raw = await request.body()
    ctype = request.headers.get("content-type", "")
    if "x-www-form-urlencoded" in ctype:
        row = {k: v for k, v in parse_qsl(raw.decode("utf-8", "replace")) if k != "session"}
        rows = [row] if row else []
    else:
        rows = parse_ingest_text(raw.decode("utf-8", "replace"))
    _TRIPS.setdefault(session, []).extend(rows)
    log.info("ingest[%s]: ctype=%s bytes=%d -> +%d rows (total %d)",
             session, ctype[:40] or "?", len(raw), len(rows), len(_TRIPS[session]))
    if not rows:
        # Show the caller a hint + a sample of what we received (for debugging).
        sample = raw[:200].decode("utf-8", "replace")
        return {"received": 0, "total": len(_TRIPS.get(session, [])),
                "note": "couldn't parse any rows from this body", "content_type": ctype,
                "sample": sample}
    return {"received": len(rows), "total": len(_TRIPS[session]), "report": f"/trip/{session}"}


@app.get("/trip/{session}", response_class=HTMLResponse)
def trip(session: str) -> HTMLResponse:
    records = _TRIPS.get(session) or []
    if not records:
        return _error_page(
            "No data yet",
            f"No samples received for session '{session}'. Point SensorLog's HTTP "
            "upload at /ingest?session=" + session + " and start logging.",
        )
    try:
        df = load_sensorlog_records(records, f"session:{session}")
        track, events, summary = analyse_dataframe(df, Thresholds())
        assessment = assess(events)
        _save_history(f"trip: {session}", summary, assessment)
        log.info("trip[%s]: %d samples -> %d/%s", session, len(records), summary.score, summary.grade)
        return HTMLResponse(build_report_html(track, events, summary, assessment,
                                              title=f"Live trip — {session}"))
    except ValueError as exc:
        return _error_page("Couldn't analyse this trip yet", str(exc))


@app.get("/progress", response_class=HTMLResponse)
def progress() -> HTMLResponse:
    entries = load_history(HISTORY_PATH)
    return HTMLResponse(build_progress_html(entries, title="Your driving progress"))


@app.get("/media/{name}")
def media(name: str):
    """Serve a stored video file."""
    safe = os.path.basename(name)
    path = os.path.join(MEDIA_DIR, safe)
    if not os.path.exists(path):
        return _error_page("Not found", "That video is no longer available.")
    return FileResponse(path)


@app.get("/frame/{vid}/{name}")
def frame(vid: str, name: str):
    """Serve one stored frame image for the timeline page."""
    path = os.path.join(MEDIA_DIR, f"{os.path.basename(vid)}_frames", os.path.basename(name))
    if not os.path.exists(path):
        return _error_page("Not found", "That frame is no longer available.")
    return FileResponse(path)


@app.get("/frames/{vid}", response_class=HTMLResponse)
def frames_page(vid: str) -> HTMLResponse:
    entry = next((v for v in _load_videos() if v.get("id") == vid), None)
    if entry is None:
        return _error_page("Not found", "No stored video with that id.")
    return HTMLResponse(build_frames_page_html(
        entry, frame_base=f"/frame/{vid}", back_link=f"/video/{vid}"))


@app.get("/video/{vid}", response_class=HTMLResponse)
def video(vid: str) -> HTMLResponse:
    entry = next((v for v in _load_videos() if v.get("id") == vid), None)
    if entry is None:
        return _error_page("Not found", "No stored video with that id.")
    has_frames = bool(entry.get("frames"))
    return HTMLResponse(build_vision_report_html(
        {"n_frames": entry.get("n_frames", 0), "analysis": entry.get("analysis", {})},
        title=f"AI drive analysis — {entry.get('label', vid)}",
        video_url=f"/media/{entry['filename']}",
        frames_link=f"/frames/{vid}" if has_frames else None,
    ))


@app.get("/videos", response_class=HTMLResponse)
def videos() -> HTMLResponse:
    items = list(reversed(_load_videos()))
    rows = "".join(
        f"<tr><td><a href='/video/{v['id']}'>{html.escape(str(v.get('label','')))}</a></td>"
        f"<td class='mono'>{html.escape(str(v.get('created','')))}</td>"
        f"<td class='mono'>{v.get('n_frames','')} frames</td>"
        f"<td><a href='/frames/{v['id']}'>frame-by-frame</a></td>"
        f"<td><a href='/media/{v['filename']}'>play</a></td></tr>"
        for v in items
    ) or "<tr><td colspan='5' class='empty'>No videos stored yet.</td></tr>"
    return HTMLResponse(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>stored videos</title>"
        "<style>body{font-family:-apple-system,sans-serif;background:#f4f5f7;color:#11181c;margin:0}"
        ".wrap{max-width:760px;margin:0 auto;padding:28px 20px}h1{font-size:22px}"
        "table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e6e8eb;border-radius:12px;overflow:hidden}"
        "th,td{text-align:left;padding:10px 12px;font-size:13px;border-bottom:1px solid #eef0f2}"
        "th{background:#fafbfc;color:#687076}.mono{font-variant-numeric:tabular-nums}"
        ".empty{text-align:center;color:#687076;padding:20px}a{color:#0091ff}</style></head>"
        "<body><div class='wrap'><h1>Stored videos</h1><p><a href='/'>← back</a></p>"
        "<table><thead><tr><th>Drive</th><th>When</th><th>Frames</th><th></th><th></th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></body></html>"
    )


def main() -> None:
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
