"""One-page web app: upload GPS driver data OR paste a video link.

Runs on 0.0.0.0:8000 so it maps straight onto a Sparkles preview.

  * CSV / GPX upload -> the deterministic telematics debrief
  * video URL        -> Claude-vision traffic-context analysis
"""
from __future__ import annotations

import collections
import html
import logging
import os
import tempfile
import time

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from .assessment import assess
from .events import Thresholds
from .io import load_sensorlog_records, load_track, parse_ingest_text
from .pipeline import analyse_dataframe
from .report import build_report_html
from .vision import (
    MODEL,
    VisionUnavailable,
    analyse_video,
    analyse_video_file,
    build_vision_report_html,
)

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

HOME = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>drive-debrief</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#0b0d12;color:#e6e8eb}}
 .wrap{{max-width:720px;margin:0 auto;padding:48px 20px 60px}}
 h1{{font-size:28px;margin:0 0 6px}} .sub{{color:#9aa4b2;margin-bottom:28px}}
 .card{{background:#151922;border:1px solid #232936;border-radius:16px;padding:22px;margin-bottom:18px}}
 .card h2{{font-size:16px;margin:0 0 4px}} .card p{{color:#9aa4b2;font-size:13px;margin:0 0 14px}}
 input[type=file],input[type=text]{{width:100%;box-sizing:border-box;padding:11px 12px;border-radius:10px;
   border:1px solid #2b3342;background:#0f131a;color:#e6e8eb;font-size:14px;margin-bottom:12px}}
 button{{background:#0091ff;color:#fff;border:0;border-radius:10px;padding:11px 18px;font-size:14px;font-weight:600;cursor:pointer}}
 button:hover{{background:#0a84e0}}
 .tag{{display:inline-block;font-size:11px;color:#0091ff;border:1px solid #22405e;background:#0e1b2a;border-radius:999px;padding:2px 9px;margin-bottom:10px}}
 code{{background:#0f131a;padding:1px 6px;border-radius:5px;font-size:12px}}
 .foot{{color:#5b6572;font-size:12px;margin-top:20px}}
</style></head><body><div class="wrap">
 <h1>🚗 drive-debrief</h1>
 <div class="sub">Turn a practice drive into a coaching debrief. Share your GPS data <em>or</em> a video.</div>

 <div class="card">
   <div class="tag">GPS · deterministic</div>
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
   <div class="tag">Phone · auto-upload</div>
   <h2>Stream live from SensorLog (iOS)</h2>
   <p>In SensorLog, turn on <strong>HTTP</strong> upload and set the target URL to
      <code id="ingestUrl">…/ingest?session=myphone</code>. Log your drive, then open the trip:</p>
   <form onsubmit="location.href='/trip/'+(this.session.value||'myphone');return false;">
     <input type="text" name="session" value="myphone" placeholder="session name">
     <button type="submit">Open my live trip →</button>
   </form>
 </div>

 <div class="foot">Practice feedback, not a substitute for a qualified instructor. ·
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
    try:
        result = analyse_video_file(path, label=video.filename)
        log.info("Video-file ok: %d frames in %.2fs", result.get("n_frames", 0), time.time() - t0)
        return HTMLResponse(build_vision_report_html(result))
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


@app.post("/ingest")
async def ingest(request: Request, session: str = "default", file: UploadFile = File(None)) -> dict:
    """Receive data auto-uploaded from a phone (SensorLog HTTP mode).

    Accepts either a one-shot file upload (any supported format) or a stream
    of SensorLog JSON rows, accumulated per ``session``. Open ``/trip/<session>``
    to see the debrief for whatever has arrived so far.
    """
    # One-shot file upload: parse straight away and expose the samples.
    if file is not None and file.filename:
        suffix = os.path.splitext(file.filename)[1].lower() or ".csv"
        data = await file.read()
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            df = load_track(path)
        finally:
            os.unlink(path)
        _TRIPS[session] = df.to_dict("records")
        log.info("ingest[%s]: file %s -> %d samples", session, file.filename, len(df))
        return {"received": len(df), "total": len(_TRIPS[session]), "report": f"/trip/{session}"}

    # Streamed body (SensorLog HTTP): tolerate JSON / NDJSON / CSV.
    raw = await request.body()
    ctype = request.headers.get("content-type", "")
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
        log.info("trip[%s]: %d samples -> %d/%s", session, len(records), summary.score, summary.grade)
        return HTMLResponse(build_report_html(track, events, summary, assessment,
                                              title=f"Live trip — {session}"))
    except ValueError as exc:
        return _error_page("Couldn't analyse this trip yet", str(exc))


def main() -> None:
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
