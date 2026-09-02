"""AI traffic-context analysis of a driving video (the 'watch the drive' path).

Pipeline: yt-dlp downloads the clip -> ffmpeg samples frames -> Claude
vision (claude-opus-4-8, best for multi-image analysis) describes the
driving context (traffic lights, lane position, following distance,
hazards). Degrades gracefully when yt-dlp / ffmpeg / the API key are
absent, so the rest of the app still runs.
"""
from __future__ import annotations

import base64
import glob
import html
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

log = logging.getLogger("drive_debrief.vision")

MODEL = "claude-opus-4-8"  # most capable; high-resolution multi-image vision
FRAME_INTERVAL_S = 4.0      # one sampled frame every N seconds

_FRAME_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "frame": {"type": "integer"},
                    "observation": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["traffic_light", "lane", "following_distance",
                                  "hazard", "signage", "positioning", "other"],
                    },
                    "severity": {"type": "string", "enum": ["good", "note", "concern"]},
                },
                "required": ["frame", "observation", "category", "severity"],
            },
        },
    },
    "required": ["overall", "findings"],
}

_PROMPT = (
    "These are sequential frames from a learner driver's forward-facing dashcam, "
    "in order. Act as a calm driving instructor reviewing the footage. For each "
    "notable moment, describe the traffic context and whether the driver is "
    "handling it well: traffic lights and whether they'd need to stop, lane "
    "position and drift, following distance to the car ahead, road signs, and any "
    "hazards (pedestrians, cyclists, junctions). Be specific and constructive. "
    "Reference frames by their number (1-based, in the order given). Then give a "
    "short overall assessment. Only comment on what is actually visible."
)


class VisionUnavailable(RuntimeError):
    """Raised when a prerequisite (binary or API key) is missing."""


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def download_video(url: str, out_dir: str, max_height: int = 480) -> str:
    """Download a video with yt-dlp; returns the file path."""
    if not _have("yt-dlp"):
        raise VisionUnavailable("yt-dlp is not installed (needed to fetch the video).")
    out_tmpl = os.path.join(out_dir, "video.%(ext)s")
    cmd = [
        "yt-dlp", "--no-playlist",
        "-f", f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best",
        "-o", out_tmpl, url,
    ]
    log.info("Downloading video: %s", url)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        msg = tail[0]
        log.warning("yt-dlp failed (%s): %s", proc.returncode, msg)
        hint = ""
        if "sign in" in msg.lower() or "bot" in msg.lower() or "403" in msg:
            hint = (" YouTube often blocks downloads from cloud/datacenter IPs "
                    "(this is a YouTube block, not the AI review). Try a direct "
                    "video-file URL, or upload the clip.")
        raise VisionUnavailable(f"Couldn't download the video: {msg}.{hint}")
    files = glob.glob(os.path.join(out_dir, "video.*"))
    if not files:
        raise VisionUnavailable("yt-dlp did not produce a video file.")
    return files[0]


def extract_frames(video_path: str, out_dir: str, every_seconds: float = FRAME_INTERVAL_S,
                   max_frames: int = 8, width: int = 768) -> List[str]:
    """Sample frames with ffmpeg (scaled down to keep token cost sane)."""
    if not _have("ffmpeg"):
        raise VisionUnavailable("ffmpeg is not installed (needed to sample frames).")
    pattern = os.path.join(out_dir, "frame_%03d.jpg")
    fps = 1.0 / every_seconds
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", video_path,
         "-vf", f"fps={fps},scale={width}:-2", "-frames:v", str(max_frames * 3),
         "-q:v", "4", pattern],
        check=True, timeout=180,
    )
    frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))
    return frames[:max_frames]


def _image_block(path: str) -> dict:
    with open(path, "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def analyse_frames(frame_paths: List[str], api_key: Optional[str] = None) -> dict:
    """Send the frames to Claude vision and return the structured analysis."""
    if not (api_key or os.getenv("ANTHROPIC_API_KEY")):
        raise VisionUnavailable(
            "ANTHROPIC_API_KEY is not set. In Sparkles, add it as a synced secret "
            "(model keys are kept out of the sandbox by default)."
        )
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise VisionUnavailable("The 'anthropic' package is not installed.") from exc

    # Identity-linked keys must name the workspace via a header.
    workspace = os.getenv("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace} if workspace else None
    kwargs = {"default_headers": headers} if headers else {}
    client = (anthropic.Anthropic(api_key=api_key, **kwargs) if api_key
              else anthropic.Anthropic(**kwargs))

    content: List[dict] = [_image_block(p) for p in frame_paths]
    content.append({"type": "text", "text": _PROMPT})

    log.info("Sending %d frames to %s (workspace=%s)", len(frame_paths), MODEL, bool(workspace))
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": _FRAME_SCHEMA}},
        )
    except Exception as exc:  # surface the actionable cause
        msg = str(exc)
        if "workspace" in msg.lower():
            raise VisionUnavailable(
                "This API key is identity-linked and needs a workspace id. Set "
                "ANTHROPIC_WORKSPACE_ID, or create a workspace-scoped key in the Console."
            ) from exc
        log.warning("Claude API error: %s", msg)
        raise VisionUnavailable(f"Claude API error: {msg}") from exc
    # A safety decline comes back as HTTP 200 with stop_reason "refusal".
    if resp.stop_reason == "refusal":
        cat = getattr(getattr(resp, "stop_details", None), "category", None)
        log.warning("Model declined the request (category=%s)", cat)
        raise VisionUnavailable(
            f"The model declined to analyse this video ({cat or 'policy'}). "
            "Try a different clip."
        )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    log.info("Vision analysis received (%d chars)", len(text))
    return json.loads(text)


def _frames_and_analyse(video_path: str, work: str, api_key, max_frames: int,
                        label: str, frames_dir: Optional[str] = None) -> dict:
    # When frames_dir is given, keep the frames there (for the timeline page).
    dest = frames_dir or work
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    frames = extract_frames(video_path, dest, max_frames=max_frames)
    if not frames:
        raise VisionUnavailable("No frames could be extracted from the video.")
    analysis = analyse_frames(frames, api_key=api_key)
    return {"url": label, "n_frames": len(frames), "analysis": analysis,
            "frames": [os.path.basename(f) for f in frames],
            "seconds_per_frame": FRAME_INTERVAL_S}


def analyse_video(url: str, api_key: Optional[str] = None, max_frames: int = 8,
                  frames_dir: Optional[str] = None) -> dict:
    """Download a video by URL, then frames -> Claude vision."""
    work = tempfile.mkdtemp(prefix="drivevision_")
    try:
        video = download_video(url, work)
        return _frames_and_analyse(video, work, api_key, max_frames, url, frames_dir=frames_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def analyse_video_file(video_path: str, api_key: Optional[str] = None,
                       max_frames: int = 8, label: Optional[str] = None,
                       frames_dir: Optional[str] = None) -> dict:
    """Analyse a local/uploaded/recorded video file (no download step)."""
    work = tempfile.mkdtemp(prefix="drivevision_")
    try:
        return _frames_and_analyse(video_path, work, api_key, max_frames,
                                   label or os.path.basename(video_path), frames_dir=frames_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)


_SEV_COLOUR = {"good": "#30a46c", "note": "#f5a623", "concern": "#e5484d"}


def _mmss(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def build_vision_report_html(result: dict, title: str = "AI drive analysis",
                             video_url: Optional[str] = None,
                             frames_link: Optional[str] = None) -> str:
    a = result.get("analysis", {})
    findings = a.get("findings", [])
    player = (f'<video src="{html.escape(video_url)}" controls playsinline '
              f'style="width:100%;border-radius:12px;margin-bottom:14px;background:#000"></video>'
              if video_url else "")
    frames_btn = (
        f'<a href="{html.escape(frames_link)}" style="display:inline-block;margin:0 0 16px;'
        f'color:#0091ff;font-size:13px;font-weight:600;text-decoration:none">🖼 View frame-by-frame →</a>'
        if frames_link else "")
    rows = "".join(
        f"<tr><td class='mono'>#{f.get('frame','')}</td>"
        f"<td><span class='dot' style='background:{_SEV_COLOUR.get(f.get('severity'),'#888')}'></span>"
        f"{html.escape(str(f.get('category','')).replace('_',' '))}</td>"
        f"<td>{html.escape(str(f.get('observation','')))}</td></tr>"
        for f in findings
    ) or "<tr><td colspan='3' class='empty'>No findings.</td></tr>"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#f4f5f7;color:#11181c}}
 .wrap{{max-width:820px;margin:0 auto;padding:28px 20px 60px}}
 h1{{font-size:22px;margin:0 0 4px}} .muted{{color:#687076;font-size:13px;margin-bottom:16px}}
 .overall{{background:#fff;border:1px solid #e6e8eb;border-radius:12px;padding:14px 16px;margin-bottom:16px}}
 table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e6e8eb;border-radius:12px;overflow:hidden}}
 th,td{{text-align:left;padding:10px 12px;font-size:13px;border-bottom:1px solid #eef0f2;vertical-align:top}}
 th{{background:#fafbfc;color:#687076}} tr:last-child td{{border-bottom:none}}
 .mono{{font-variant-numeric:tabular-nums;white-space:nowrap}}
 .dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px}}
 .empty{{text-align:center;color:#687076;padding:20px}}
 a{{color:#0091ff}}
</style></head><body><div class="wrap">
 <h1>{html.escape(title)}</h1>
 <div class="muted">{result.get('n_frames','?')} frames analysed by {MODEL}</div>
 {player}
 {frames_btn}
 <div class="overall"><strong>Overall:</strong> {html.escape(str(a.get('overall','')))}</div>
 <table><thead><tr><th>Frame</th><th>Type</th><th>Instructor note</th></tr></thead>
 <tbody>{rows}</tbody></table>
 <p class="muted">AI-assisted practice feedback from video, not a substitute for a qualified instructor.</p>
</div></body></html>"""


def build_frames_page_html(entry: dict, frame_base: str, back_link: Optional[str] = None) -> str:
    """A timeline page: each sampled frame shown with the AI's note for it."""
    frames = entry.get("frames", [])
    a = entry.get("analysis", {})
    spf = float(entry.get("seconds_per_frame", FRAME_INTERVAL_S))
    label = entry.get("label", "")

    by_frame: dict = {}
    for f in a.get("findings", []):
        by_frame.setdefault(int(f.get("frame", 0) or 0), []).append(f)

    cards = []
    for i, name in enumerate(frames):
        fnum = i + 1
        notes = by_frame.get(fnum, [])
        if notes:
            body = "".join(
                f'<div class="note"><span class="dot" style="background:{_SEV_COLOUR.get(n.get("severity"), "#888")}"></span>'
                f'<span class="cat">{html.escape(str(n.get("category", "")).replace("_", " "))}</span>'
                f'<span class="obs">{html.escape(str(n.get("observation", "")))}</span></div>'
                for n in notes)
        else:
            body = '<div class="note muted">No notable events in this frame.</div>'
        cards.append(
            f'<div class="frame"><img src="{html.escape(frame_base)}/{html.escape(name)}" loading="lazy" alt="frame {fnum}">'
            f'<div class="side"><div class="t">{_mmss(i * spf)} <span class="fn">· frame {fnum}/{len(frames)}</span></div>'
            f'{body}</div></div>')
    cards_html = "".join(cards) or '<div class="muted">No frames stored for this clip.</div>'
    back = f'<a href="{html.escape(back_link)}">← back</a> · ' if back_link else ""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Frame-by-frame — {html.escape(str(label))}</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#f4f5f7;color:#11181c}}
 .wrap{{max-width:760px;margin:0 auto;padding:28px 20px 60px}}
 h1{{font-size:22px;margin:0 0 4px}} .muted{{color:#687076;font-size:13px}}
 .overall{{background:#fff;border:1px solid #e6e8eb;border-radius:12px;padding:14px 16px;margin:14px 0 18px}}
 .frame{{display:flex;gap:14px;background:#fff;border:1px solid #e6e8eb;border-radius:14px;padding:12px;margin-bottom:14px}}
 .frame img{{width:260px;max-width:42%;border-radius:10px;object-fit:cover;background:#000}}
 .side{{flex:1;min-width:0}}
 .t{{font-variant-numeric:tabular-nums;font-weight:700;font-size:14px;margin-bottom:8px}}
 .fn{{color:#889;font-weight:500}}
 .note{{font-size:13px;line-height:1.5;margin-bottom:8px}}
 .dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:7px;vertical-align:middle}}
 .cat{{display:inline-block;font-size:11px;text-transform:capitalize;color:#687076;margin-right:6px}}
 .obs{{display:block;margin-top:2px}}
 a{{color:#0091ff;text-decoration:none}}
 @media(max-width:540px){{.frame{{flex-direction:column}}.frame img{{width:100%;max-width:100%}}}}
</style></head><body><div class="wrap">
 <h1>🖼 Frame-by-frame</h1>
 <div class="muted">{back}{len(frames)} frames from “{html.escape(str(label))}”, sampled every {spf:g}s · {MODEL}</div>
 <div class="overall"><strong>Overall:</strong> {html.escape(str(a.get('overall', '')))}</div>
 {cards_html}
</div></body></html>"""
