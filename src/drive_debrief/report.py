"""Self-contained HTML debrief with an inline SVG route map.

No network, no map tiles, no JS: the whole report is one HTML string,
which is exactly what you want when it has to render from a headless
cloud sandbox.
"""
from __future__ import annotations

import html
from typing import List

import numpy as np

from .assessment import Assessment
from .events import Event
from .kinematics import G, Track
from .scoring import Summary

_COLOURS = {
    "harsh_braking": "#e5484d",
    "harsh_acceleration": "#f76808",
    "hard_cornering": "#8e4ec6",
    "long_stop": "#0091ff",
    "stop": "#8b8d98",
}
_GRADE_COLOUR = {"A": "#30a46c", "B": "#30a46c", "C": "#f5a623", "D": "#f76808", "E": "#e5484d"}


def _mmss(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _project(track: Track, width: int, height: int, pad: int):
    """Equirectangular projection of the track into an SVG viewbox."""
    lat, lon = track.lat, track.lon
    lat0 = float(np.mean(lat))
    x = (lon - lon.min()) * np.cos(np.radians(lat0))
    y = (lat.max() - lat)  # north up
    span_x = max(x.max() - x.min(), 1e-9)
    span_y = max(y.max() - y.min(), 1e-9)
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
    # Centre the drawing.
    off_x = pad + ((width - 2 * pad) - span_x * scale) / 2
    off_y = pad + ((height - 2 * pad) - span_y * scale) / 2
    px = off_x + (x - x.min()) * scale
    py = off_y + (y - y.min()) * scale
    return px, py


def _nearest_index(track: Track, t: float) -> int:
    return int(np.argmin(np.abs(track.t - t)))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _roughness_colour(g: float) -> str:
    """Green (smooth) → amber → red (rough) by combined g-force."""
    g = max(0.0, min(g, 0.45))
    t = g / 0.45
    green, amber, red = (48, 164, 108), (245, 166, 35), (229, 72, 77)
    if t < 0.5:
        u = t / 0.5
        c = [_lerp(green[i], amber[i], u) for i in range(3)]
    else:
        u = (t - 0.5) / 0.5
        c = [_lerp(amber[i], red[i], u) for i in range(3)]
    return f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"


def _heatmap_segments(track: Track, px: np.ndarray, py: np.ndarray) -> str:
    """Colour each route segment by how rough the driving was there."""
    rough = np.sqrt(track.a_long ** 2 + track.a_lat ** 2) / G
    segs = []
    for i in range(len(px) - 1):
        col = _roughness_colour(float(max(rough[i], rough[i + 1])))
        segs.append(
            f'<line x1="{px[i]:.1f}" y1="{py[i]:.1f}" x2="{px[i+1]:.1f}" y2="{py[i+1]:.1f}" '
            f'stroke="{col}" stroke-width="4" stroke-linecap="round"/>'
        )
    return "".join(segs)


def _svg_speed(track: Track, events: List[Event], width: int = 720, height: int = 190) -> str:
    """Speed-over-time line with dashed markers at each flagged event."""
    pad_l, pad_r, pad_t, pad_b = 40, 16, 16, 24
    t = track.t
    mph = track.speed * 2.23694
    tmin, tmax = float(t[0]), float(t[-1])
    tspan = (tmax - tmin) or 1.0
    vmax = max(float(np.max(mph)) if len(mph) else 1.0, 1.0) * 1.15

    def X(tt):
        return pad_l + (tt - tmin) / tspan * (width - pad_l - pad_r)

    def Y(v):
        return height - pad_b - (v / vmax) * (height - pad_t - pad_b)

    pts = " ".join(f"{X(tt):.1f},{Y(v):.1f}" for tt, v in zip(t, mph))
    marks = []
    for e in events:
        if e.kind == "stop":
            continue
        col = _COLOURS.get(e.kind, "#888")
        x = X(e.t_peak)
        marks.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{height - pad_b}" stroke="{col}" '
            f'stroke-width="1.5" stroke-dasharray="3,3" opacity="0.75"/>'
            f'<circle cx="{x:.1f}" cy="{pad_t}" r="3.5" fill="{col}">'
            f"<title>{html.escape(e.label)} @ {_mmss(e.t_peak)}</title></circle>"
        )
    mid = int(vmax * 0.87)
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Speed over time">
  <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#e6e8eb"/>
  <text x="{pad_l - 6}" y="{Y(0):.1f}" text-anchor="end" font-size="10" fill="#889">0</text>
  <text x="{pad_l - 6}" y="{Y(mid):.1f}" text-anchor="end" font-size="10" fill="#889">{mid}</text>
  <text x="6" y="{pad_t + 4}" font-size="10" fill="#889">mph</text>
  <polyline points="{pts}" fill="none" stroke="#0091ff" stroke-width="2"/>
  {''.join(marks)}
  <text x="{width - pad_r}" y="{height - 6}" text-anchor="end" font-size="10" fill="#889">time (s) →</text>
</svg>"""


def _svg_map(track: Track, events: List[Event], width: int = 720, height: int = 420) -> str:
    px, py = _project(track, width, height, pad=28)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(px, py))
    segments = _heatmap_segments(track, px, py)

    markers = []
    for e in events:
        if e.kind == "stop":
            continue  # keep the map readable; ordinary stops are expected
        i = _nearest_index(track, e.t_peak)
        colour = _COLOURS.get(e.kind, "#8b8d98")
        markers.append(
            f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="7" fill="{colour}" '
            f'fill-opacity="0.85" stroke="white" stroke-width="2">'
            f"<title>{html.escape(e.label)} @ {_mmss(e.t_peak)}</title></circle>"
        )

    sx, sy = px[0], py[0]
    ex, ey = px[-1], py[-1]
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Route map">
  <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#0b0d12"/>
  <polyline points="{pts}" fill="none" stroke="#0b0d12" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
  {segments}
  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="#30a46c" stroke="white" stroke-width="2"><title>Start</title></circle>
  <circle cx="{ex:.1f}" cy="{ey:.1f}" r="6" fill="#111" stroke="white" stroke-width="2"><title>End</title></circle>
  {''.join(markers)}
</svg>"""


def _card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="card"><div class="k">{html.escape(label)}</div>'
        f'<div class="v">{html.escape(value)}</div>{sub_html}</div>'
    )


def _event_rows(events: List[Event]) -> str:
    if not events:
        return '<tr><td colspan="5" class="empty">No events flagged — smooth drive!</td></tr>'
    rows = []
    for e in events:
        colour = _COLOURS.get(e.kind, "#8b8d98")
        val = f"{e.peak_value}{e.unit}"
        rows.append(
            f"<tr>"
            f'<td class="mono">{_mmss(e.t_peak)}</td>'
            f'<td><span class="dot" style="background:{colour}"></span>{html.escape(e.label)}</td>'
            f'<td><span class="sev sev-{e.severity}">{e.severity}</span></td>'
            f'<td class="mono">{html.escape(val)}</td>'
            f"<td>{html.escape(e.detail)}</td>"
            f"</tr>"
        )
    return "".join(rows)


def _legend(events: List[Event]) -> str:
    kinds = []
    for e in events:
        if e.kind not in kinds and e.kind != "stop":
            kinds.append(e.kind)
    chips = []
    for k in kinds:
        label = next(e.label for e in events if e.kind == k)
        chips.append(
            f'<span class="chip"><span class="dot" style="background:{_COLOURS.get(k, "#888")}"></span>'
            f"{html.escape(label)}</span>"
        )
    return "".join(chips)


def _verdict_banner(assessment: Assessment) -> str:
    if assessment is None:
        return ""
    passed = assessment.passed
    bg = "#e6f6ec" if passed else "#ffe9e9"
    fg = "#18794e" if passed else "#cd2b31"
    icon = "✓" if passed else "✕"
    tally = (
        f"{assessment.minors} driving &middot; "
        f"{assessment.serious} serious &middot; "
        f"{assessment.dangerous} dangerous"
    )
    return (
        f'<div class="verdict" style="background:{bg};color:{fg}">'
        f'<span class="vicon">{icon}</span>'
        f'<span class="vtext"><strong>Mock test: {html.escape(assessment.verdict)}</strong>'
        f'<span class="vtally">{tally} faults</span></span></div>'
    )


def build_report_html(track: Track, events: List[Event], summary: Summary,
                      assessment: Assessment = None, title: str = "Practice-drive debrief") -> str:
    grade_colour = _GRADE_COLOUR.get(summary.grade, "#888")
    counts = summary.counts
    brake = counts.get("harsh_braking", 0)
    accel = counts.get("harsh_acceleration", 0)
    corner = counts.get("hard_cornering", 0)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #f4f5f7; color: #11181c; }}
  .wrap {{ max-width: 860px; margin: 0 auto; padding: 28px 20px 60px; }}
  header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 20px; }}
  h1 {{ font-size: 22px; margin: 0; }}
  .muted {{ color: #687076; font-size: 13px; margin-top: 4px; }}
  .score {{ text-align: center; min-width: 116px; }}
  .score .n {{ font-size: 44px; font-weight: 800; line-height: 1; color: {grade_colour}; }}
  .score .g {{ font-size: 13px; color: #687076; }}
  .verdict {{ display: flex; align-items: center; gap: 12px; border-radius: 12px; padding: 14px 16px; margin: 4px 0 6px; }}
  .verdict .vicon {{ font-size: 20px; font-weight: 800; }}
  .verdict .vtext {{ display: flex; flex-direction: column; }}
  .verdict .vtally {{ font-size: 12px; opacity: 0.85; margin-top: 2px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin: 18px 0; }}
  .card {{ background: white; border: 1px solid #e6e8eb; border-radius: 12px; padding: 14px; }}
  .card .k {{ font-size: 12px; color: #687076; }}
  .card .v {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .card .sub {{ font-size: 12px; color: #889; margin-top: 2px; }}
  .mapwrap {{ background: white; border: 1px solid #e6e8eb; border-radius: 14px; padding: 12px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
  .chip {{ font-size: 12px; color: #333; display: inline-flex; align-items: center; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
  .grad {{ width: 26px; height: 10px; border-radius: 3px; display: inline-block; margin-right: 6px;
           background: linear-gradient(90deg, rgb(48,164,108), rgb(245,166,35), rgb(229,72,77)); }}
  .chartlabel {{ font-size: 12px; color: #687076; margin: 2px 4px 8px; }}
  h2 {{ font-size: 16px; margin: 26px 0 10px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e6e8eb; border-radius: 12px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 10px 12px; font-size: 13px; border-bottom: 1px solid #eef0f2; vertical-align: top; }}
  th {{ background: #fafbfc; color: #687076; font-weight: 600; }}
  tr:last-child td {{ border-bottom: none; }}
  .mono {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .empty {{ text-align: center; color: #687076; padding: 22px; }}
  .sev {{ font-size: 11px; padding: 2px 8px; border-radius: 999px; text-transform: capitalize; }}
  .sev-minor {{ background: #e6f6ec; color: #18794e; }}
  .sev-moderate {{ background: #fff4e5; color: #ad5700; }}
  .sev-severe {{ background: #ffe9e9; color: #cd2b31; }}
  footer {{ margin-top: 30px; font-size: 12px; color: #889; text-align: center; }}
</style></head>
<body><div class="wrap">
  <header>
    <div>
      <h1>{html.escape(title)}</h1>
      <div class="muted">{summary.distance_km} km &middot; {summary.duration_min} min &middot; generated by drive-debrief</div>
    </div>
    <div class="score"><div class="n">{summary.score}</div><div class="g">grade {summary.grade}</div></div>
  </header>

  {_verdict_banner(assessment)}

  <div class="grid">
    {_card("Harsh braking", str(brake))}
    {_card("Harsh accel.", str(accel))}
    {_card("Hard corners", str(corner))}
    {_card("Max speed", f"{summary.max_speed_mph:.0f} mph")}
    {_card("Avg speed", f"{summary.avg_speed_mph:.0f} mph")}
  </div>

  <div class="mapwrap">
    {_svg_map(track, events)}
    <div class="legend">
      <span class="chip"><span class="grad"></span>route colour = smoothness (green → red)</span>
      {_legend(events)}
    </div>
  </div>

  <div class="mapwrap" style="margin-top:16px">
    <div class="chartlabel">Speed &amp; flagged events</div>
    {_svg_speed(track, events)}
  </div>

  <h2>What happened, minute by minute</h2>
  <table>
    <thead><tr><th>Time</th><th>Event</th><th>Severity</th><th>Peak</th><th>Coaching note</th></tr></thead>
    <tbody>{_event_rows(events)}</tbody>
  </table>

  <footer>Events derived from GPS speed &amp; heading. This is practice feedback, not a substitute for a qualified instructor.</footer>
</div></body></html>"""
