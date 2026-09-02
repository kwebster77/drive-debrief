"""Render a progress report (trend across drives) as self-contained HTML."""
from __future__ import annotations

import html
from typing import List

_W, _H, _PAD = 640, 160, 28


def _line_chart(values: List[float], colour: str, lower_is_better: bool = False) -> str:
    """A small SVG line chart with points and a last-value label."""
    if not values:
        return '<div class="empty">No drives yet.</div>'
    n = len(values)
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0
    # Pad the range a little so the line isn't glued to the edges.
    vmin -= span * 0.15
    vmax += span * 0.15
    span = vmax - vmin

    def x(i):
        return _PAD if n == 1 else _PAD + (i / (n - 1)) * (_W - 2 * _PAD)

    def y(v):
        return _H - _PAD - ((v - vmin) / span) * (_H - 2 * _PAD)

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    dots = "".join(
        f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.5" fill="{colour}"/>'
        for i, v in enumerate(values)
    )
    last, first = values[-1], values[0]
    if last == first:
        trend, improved = "→", True
    elif last > first:
        trend, improved = "▲", (not lower_is_better)
    else:
        trend, improved = "▼", lower_is_better
    trend_colour = "#30a46c" if improved else "#e5484d"
    return f"""<svg viewBox="0 0 {_W} {_H}" width="100%" xmlns="http://www.w3.org/2000/svg">
  <line x1="{_PAD}" y1="{_H - _PAD}" x2="{_W - _PAD}" y2="{_H - _PAD}" stroke="#e6e8eb"/>
  <polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.5" stroke-linejoin="round"/>
  {dots}
  <text x="{_W - _PAD}" y="18" text-anchor="end" font-size="13" fill="{trend_colour}"
        font-family="sans-serif" font-weight="700">{last:g} {trend}</text>
</svg>"""


def _rows(entries: List[dict]) -> str:
    rows = []
    for e in reversed(entries):  # newest first
        pass_badge = "✓" if e.get("passed") else "✕"
        pass_colour = "#18794e" if e.get("passed") else "#cd2b31"
        rows.append(
            f"<tr><td>{html.escape(str(e.get('label', '')))}</td>"
            f"<td class='mono'>{html.escape(str(e.get('timestamp', '')))}</td>"
            f"<td class='mono'>{e.get('score', '')}</td>"
            f"<td class='mono'>{e.get('events_per_km', '')}</td>"
            f"<td style='color:{pass_colour};font-weight:700'>{pass_badge}</td></tr>"
        )
    return "".join(rows)


def build_progress_html(entries: List[dict], title: str = "Driving progress") -> str:
    scores = [e.get("score", 0) for e in entries]
    per_km = [e.get("events_per_km", 0) for e in entries]
    n = len(entries)
    latest = scores[-1] if scores else "—"
    best = max(scores) if scores else "—"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #f4f5f7; color: #11181c; }}
  .wrap {{ max-width: 760px; margin: 0 auto; padding: 28px 20px 60px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .muted {{ color: #687076; font-size: 13px; margin-bottom: 18px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }}
  .card {{ background: white; border: 1px solid #e6e8eb; border-radius: 12px; padding: 14px; }}
  .card .k {{ font-size: 12px; color: #687076; }}
  .card .v {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
  .panel {{ background: white; border: 1px solid #e6e8eb; border-radius: 14px; padding: 14px 16px; margin-bottom: 16px; }}
  .panel h2 {{ font-size: 14px; margin: 0 0 6px; color: #11181c; }}
  .panel .hint {{ font-size: 12px; color: #889; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e6e8eb; border-radius: 12px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 9px 12px; font-size: 13px; border-bottom: 1px solid #eef0f2; }}
  th {{ background: #fafbfc; color: #687076; }}
  tr:last-child td {{ border-bottom: none; }}
  .mono {{ font-variant-numeric: tabular-nums; }}
  .empty {{ color: #687076; padding: 20px; text-align: center; }}
</style></head>
<body><div class="wrap">
  <h1>{html.escape(title)}</h1>
  <div class="muted">{n} drive(s) recorded</div>
  <div class="grid">
    <div class="card"><div class="k">Latest score</div><div class="v">{latest}</div></div>
    <div class="card"><div class="k">Best score</div><div class="v">{best}</div></div>
    <div class="card"><div class="k">Drives</div><div class="v">{n}</div></div>
  </div>
  <div class="panel"><h2>Smoothness score over time</h2>{_line_chart(scores, "#0091ff")}
    <div class="hint">Higher is better.</div></div>
  <div class="panel"><h2>Harsh events per km</h2>{_line_chart(per_km, "#f76808", lower_is_better=True)}
    <div class="hint">Lower is better — fewer harsh braking / cornering events per km.</div></div>
  <table>
    <thead><tr><th>Drive</th><th>When</th><th>Score</th><th>Events/km</th><th>Pass</th></tr></thead>
    <tbody>{_rows(entries) or '<tr><td colspan="5" class="empty">No drives yet.</td></tr>'}</tbody>
  </table>
</div></body></html>"""
