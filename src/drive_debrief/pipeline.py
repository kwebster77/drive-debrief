"""End-to-end: CSV in -> HTML debrief + machine-readable summary out."""
from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

import pandas as pd

from .assessment import assess
from .events import Event, Thresholds, detect_events
from .io import load_track, normalise_dataframe
from .kinematics import build_track
from .report import build_report_html
from .scoring import Summary, summarise


def analyse_dataframe(df: pd.DataFrame, thresholds: Optional[Thresholds] = None):
    """Run the analysis on an in-memory canonical DataFrame."""
    df = normalise_dataframe(df)
    track = build_track(df)
    events = detect_events(track, thresholds or Thresholds())
    summary = summarise(track, events)
    return track, events, summary


def _events_to_dicts(events: List[Event]) -> list:
    return [asdict(e) for e in events]


def run_debrief(
    csv_path: str,
    out_html: Optional[str] = None,
    thresholds: Optional[Thresholds] = None,
    title: str = "Practice-drive debrief",
) -> dict:
    """Load a CSV, analyse it, write the HTML report, return a summary dict."""
    df = load_track(csv_path)
    track, events, summary = analyse_dataframe(df, thresholds)
    assessment = assess(events)

    report_path = None
    if out_html:
        html = build_report_html(track, events, summary, assessment, title=title)
        with open(out_html, "w", encoding="utf-8") as fh:
            fh.write(html)
        report_path = out_html

    result = {
        "summary": _summary_to_dict(summary),
        "assessment": asdict(assessment),
        "events": _events_to_dicts(events),
        "report_path": report_path,
    }
    return result


def _summary_to_dict(summary: Summary) -> dict:
    d = asdict(summary)
    return d
