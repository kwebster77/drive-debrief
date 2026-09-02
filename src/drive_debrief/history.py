"""Persist a summary of each drive so we can show improvement over time.

Storage is a plain JSON list on disk — no database, no network, so it
works unchanged in a sandbox. Each entry is one drive.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

DEFAULT_HISTORY = "drive_history.json"


def load_history(path: str = DEFAULT_HISTORY) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def append_run(entry: Dict, path: str = DEFAULT_HISTORY) -> List[Dict]:
    history = load_history(path)
    history.append(entry)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    return history


def build_entry(label: str, timestamp: str, result: Dict) -> Dict:
    """Distil a pipeline result into a compact, comparable history entry."""
    s = result["summary"]
    a = result["assessment"]
    counts = s.get("counts", {})
    harsh = (
        counts.get("harsh_braking", 0)
        + counts.get("harsh_acceleration", 0)
        + counts.get("hard_cornering", 0)
    )
    dist = s.get("distance_km") or 0.0
    per_km = round(harsh / dist, 2) if dist > 0 else 0.0
    return {
        "label": label,
        "timestamp": timestamp,
        "score": s["score"],
        "grade": s["grade"],
        "distance_km": s["distance_km"],
        "duration_min": s["duration_min"],
        "harsh_events": harsh,
        "events_per_km": per_km,
        "minors": a["minors"],
        "serious": a["serious"],
        "dangerous": a["dangerous"],
        "passed": a["passed"],
    }
