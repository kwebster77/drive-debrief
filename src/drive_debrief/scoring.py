"""Turn a Track + events into a smoothness score and summary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .events import Event
from .kinematics import Track

# Points deducted per event by severity.
_PENALTY = {"minor": 3, "moderate": 7, "severe": 14}


@dataclass
class Summary:
    score: int
    grade: str
    distance_km: float
    duration_min: float
    max_speed_mph: float
    avg_speed_mph: float
    counts: Dict[str, int]
    rms_jerk: float


def _grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 55:
        return "D"
    return "E"


def summarise(track: Track, events: List[Event]) -> Summary:
    counts: Dict[str, int] = {}
    penalty = 0
    for e in events:
        counts[e.kind] = counts.get(e.kind, 0) + 1
        # Stops aren't a fault in themselves; only long stops carry a small penalty.
        if e.kind in ("stop",):
            continue
        penalty += _PENALTY.get(e.severity, 5)

    score = int(max(0, min(100, 100 - penalty)))

    speed_mph = track.speed * 2.23694
    rms_jerk = float(np.sqrt(np.mean(track.jerk ** 2))) if len(track) else 0.0

    return Summary(
        score=score,
        grade=_grade(score),
        distance_km=round(track.distance_m / 1000.0, 2),
        duration_min=round(track.duration_s / 60.0, 1),
        max_speed_mph=round(float(np.max(speed_mph)) if len(track) else 0.0, 0),
        avg_speed_mph=round(float(np.mean(speed_mph)) if len(track) else 0.0, 0),
        counts=counts,
        rms_jerk=round(rms_jerk, 2),
    )
