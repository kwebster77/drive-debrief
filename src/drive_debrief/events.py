"""Detect coaching events from a Track.

Each detector finds contiguous runs where a signal crosses a threshold
and collapses each run into a single Event carrying the peak value and a
learner-friendly description. Thresholds are in g so they read the way
telematics/insurers talk about them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .kinematics import G, Track


@dataclass
class Event:
    kind: str            # e.g. "harsh_braking"
    label: str           # human label, e.g. "Harsh braking"
    t_start: float
    t_end: float
    t_peak: float
    peak_value: float    # in the event's natural unit
    unit: str
    severity: str        # "minor" | "moderate" | "severe"
    detail: str
    meta: dict = field(default_factory=dict)


@dataclass
class Thresholds:
    brake_g: float = 0.35
    accel_g: float = 0.30
    lateral_g: float = 0.35
    stop_speed_mps: float = 0.5      # below this counts as stopped
    stop_min_s: float = 3.0          # sustained this long to count as a stop
    long_stop_s: float = 25.0        # a stop longer than this is flagged
    accel_min_s: float = 0.6         # brake/accel must persist this long
    corner_min_s: float = 1.5        # a corner must persist this long
    merge_gap_s: float = 1.5         # merge same-kind events closer than this


def _duration(track, a: int, b: int) -> float:
    # A single-sample run still spans ~one sample interval.
    return float(track.t[b] - track.t[a]) + track.dt


def _runs(mask: np.ndarray):
    """Yield (start_idx, end_idx) inclusive for each True run in mask."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(g[0], g[-1]) for g in np.split(idx, splits)]


def _severity(ratio: float) -> str:
    """ratio = peak / threshold."""
    if ratio >= 1.6:
        return "severe"
    if ratio >= 1.25:
        return "moderate"
    return "minor"


def _accel_events(track: Track, thr: Thresholds) -> List[Event]:
    events: List[Event] = []
    thr_ms2 = thr.brake_g * G
    for a, b in _runs(track.a_long < -thr_ms2):
        if _duration(track, a, b) < thr.accel_min_s:
            continue
        seg = track.a_long[a : b + 1]
        k = a + int(np.argmin(seg))
        peak_g = -track.a_long[k] / G
        events.append(
            Event(
                kind="harsh_braking",
                label="Harsh braking",
                t_start=float(track.t[a]),
                t_end=float(track.t[b]),
                t_peak=float(track.t[k]),
                peak_value=round(peak_g, 2),
                unit="g",
                severity=_severity(peak_g / thr.brake_g),
                detail=(
                    f"Braked at {peak_g:.2f}g "
                    f"(from {track.speed[a] * 2.237:.0f} to {track.speed[b] * 2.237:.0f} mph). "
                    "Look further ahead and ease off the gas earlier."
                ),
            )
        )

    thr_ms2 = thr.accel_g * G
    for a, b in _runs(track.a_long > thr_ms2):
        if _duration(track, a, b) < thr.accel_min_s:
            continue
        seg = track.a_long[a : b + 1]
        k = a + int(np.argmax(seg))
        peak_g = track.a_long[k] / G
        events.append(
            Event(
                kind="harsh_acceleration",
                label="Harsh acceleration",
                t_start=float(track.t[a]),
                t_end=float(track.t[b]),
                t_peak=float(track.t[k]),
                peak_value=round(peak_g, 2),
                unit="g",
                severity=_severity(peak_g / thr.accel_g),
                detail=f"Accelerated at {peak_g:.2f}g. Squeeze the pedal more gently for a smoother ride.",
            )
        )
    return events


def _corner_events(track: Track, thr: Thresholds) -> List[Event]:
    events: List[Event] = []
    thr_ms2 = thr.lateral_g * G
    for a, b in _runs(np.abs(track.a_lat) > thr_ms2):
        if _duration(track, a, b) < thr.corner_min_s:
            continue
        seg = np.abs(track.a_lat[a : b + 1])
        k = a + int(np.argmax(seg))
        peak_g = abs(track.a_lat[k]) / G
        side = "left" if track.a_lat[k] > 0 else "right"
        events.append(
            Event(
                kind="hard_cornering",
                label="Hard cornering",
                t_start=float(track.t[a]),
                t_end=float(track.t[b]),
                t_peak=float(track.t[k]),
                peak_value=round(peak_g, 2),
                unit="g",
                severity=_severity(peak_g / thr.lateral_g),
                detail=(
                    f"Took a {side} bend at {peak_g:.2f}g "
                    f"({track.speed[k] * 2.237:.0f} mph). Slow down more before turning in."
                ),
                meta={"side": side},
            )
        )
    return events


def _stop_events(track: Track, thr: Thresholds) -> List[Event]:
    events: List[Event] = []
    for a, b in _runs(track.speed < thr.stop_speed_mps):
        dur = float(track.t[b] - track.t[a])
        if dur < thr.stop_min_s:
            continue
        kind = "long_stop" if dur >= thr.long_stop_s else "stop"
        label = "Long stop" if kind == "long_stop" else "Stop"
        detail = (
            f"Stationary for {dur:.0f}s."
            + (" Long hold-ups can signal hesitation at a junction." if kind == "long_stop" else "")
        )
        events.append(
            Event(
                kind=kind,
                label=label,
                t_start=float(track.t[a]),
                t_end=float(track.t[b]),
                t_peak=float(track.t[a]),
                peak_value=round(dur, 1),
                unit="s",
                severity="moderate" if kind == "long_stop" else "minor",
                detail=detail,
            )
        )
    return events


def _merge_nearby(events: List[Event], gap_s: float) -> List[Event]:
    """Collapse same-kind events separated by less than ``gap_s`` into the
    stronger one, so one messy manoeuvre reads as a single note."""
    by_kind = {}
    for e in events:
        by_kind.setdefault(e.kind, []).append(e)

    merged: List[Event] = []
    for kind, group in by_kind.items():
        group.sort(key=lambda e: e.t_start)
        cur = group[0]
        for nxt in group[1:]:
            if nxt.t_start - cur.t_end <= gap_s:
                cur = cur if abs(cur.peak_value) >= abs(nxt.peak_value) else nxt
                cur.t_start = min(cur.t_start, nxt.t_start)
                cur.t_end = max(cur.t_end, nxt.t_end)
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
    merged.sort(key=lambda e: e.t_peak)
    return merged


def detect_events(track: Track, thr: Thresholds = None) -> List[Event]:
    """Run all detectors and return events sorted by time."""
    thr = thr or Thresholds()
    events: List[Event] = []
    events += _accel_events(track, thr)
    events += _corner_events(track, thr)
    events += _stop_events(track, thr)
    events = _merge_nearby(events, thr.merge_gap_s)
    return events
