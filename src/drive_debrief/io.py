"""Load a recorded drive into a canonical DataFrame.

Canonical columns: ``t`` (seconds, monotonic), ``lat``, ``lon`` and an
optional ``speed`` (m/s). We accept our own simple CSV plus the exports
from two common free phone sensor-logger apps:

* phyphox "Location (GPS)"  -> "Time (s)","Latitude (deg)",...,"Velocity (m/s)"
* SensorLog                 -> "locationTimestamp_since1970(s)","locationLatitude(WGS84)",...

The loader is deliberately forgiving about column naming so a real
export can be dropped in with no editing.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


# Candidate source column names -> canonical name. Matching is done on a
# lowercased, stripped version of the header so units/casing don't matter.
_ALIASES = {
    "t": ["t", "time (s)", "time", "seconds_elapsed", "locationtimestamp_since1970(s)"],
    "lat": ["lat", "latitude", "latitude (deg)", "latitude (°)", "locationlatitude(wgs84)"],
    "lon": ["lon", "lng", "longitude", "longitude (deg)", "longitude (°)", "locationlongitude(wgs84)"],
    "speed": ["speed", "speed_mps", "velocity (m/s)", "velocity", "locationspeed(m/s)"],
    "course": [
        "course", "heading", "bearing", "direction",
        "direction (deg)", "direction (°)",
        "locationcourse(°)", "locationcourse", "locationtrueheading(°)",
    ],
}


def _resolve(columns) -> dict:
    lower = {str(c).strip().lower(): c for c in columns}
    resolved = {}
    for canonical, options in _ALIASES.items():
        for opt in options:
            if opt in lower:
                resolved[canonical] = lower[opt]
                break
    return resolved


def load_track_csv(path: str) -> pd.DataFrame:
    """Read ``path`` and return a canonical DataFrame sorted by time.

    Raises ValueError with an actionable message if required columns are
    missing, so a bad export fails loudly instead of silently mis-parsing.
    """
    raw = pd.read_csv(path)
    resolved = _resolve(raw.columns)

    missing = [c for c in ("t", "lat", "lon") if c not in resolved]
    if missing:
        raise ValueError(
            f"{path}: could not find columns {missing}. "
            f"Saw headers: {list(raw.columns)}"
        )

    df = pd.DataFrame(
        {
            "t": pd.to_numeric(raw[resolved["t"]], errors="coerce"),
            "lat": pd.to_numeric(raw[resolved["lat"]], errors="coerce"),
            "lon": pd.to_numeric(raw[resolved["lon"]], errors="coerce"),
        }
    )
    if "speed" in resolved:
        df["speed"] = pd.to_numeric(raw[resolved["speed"]], errors="coerce")
    if "course" in resolved:
        # Course is often NaN while stationary; fill so it stays usable.
        course = pd.to_numeric(raw[resolved["course"]], errors="coerce")
        df["course"] = course.ffill().bfill()

    df = df.dropna(subset=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)

    # Normalise time so the first sample is t=0 (absolute epochs are common).
    if len(df):
        df["t"] = df["t"] - df["t"].iloc[0]

    if len(df) < 3:
        raise ValueError(f"{path}: need at least 3 valid samples, got {len(df)}")

    return df


def normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Same normalisation as ``load_track_csv`` for an in-memory frame."""
    out = df.copy()
    out = out.dropna(subset=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)
    out["t"] = out["t"] - out["t"].iloc[0]
    return out
