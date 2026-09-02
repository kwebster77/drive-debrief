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

import datetime
import os
import xml.etree.ElementTree as ET

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

    return _finalise(df, path)


def _finalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Drop bad rows, sort by time, zero the clock, sanity-check length."""
    df = df.dropna(subset=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)
    if len(df):
        df["t"] = df["t"] - df["t"].iloc[0]
    if len(df) < 3:
        raise ValueError(f"{source}: need at least 3 valid samples, got {len(df)}")
    return df


def _local(tag: str) -> str:
    """Strip the XML namespace, leaving the local tag name."""
    return tag.rsplit("}", 1)[-1]


def _parse_gpx_time(text: str):
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def load_gpx(path: str) -> pd.DataFrame:
    """Parse a GPX track (Strava, dashcams, most GPS apps).

    Namespace-agnostic: we match on local tag names so the many GPX
    dialects all parse. Uses <time> for the clock (falling back to a 1 Hz
    index if absent) and picks up <speed>/<course> when present.
    """
    root = ET.parse(path).getroot()
    times, lats, lons, speeds, courses = [], [], [], [], []
    for el in root.iter():
        if _local(el.tag) not in ("trkpt", "rtept", "wpt"):
            continue
        if "lat" not in el.attrib or "lon" not in el.attrib:
            continue
        tt = spd = crs = None
        for child in el.iter():
            lt = _local(child.tag)
            if child.text is None:
                continue
            if lt == "time":
                tt = _parse_gpx_time(child.text)
            elif lt == "speed":
                try:
                    spd = float(child.text)
                except ValueError:
                    pass
            elif lt in ("course", "bearing"):
                try:
                    crs = float(child.text)
                except ValueError:
                    pass
        lats.append(float(el.attrib["lat"]))
        lons.append(float(el.attrib["lon"]))
        times.append(tt)
        speeds.append(spd)
        courses.append(crs)

    n = len(lats)
    if n < 3:
        raise ValueError(f"{path}: found {n} track points, need at least 3")

    # No timestamps? Assume an even 1 Hz cadence.
    if any(t is None for t in times):
        times = list(range(n))

    df = pd.DataFrame({"t": times, "lat": lats, "lon": lons})
    if all(s is not None for s in speeds):
        df["speed"] = speeds
    if all(c is not None for c in courses):
        df["course"] = courses
    return _finalise(df, path)


def load_track(path: str) -> pd.DataFrame:
    """Load a drive from CSV or GPX, dispatched by file extension."""
    if os.path.splitext(path)[1].lower() == ".gpx":
        return load_gpx(path)
    return load_track_csv(path)


def to_gpx(df: pd.DataFrame, base_iso: str = "2026-01-01T00:00:00+00:00") -> str:
    """Serialise a canonical DataFrame to a GPX 1.1 string.

    Timestamps are the drive's ``t`` offsets from a fixed base (the base is
    arbitrary — only the intervals matter to the analysis).
    """
    base = datetime.datetime.fromisoformat(base_iso)
    has_speed = "speed" in df.columns
    has_course = "course" in df.columns
    rows = ['<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="drive-debrief" xmlns="http://www.topografix.com/GPX/1/1">',
            "<trk><name>drive</name><trkseg>"]
    for r in df.itertuples(index=False):
        stamp = (base + datetime.timedelta(seconds=float(r.t))).isoformat().replace("+00:00", "Z")
        extra = f"<time>{stamp}</time>"
        if has_speed:
            extra += f"<speed>{float(r.speed):.3f}</speed>"
        if has_course:
            extra += f"<course>{float(r.course):.2f}</course>"
        rows.append(f'<trkpt lat="{r.lat:.7f}" lon="{r.lon:.7f}">{extra}</trkpt>')
    rows.append("</trkseg></trk></gpx>")
    return "\n".join(rows)


def normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Same normalisation as ``load_track_csv`` for an in-memory frame."""
    out = df.copy()
    out = out.dropna(subset=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)
    out["t"] = out["t"] - out["t"].iloc[0]
    return out
