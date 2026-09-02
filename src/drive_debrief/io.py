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
import json
import os
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from .geo import segment_distances_m


# Candidate source column names -> canonical name. Matching is done on a
# lowercased, stripped version of the header so units/casing don't matter.
_ALIASES = {
    "t": ["t", "time (s)", "time", "seconds_elapsed",
          "locationtimestamp_since1970(s)", "locationtimestamp_since1970", "loggingtime(txt)"],
    "lat": ["lat", "latitude", "latitude (deg)", "latitude (°)",
            "locationlatitude(wgs84)", "locationlatitude"],
    "lon": ["lon", "lng", "longitude", "longitude (deg)", "longitude (°)",
            "locationlongitude(wgs84)", "locationlongitude"],
    "speed": ["speed", "speed_mps", "velocity (m/s)", "velocity",
              "locationspeed(m/s)", "locationspeed"],
    "course": [
        "course", "heading", "bearing", "direction",
        "direction (deg)", "direction (°)",
        "locationcourse(°)", "locationcourse", "locationtrueheading(°)", "locationtrueheading",
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


def _canonical_from_raw(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Resolve aliased columns from any tabular source into canonical form."""
    resolved = _resolve(raw.columns)
    missing = [c for c in ("t", "lat", "lon") if c not in resolved]
    if missing:
        raise ValueError(
            f"{source}: could not find columns {missing}. "
            f"Saw headers: {list(raw.columns)}"
        )
    df = pd.DataFrame({
        "t": pd.to_numeric(raw[resolved["t"]], errors="coerce"),
        "lat": pd.to_numeric(raw[resolved["lat"]], errors="coerce"),
        "lon": pd.to_numeric(raw[resolved["lon"]], errors="coerce"),
    })
    if "speed" in resolved:
        df["speed"] = pd.to_numeric(raw[resolved["speed"]], errors="coerce")
    if "course" in resolved:
        course = pd.to_numeric(raw[resolved["course"]], errors="coerce")
        df["course"] = course.ffill().bfill()
    return _finalise(df, source)


def load_track_csv(path: str) -> pd.DataFrame:
    """Read a CSV (our schema or a phyphox / SensorLog export) into canonical form."""
    return _canonical_from_raw(pd.read_csv(path), path)


def load_sensorlog_records(records, source: str = "sensorlog") -> pd.DataFrame:
    """Parse SensorLog JSON (a list of per-sample dicts) into canonical form.

    SensorLog's HTTP streaming/upload sends JSON rows with keys like
    ``locationLatitude`` / ``locationSpeed`` / ``locationCourse`` — the same
    fields as its CSV, so the alias resolver handles them.
    """
    if not records:
        raise ValueError(f"{source}: no records to parse")
    return _canonical_from_raw(pd.DataFrame(list(records)), source)


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
    """Load a drive from CSV, GPX, KML/KMZ, or a Google Takeout JSON."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gpx":
        return load_gpx(path)
    if ext in (".kml", ".kmz"):
        return load_kml(path)
    if ext == ".json":
        return load_json(path)
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


# --------------------------------------------------------------------------
# Google location data (Takeout / Timeline) and KML/KMZ
#
# Google has no public API for Location History, so "connecting" means
# uploading a Takeout export. These are usually multi-trip *histories*, so we
# extract the candidate drives and keep the single longest one.
# --------------------------------------------------------------------------

_DRIVING = {"IN_PASSENGER_VEHICLE", "IN_VEHICLE", "DRIVING", "MOTORCYCLING"}


def _e7(v) -> float:
    return float(v) / 1e7


def _parse_ts(value):
    """Epoch seconds from a Google timestamp (ms-epoch or ISO-8601)."""
    if value is None:
        return None
    s = str(value).strip()
    if s.isdigit():  # milliseconds since epoch
        return int(s) / 1000.0
    return _parse_gpx_time(s)  # ISO-8601 (handles trailing 'Z')


def _track_distance_m(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    return float(np.sum(segment_distances_m(df["lat"].to_numpy(float), df["lon"].to_numpy(float))))


def _clean_optional(df: pd.DataFrame) -> pd.DataFrame:
    """Drop a partial speed column; forward/back-fill course."""
    if "speed" in df.columns and df["speed"].isna().any():
        df = df.drop(columns=["speed"])
    if "course" in df.columns:
        df["course"] = df["course"].ffill().bfill()
        if df["course"].isna().all():
            df = df.drop(columns=["course"])
    return df


def _pick_longest(tracks, source: str) -> pd.DataFrame:
    usable = [t for t in tracks if len(t) >= 3]
    if not usable:
        raise ValueError(f"{source}: no usable drive found (need a segment with 3+ points)")
    best = max(usable, key=_track_distance_m)
    return _finalise(_clean_optional(best), source)


def _split_on_gaps(rows: list, gap_s: float = 300.0) -> list:
    """Split a continuous point list into trips on time gaps."""
    df = pd.DataFrame(rows).dropna(subset=["t", "lat", "lon"]).sort_values("t").reset_index(drop=True)
    if len(df) < 3:
        return [df]
    breaks = np.where(np.diff(df["t"].to_numpy(float)) > gap_s)[0] + 1
    return [df.iloc[p].reset_index(drop=True) for p in np.split(np.arange(len(df)), breaks)]


def _from_records(locations: list) -> pd.DataFrame:
    """Raw Location History (`Records.json` -> `locations`)."""
    rows = []
    for L in locations:
        if "latitudeE7" not in L or "longitudeE7" not in L:
            continue
        row = {
            "t": _parse_ts(L.get("timestampMs") or L.get("timestamp")),
            "lat": _e7(L["latitudeE7"]),
            "lon": _e7(L["longitudeE7"]),
        }
        if "velocity" in L and L["velocity"] is not None:
            row["speed"] = float(L["velocity"])
        if "heading" in L and L["heading"] is not None:
            row["course"] = float(L["heading"])
        rows.append(row)
    return _pick_longest(_split_on_gaps(rows), "Google records")


def _seg_points(seg: dict) -> list:
    """Extract timestamped points from one semantic activitySegment."""
    raw = (seg.get("simplifiedRawPath") or {}).get("points") or []
    if raw:
        pts = []
        for p in raw:
            if "latE7" not in p:
                continue
            pts.append({"t": _parse_ts(p.get("timestampMs") or p.get("timestamp")),
                        "lat": _e7(p["latE7"]), "lon": _e7(p["lngE7"])})
        return pts
    # Fall back to the waypoint path, spacing times evenly across the segment.
    wps = (seg.get("waypointPath") or {}).get("waypoints") or []
    dur = seg.get("duration") or {}
    t0 = _parse_ts(dur.get("startTimestamp")) or 0.0
    t1 = _parse_ts(dur.get("endTimestamp")) or (t0 + len(wps))
    if len(wps) < 2:
        return []
    times = np.linspace(t0, t1, len(wps))
    return [{"t": float(times[i]), "lat": _e7(w["latE7"]), "lon": _e7(w["lngE7"])}
            for i, w in enumerate(wps) if "latE7" in w]


def _from_semantic(objects: list) -> pd.DataFrame:
    """Semantic Location History (`timelineObjects`)."""
    driving, other = [], []
    for o in objects:
        seg = o.get("activitySegment")
        if not seg:
            continue
        pts = _seg_points(seg)
        if len(pts) < 3:
            continue
        df = pd.DataFrame(pts)
        (driving if seg.get("activityType") in _DRIVING else other).append(df)
    return _pick_longest(driving or other, "Google semantic history")


def _parse_latlng_str(s: str):
    """'51.5327°, -0.1050°' -> (51.5327, -0.1050)."""
    parts = s.replace("°", "").split(",")
    return float(parts[0]), float(parts[1])


def _from_new_timeline(segments: list) -> pd.DataFrame:
    """On-device Timeline export (`semanticSegments` -> `timelinePath`)."""
    tracks = []
    for s in segments:
        path = s.get("timelinePath")
        if not path:
            continue
        base = _parse_ts(s.get("startTime")) or 0.0
        pts = []
        for p in path:
            try:
                lat, lon = _parse_latlng_str(p["point"])
            except (KeyError, ValueError, IndexError):
                continue
            if "time" in p:
                t = _parse_ts(p["time"])
            else:
                t = base + float(p.get("durationMinutesOffsetFromStartTime", 0)) * 60.0
            pts.append({"t": t, "lat": lat, "lon": lon})
        if len(pts) >= 3:
            tracks.append(pd.DataFrame(pts))
    return _pick_longest(tracks, "Google timeline")


def load_google_json(path: str) -> pd.DataFrame:
    """Load a Google Takeout / Timeline JSON, returning the longest drive."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        if "locations" in data:
            return _from_records(data["locations"])
        if "timelineObjects" in data:
            return _from_semantic(data["timelineObjects"])
        if "semanticSegments" in data:
            return _from_new_timeline(data["semanticSegments"])
    raise ValueError(
        f"{path}: unrecognised Google location JSON "
        "(expected 'locations', 'timelineObjects', or 'semanticSegments')."
    )


def load_json(path: str) -> pd.DataFrame:
    """Dispatch a .json file to the Google or SensorLog parser by shape."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and any(k in data for k in ("locations", "timelineObjects", "semanticSegments")):
        return load_google_json(path)
    if isinstance(data, list):
        return load_sensorlog_records(data, path)
    if isinstance(data, dict):
        for key in ("data", "rows", "samples", "records"):
            if isinstance(data.get(key), list):
                return load_sensorlog_records(data[key], path)
    raise ValueError(f"{path}: unrecognised JSON (not a Google export or a SensorLog log)")


def _read_kml_text(path: str) -> str:
    if path.lower().endswith(".kmz"):
        with zipfile.ZipFile(path) as zf:
            name = next((n for n in zf.namelist() if n.lower().endswith(".kml")), None)
            if name is None:
                raise ValueError(f"{path}: no .kml inside the .kmz")
            return zf.read(name).decode("utf-8", errors="replace")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def load_kml(path: str) -> pd.DataFrame:
    """Parse a KML/KMZ track (Google Timeline day export, My Maps, GPS loggers).

    Prefers ``<gx:Track>`` (has timestamps); falls back to ``<LineString>``
    coordinates with a synthesised 1 Hz clock.
    """
    root = ET.fromstring(_read_kml_text(path))
    tracks = []
    for el in root.iter():
        if _local(el.tag) != "Track":
            continue
        whens, coords = [], []
        for c in el.iter():
            lt = _local(c.tag)
            if lt == "when" and c.text:
                whens.append(_parse_ts(c.text))
            elif lt == "coord" and c.text:
                lon, lat = c.text.split()[:2]
                coords.append((float(lat), float(lon)))
        n = min(len(whens), len(coords)) if whens else len(coords)
        if n < 3:
            continue
        times = whens[:n] if whens else list(range(n))
        df = pd.DataFrame({"t": times,
                           "lat": [c[0] for c in coords[:n]],
                           "lon": [c[1] for c in coords[:n]]})
        tracks.append(df)

    if not tracks:  # LineString fallback (no timestamps)
        for el in root.iter():
            if _local(el.tag) != "coordinates" or not el.text:
                continue
            pts = []
            for tok in el.text.split():
                bits = tok.split(",")
                if len(bits) >= 2:
                    pts.append((float(bits[1]), float(bits[0])))
            if len(pts) >= 3:
                tracks.append(pd.DataFrame({"t": range(len(pts)),
                                            "lat": [p[0] for p in pts],
                                            "lon": [p[1] for p in pts]}))
    return _pick_longest(tracks, path)


def to_kml(df: pd.DataFrame, base_iso: str = "2026-01-01T00:00:00+00:00") -> str:
    """Serialise a canonical DataFrame to a gx:Track KML string (for demos/tests)."""
    base = datetime.datetime.fromisoformat(base_iso)
    rows = ['<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">',
            "<Document><Placemark><gx:Track>"]
    for r in df.itertuples(index=False):
        stamp = (base + datetime.timedelta(seconds=float(r.t))).isoformat().replace("+00:00", "Z")
        rows.append(f"<when>{stamp}</when>")
        rows.append(f"<gx:coord>{r.lon:.7f} {r.lat:.7f} 0</gx:coord>")
    rows.append("</gx:Track></Placemark></Document></kml>")
    return "\n".join(rows)


def to_google_records(df: pd.DataFrame, base_iso: str = "2026-01-01T00:00:00+00:00") -> str:
    """Serialise to a Google 'Records.json'-style string (for demos/tests)."""
    base = datetime.datetime.fromisoformat(base_iso)
    has_speed, has_course = "speed" in df.columns, "course" in df.columns
    locs = []
    for r in df.itertuples(index=False):
        ms = int((base + datetime.timedelta(seconds=float(r.t))).timestamp() * 1000)
        loc = {"latitudeE7": int(round(r.lat * 1e7)),
               "longitudeE7": int(round(r.lon * 1e7)),
               "timestampMs": str(ms)}
        if has_speed:
            loc["velocity"] = int(round(float(r.speed)))
        if has_course:
            loc["heading"] = int(round(float(r.course)))
        locs.append(loc)
    return json.dumps({"locations": locs})


def to_google_semantic(df: pd.DataFrame, activity_type: str = "IN_PASSENGER_VEHICLE",
                       base_iso: str = "2026-01-01T00:00:00+00:00") -> str:
    """Serialise to a Semantic Location History string (for demos/tests)."""
    base = datetime.datetime.fromisoformat(base_iso)

    def stamp(sec):
        return (base + datetime.timedelta(seconds=float(sec))).isoformat().replace("+00:00", "Z")

    points = [
        {"latE7": int(round(r.lat * 1e7)), "lngE7": int(round(r.lon * 1e7)),
         "timestampMs": str(int((base + datetime.timedelta(seconds=float(r.t))).timestamp() * 1000))}
        for r in df.itertuples(index=False)
    ]
    obj = {"timelineObjects": [{"activitySegment": {
        "activityType": activity_type,
        "duration": {"startTimestamp": stamp(df["t"].iloc[0]),
                     "endTimestamp": stamp(df["t"].iloc[-1])},
        "simplifiedRawPath": {"points": points},
    }}]}
    return json.dumps(obj)
