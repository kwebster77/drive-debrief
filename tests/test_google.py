"""Google Takeout (records / semantic / timeline) and KML/KMZ parsing."""
import json

from drive_debrief.io import (
    load_google_json,
    load_kml,
    load_track,
    to_google_records,
    to_kml,
)
from drive_debrief.pipeline import analyse_dataframe
from drive_debrief.synth import generate_drive


def _kinds(df):
    _, events, _ = analyse_dataframe(df)
    return {e.kind for e in events}


def test_google_records_roundtrip(tmp_path):
    drive = generate_drive(noise_m=0.0)
    p = tmp_path / "Records.json"
    p.write_text(to_google_records(drive))
    df = load_google_json(str(p))
    assert {"t", "lat", "lon"}.issubset(df.columns)
    assert "harsh_braking" in _kinds(df)


def test_kml_roundtrip(tmp_path):
    drive = generate_drive(noise_m=0.0)
    p = tmp_path / "drive.kml"
    p.write_text(to_kml(drive))
    df = load_kml(str(p))
    kinds = _kinds(df)
    assert "harsh_braking" in kinds and "hard_cornering" in kinds


def test_load_track_dispatches_google_and_kml(tmp_path):
    drive = generate_drive(noise_m=0.0)
    (tmp_path / "d.json").write_text(to_google_records(drive))
    (tmp_path / "d.kml").write_text(to_kml(drive))
    assert len(load_track(str(tmp_path / "d.json"))) >= 3
    assert len(load_track(str(tmp_path / "d.kml"))) >= 3


def test_semantic_history_picks_driving_segment(tmp_path):
    drive = generate_drive(noise_m=0.0)
    base_ms = 1_700_000_000_000
    points = [
        {"latE7": int(round(r.lat * 1e7)), "lngE7": int(round(r.lon * 1e7)),
         "timestampMs": str(base_ms + int(r.t * 1000))}
        for r in drive.itertuples(index=False)
    ]
    payload = {
        "timelineObjects": [
            {"activitySegment": {  # a short walk — should be ignored
                "activityType": "WALKING",
                "simplifiedRawPath": {"points": points[:4]},
            }},
            {"activitySegment": {  # the real drive
                "activityType": "IN_PASSENGER_VEHICLE",
                "simplifiedRawPath": {"points": points},
            }},
        ]
    }
    p = tmp_path / "semantic.json"
    p.write_text(json.dumps(payload))
    df = load_google_json(str(p))
    assert len(df) == len(points)
    assert "harsh_braking" in _kinds(df)


def test_records_history_picks_longest_trip(tmp_path):
    drive = generate_drive(noise_m=0.0)
    base_ms = 1_700_000_000_000
    locs = [
        {"latitudeE7": int(round(r.lat * 1e7)), "longitudeE7": int(round(r.lon * 1e7)),
         "timestampMs": str(base_ms + int(r.t * 1000))}
        for r in drive.itertuples(index=False)
    ]
    # A tiny second "trip" far later in time (big gap) and barely moving.
    later = base_ms + 10_000_000
    locs += [
        {"latitudeE7": 515000000, "longitudeE7": -1000000, "timestampMs": str(later)},
        {"latitudeE7": 515000010, "longitudeE7": -1000000, "timestampMs": str(later + 1000)},
        {"latitudeE7": 515000020, "longitudeE7": -1000000, "timestampMs": str(later + 2000)},
    ]
    p = tmp_path / "Records.json"
    p.write_text(json.dumps({"locations": locs}))
    df = load_google_json(str(p))
    # The longest trip is the real drive, so its braking event survives.
    assert "harsh_braking" in _kinds(df)


def test_unrecognised_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"nope": []}))
    try:
        load_google_json(str(p))
        assert False, "expected ValueError"
    except ValueError:
        pass
