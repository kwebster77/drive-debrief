"""SensorLog JSON parsing and .json dispatch (Google vs SensorLog)."""
import json

from drive_debrief.io import load_json, load_sensorlog_records, to_google_records
from drive_debrief.pipeline import analyse_dataframe
from drive_debrief.synth import generate_drive


def _sensorlog_rows(df):
    # Mimic SensorLog's per-sample JSON row keys.
    return [
        {"locationTimestamp_since1970": r.t,
         "locationLatitude": r.lat, "locationLongitude": r.lon,
         "locationSpeed": r.speed, "locationCourse": r.course}
        for r in df.itertuples(index=False)
    ]


def test_sensorlog_records_parse_and_analyse():
    df = generate_drive(noise_m=0.0)
    got = load_sensorlog_records(_sensorlog_rows(df))
    assert {"t", "lat", "lon", "speed", "course"}.issubset(got.columns)
    _, events, _ = analyse_dataframe(got)
    assert "harsh_braking" in {e.kind for e in events}


def test_load_json_dispatches_sensorlog(tmp_path):
    df = generate_drive(noise_m=0.0)
    p = tmp_path / "log.json"
    p.write_text(json.dumps(_sensorlog_rows(df)))
    assert len(load_json(str(p))) >= 3


def test_load_json_still_dispatches_google(tmp_path):
    df = generate_drive(noise_m=0.0)
    p = tmp_path / "g.json"
    p.write_text(to_google_records(df))
    assert len(load_json(str(p))) >= 3


def test_sensorlog_wrapped_in_data_key(tmp_path):
    df = generate_drive(noise_m=0.0)
    p = tmp_path / "wrapped.json"
    p.write_text(json.dumps({"data": _sensorlog_rows(df)}))
    assert len(load_json(str(p))) >= 3


def test_empty_records_raises():
    try:
        load_sensorlog_records([])
        assert False, "expected ValueError"
    except ValueError:
        pass
