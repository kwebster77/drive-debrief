"""CSV/GPX loading, format dispatch, and GPX round-trip."""
import pandas as pd
import pytest

from drive_debrief.io import load_gpx, load_track, load_track_csv, to_gpx
from drive_debrief.pipeline import analyse_dataframe
from drive_debrief.synth import generate_drive


def test_csv_alias_headers(tmp_path):
    # phyphox-style headers should map to canonical columns.
    p = tmp_path / "d.csv"
    p.write_text(
        "Time (s),Latitude (°),Longitude (°),Velocity (m/s)\n"
        "0,51.5,-0.1,0\n1,51.5001,-0.1,2.5\n2,51.5002,-0.1,5.0\n"
    )
    df = load_track_csv(str(p))
    assert list(df.columns[:3]) == ["t", "lat", "lon"]
    assert "speed" in df.columns
    assert df["t"].iloc[0] == 0


def test_gpx_roundtrip_matches_csv(tmp_path):
    drive = generate_drive(noise_m=0.0)
    gpx = tmp_path / "d.gpx"
    gpx.write_text(to_gpx(drive))

    gdf = load_gpx(str(gpx))
    assert {"t", "lat", "lon"}.issubset(gdf.columns)
    assert "speed" in gdf.columns and "course" in gdf.columns
    assert len(gdf) == len(drive)
    # Same events should be recovered from the GPX as from the source frame.
    _, events_src, _ = analyse_dataframe(drive)
    _, events_gpx, _ = analyse_dataframe(gdf)
    kinds_src = sorted(e.kind for e in events_src)
    kinds_gpx = sorted(e.kind for e in events_gpx)
    assert kinds_src == kinds_gpx


def test_load_track_dispatches_by_extension(tmp_path):
    drive = generate_drive(noise_m=0.0)
    (tmp_path / "d.gpx").write_text(to_gpx(drive))
    df = load_track(str(tmp_path / "d.gpx"))
    assert len(df) == len(drive)


def test_gpx_without_time_falls_back_to_1hz(tmp_path):
    gpx = tmp_path / "notime.gpx"
    gpx.write_text(
        '<?xml version="1.0"?>\n<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
        "<trk><trkseg>"
        '<trkpt lat="51.5" lon="-0.10"></trkpt>'
        '<trkpt lat="51.5" lon="-0.099"></trkpt>'
        '<trkpt lat="51.5" lon="-0.098"></trkpt>'
        "</trkseg></trk></gpx>"
    )
    df = load_gpx(str(gpx))
    assert list(df["t"]) == [0, 1, 2]


def test_too_few_points_raises(tmp_path):
    gpx = tmp_path / "short.gpx"
    gpx.write_text(
        '<?xml version="1.0"?>\n<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
        '<trk><trkseg><trkpt lat="51.5" lon="-0.1"></trkpt></trkseg></trk></gpx>'
    )
    with pytest.raises(ValueError):
        load_gpx(str(gpx))
