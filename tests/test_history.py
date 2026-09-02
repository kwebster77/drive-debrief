"""History persistence + progress rendering."""
from drive_debrief.history import append_run, build_entry, load_history
from drive_debrief.progress import build_progress_html


def _result(score, harsh, dist=1.0):
    return {
        "summary": {
            "score": score, "grade": "B", "distance_km": dist, "duration_min": 5.0,
            "counts": {"harsh_braking": harsh},
        },
        "assessment": {"minors": harsh, "serious": 0, "dangerous": 0, "passed": True},
    }


def test_build_entry_computes_events_per_km():
    e = build_entry("drive1", "2026-09-02T10:00:00", _result(80, 4, dist=2.0))
    assert e["score"] == 80
    assert e["harsh_events"] == 4
    assert e["events_per_km"] == 2.0


def test_append_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "hist.json")
    assert load_history(path) == []
    append_run(build_entry("a", "t1", _result(70, 6)), path)
    append_run(build_entry("b", "t2", _result(85, 2)), path)
    hist = load_history(path)
    assert len(hist) == 2
    assert [h["score"] for h in hist] == [70, 85]


def test_load_missing_file_is_empty(tmp_path):
    assert load_history(str(tmp_path / "nope.json")) == []


def test_progress_html_renders_scores(tmp_path):
    entries = [
        build_entry("a", "t1", _result(70, 6)),
        build_entry("b", "t2", _result(85, 2)),
    ]
    html = build_progress_html(entries)
    assert "Driving progress" in html
    assert "<svg" in html
    assert "2 drive(s)" in html
