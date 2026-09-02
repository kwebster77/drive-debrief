"""The DVSA-style verdict should follow the fault-count rules."""
from drive_debrief.assessment import assess
from drive_debrief.events import Event


def _event(kind, peak, t=10.0):
    return Event(
        kind=kind, label=kind, t_start=t, t_end=t, t_peak=t,
        peak_value=peak, unit="g", severity="minor", detail="x",
    )


def test_clean_drive_passes():
    a = assess([])
    assert a.passed
    assert a.minors == a.serious == a.dangerous == 0
    assert "Clean pass" in a.verdict


def test_ordinary_harsh_events_are_minor_faults():
    a = assess([_event("harsh_braking", 0.4), _event("hard_cornering", 0.4)])
    assert a.minors == 2 and a.serious == 0 and a.dangerous == 0
    assert a.passed


def test_severe_brake_is_serious_and_fails():
    a = assess([_event("harsh_braking", 0.6)])
    assert a.serious == 1
    assert not a.passed


def test_extreme_corner_is_dangerous_and_fails():
    a = assess([_event("hard_cornering", 0.8)])
    assert a.dangerous == 1
    assert not a.passed


def test_too_many_minors_fails():
    a = assess([_event("harsh_acceleration", 0.35) for _ in range(16)])
    assert a.serious == 0 and a.dangerous == 0
    assert not a.passed
    assert "16" in a.verdict


def test_stops_are_not_faults():
    a = assess([_event("stop", 5.0)])
    assert a.minors == 0 and a.passed
