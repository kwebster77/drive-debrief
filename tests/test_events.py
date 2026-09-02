"""Detectors should find the events we deliberately built into the drive."""
import numpy as np

from drive_debrief.events import Thresholds, detect_events
from drive_debrief.kinematics import build_track
from drive_debrief.pipeline import analyse_dataframe
from drive_debrief.synth import generate_drive


def _events_of(kind, events):
    return [e for e in events if e.kind == kind]


def test_harsh_brake_detected_at_expected_time():
    df = generate_drive(noise_m=0.0)
    track = build_track(df)
    events = detect_events(track)

    brakes = _events_of("harsh_braking", events)
    assert len(brakes) >= 1, "expected the built-in harsh brake to be flagged"
    peak = max(brakes, key=lambda e: e.peak_value)
    assert 19.0 <= peak.t_peak <= 24.0          # brake happens ~20-22.5s
    assert peak.peak_value >= 0.40              # designed ~0.55g


def test_hard_corner_detected():
    df = generate_drive(noise_m=0.0)
    _, events, _ = analyse_dataframe(df)
    corners = _events_of("hard_cornering", events)
    assert len(corners) >= 1
    assert 36.0 <= corners[0].t_peak <= 42.0    # bend is ~37-41s


def test_harsh_accel_detected():
    df = generate_drive(noise_m=0.0)
    _, events, _ = analyse_dataframe(df)
    assert len(_events_of("harsh_acceleration", events)) >= 1


def test_gentle_stop_not_flagged_as_harsh():
    # The final 0.28g stop is below threshold and must not be a harsh brake.
    df = generate_drive(noise_m=0.0)
    _, events, _ = analyse_dataframe(df)
    late_brakes = [e for e in events if e.kind == "harsh_braking" and e.t_peak > 44.0]
    assert late_brakes == []


def test_smooth_drive_scores_higher_than_erratic():
    smooth = generate_drive(noise_m=0.0)
    _, _, smooth_summary = analyse_dataframe(smooth)
    # A tighter threshold set flags more -> should not score *higher*.
    _, _, strict_summary = analyse_dataframe(
        smooth, Thresholds(brake_g=0.2, accel_g=0.2, lateral_g=0.2)
    )
    assert smooth_summary.score >= strict_summary.score


def test_no_events_when_thresholds_are_high():
    df = generate_drive(noise_m=0.0)
    _, events, summary = analyse_dataframe(
        df, Thresholds(brake_g=2.0, accel_g=2.0, lateral_g=2.0)
    )
    harsh = [e for e in events if e.kind in ("harsh_braking", "harsh_acceleration", "hard_cornering")]
    assert harsh == []
    assert summary.score == 100


def test_robust_to_gps_noise():
    # With realistic jitter the core events should still surface.
    df = generate_drive(noise_m=3.0, seed=1)
    _, events, _ = analyse_dataframe(df)
    kinds = {e.kind for e in events}
    assert "harsh_braking" in kinds
    assert "hard_cornering" in kinds
