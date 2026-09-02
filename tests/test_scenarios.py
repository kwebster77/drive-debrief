"""Every named scenario runs, and the notable ones assess as designed."""
from drive_debrief.assessment import assess
from drive_debrief.pipeline import analyse_dataframe
from drive_debrief.synth import SCENARIOS, generate_scenario


def _assess(name, **kw):
    df = generate_scenario(name, **kw)
    _, events, _ = analyse_dataframe(df)
    return assess(events), {e.kind for e in events}


def test_all_scenarios_produce_a_drive():
    for name in SCENARIOS:
        df = generate_scenario(name)
        assert len(df) >= 5
        assert {"t", "lat", "lon"}.issubset(df.columns)


def test_dangerous_scenario_fails_with_dangerous_fault():
    a, _ = _assess("test_fail_dangerous")
    assert a.dangerous >= 1 and not a.passed


def test_smooth_suburban_is_a_clean_pass():
    a, _ = _assess("smooth_suburban")
    assert a.passed and a.serious == 0 and a.dangerous == 0


def test_nervous_learner_has_a_long_stop():
    _, kinds = _assess("nervous_learner")
    assert "long_stop" in kinds


def test_roundabouts_flag_cornering():
    _, kinds = _assess("roundabouts")
    assert "hard_cornering" in kinds


def test_motorway_cruise_passes():
    a, _ = _assess("motorway_cruise")
    assert a.passed
