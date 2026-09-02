"""Reframe raw events as a DVSA-style test outcome.

The UK practical test is scored in *faults*:

* **driving fault** ("minor") — not dangerous on its own; 16+ fails you
* **serious fault** — potentially dangerous; a single one fails you
* **dangerous fault** — actual danger; a single one fails you

This maps each detected event onto that model so a learner sees the
number that actually matters to them: *would this have passed?* The
mapping is a heuristic (real examiners judge context we can't see) and is
presented as practice guidance, not an official result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .events import Event

# Peak-g at which a control event escalates. Tuned to be conservative:
# ordinary "harsh" driving is a minor; only clearly unsafe forces escalate.
_ESCALATION = {
    "harsh_braking": {"serious": 0.55, "dangerous": 0.75},
    "hard_cornering": {"serious": 0.50, "dangerous": 0.70},
    "harsh_acceleration": {"serious": 0.60, "dangerous": 0.90},
}

_MAX_MINORS = 16  # 16+ driving faults is a fail

_CATEGORY = {
    "harsh_braking": "Control – braking",
    "harsh_acceleration": "Control – accelerator",
    "hard_cornering": "Control – steering",
    "long_stop": "Progress – undue hesitation",
}


@dataclass
class Fault:
    t: float
    grade: str      # "minor" | "serious" | "dangerous"
    label: str
    category: str
    note: str


@dataclass
class Assessment:
    minors: int = 0
    serious: int = 0
    dangerous: int = 0
    passed: bool = True
    verdict: str = "Clean pass"
    faults: List[Fault] = field(default_factory=list)


def _grade_for(event: Event) -> str:
    rules = _ESCALATION.get(event.kind)
    if rules is not None:
        peak = abs(event.peak_value)
        if peak >= rules["dangerous"]:
            return "dangerous"
        if peak >= rules["serious"]:
            return "serious"
        return "minor"
    if event.kind == "long_stop":
        return "minor"
    return ""  # ordinary stops etc. are not faults


def assess(events: List[Event]) -> Assessment:
    faults: List[Fault] = []
    for e in events:
        grade = _grade_for(e)
        if not grade:
            continue
        faults.append(
            Fault(
                t=e.t_peak,
                grade=grade,
                label=e.label,
                category=_CATEGORY.get(e.kind, "Control"),
                note=e.detail,
            )
        )

    minors = sum(f.grade == "minor" for f in faults)
    serious = sum(f.grade == "serious" for f in faults)
    dangerous = sum(f.grade == "dangerous" for f in faults)

    passed = dangerous == 0 and serious == 0 and minors < _MAX_MINORS
    if dangerous:
        verdict = "Not a pass — dangerous fault"
    elif serious:
        verdict = "Not a pass — serious fault"
    elif minors >= _MAX_MINORS:
        verdict = f"Not a pass — {minors} driving faults"
    elif minors == 0:
        verdict = "Clean pass — no faults"
    else:
        verdict = "Likely pass"

    faults.sort(key=lambda f: f.t)
    return Assessment(
        minors=minors,
        serious=serious,
        dangerous=dangerous,
        passed=passed,
        verdict=verdict,
        faults=faults,
    )
