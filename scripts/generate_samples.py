"""Generate a library of sample drives across routes / cities / times / formats.

    python scripts/generate_samples.py            # writes sample_data/library/*

Every entry is emitted in several formats so each loader (CSV, GPX, KML,
Google records + semantic) has plenty of realistic test inputs.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drive_debrief.io import to_google_records, to_google_semantic, to_gpx, to_kml  # noqa: E402
from drive_debrief.synth import generate_scenario  # noqa: E402

OUT_DIR = os.path.join("sample_data", "library")

CITIES = {
    "london": (51.5326, -0.1050),
    "manchester": (53.4808, -2.2426),
    "edinburgh": (55.9533, -3.1883),
    "bristol": (51.4545, -2.5879),
    "leeds": (53.8008, -1.5491),
}

# time-of-day label -> ISO base timestamp (only the timestamps differ)
TIMES = {
    "morning": "2026-03-02T08:15:00+00:00",
    "midday": "2026-03-02T12:30:00+00:00",
    "rush": "2026-03-02T17:35:00+00:00",
    "night": "2026-03-02T21:50:00+00:00",
}

# (scenario, city, time, dt, position-noise m, formats)
PLAN = [
    ("harsh_braking_drill", "london", "morning", 1.0, 0.0, "csv gpx kml google semantic"),
    ("smooth_suburban", "leeds", "midday", 1.0, 0.0, "csv gpx kml"),
    ("smooth_suburban", "leeds", "night", 1.0, 3.0, "csv google"),
    ("motorway_cruise", "bristol", "morning", 0.2, 0.0, "csv gpx"),        # 5 Hz
    ("motorway_cruise", "bristol", "rush", 1.0, 3.0, "csv kml google"),
    ("urban_stop_go", "london", "rush", 1.0, 0.0, "csv gpx kml google"),
    ("urban_stop_go", "manchester", "morning", 1.0, 4.0, "csv google"),
    ("roundabouts", "manchester", "midday", 1.0, 0.0, "csv gpx kml"),
    ("roundabouts", "edinburgh", "night", 1.0, 3.0, "csv google semantic"),
    ("test_fail_dangerous", "london", "night", 1.0, 0.0, "csv gpx kml google"),
    ("nervous_learner", "edinburgh", "morning", 1.0, 0.0, "csv gpx google semantic"),
    ("dual_carriageway_merge", "leeds", "morning", 1.0, 0.0, "csv gpx kml"),
    ("school_run", "bristol", "morning", 1.0, 0.0, "csv gpx google"),
    ("school_run", "london", "rush", 1.0, 3.0, "csv kml"),
    ("rush_hour_crawl", "london", "rush", 1.0, 0.0, "csv gpx kml google"),
    ("country_lanes", "edinburgh", "midday", 1.0, 0.0, "csv gpx kml google semantic"),
    ("country_lanes", "manchester", "night", 1.0, 3.0, "csv google"),
]


def _write(df, base_iso, stem, fmts):
    written = []
    if "csv" in fmts:
        df.to_csv(stem + ".csv", index=False); written.append(".csv")
    if "gpx" in fmts:
        open(stem + ".gpx", "w").write(to_gpx(df, base_iso=base_iso)); written.append(".gpx")
    if "kml" in fmts:
        open(stem + ".kml", "w").write(to_kml(df, base_iso=base_iso)); written.append(".kml")
    if "google" in fmts:
        open(stem + ".google.json", "w").write(to_google_records(df, base_iso=base_iso)); written.append(".google.json")
    if "semantic" in fmts:
        open(stem + ".semantic.json", "w").write(to_google_semantic(df, base_iso=base_iso)); written.append(".semantic.json")
    return written


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for i, (scenario, city, tod, dt, noise, fmts) in enumerate(PLAN):
        lat0, lon0 = CITIES[city]
        base_iso = TIMES[tod]
        df = generate_scenario(
            scenario, dt=dt, lat0=lat0, lon0=lon0,
            noise_m=noise,
            speed_noise_mps=0.3 if noise else 0.0,
            course_noise_deg=2.0 if noise else 0.0,
            seed=7 + i,
            # KML/position-only variants are more realistic without a course column
            include_speed="google" in fmts or "csv" in fmts or "gpx" in fmts or "semantic" in fmts,
        )
        stem = os.path.join(OUT_DIR, f"{scenario}__{city}_{tod}")
        written = _write(df, base_iso, stem, fmts.split())
        total += len(written)
        print(f"{scenario:22s} {city:10s} {tod:7s} {dt}Hz noise={noise}m -> {' '.join(written)}")
    print(f"\nWrote {total} files across {len(PLAN)} drives into {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
