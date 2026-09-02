"""Generate synthetic drives with *known* events.

Each drive is built from an explicit speed + yaw profile and integrated to
lat/lon, so we know exactly where every event should land — good for demos
*and* as ground truth for the tests. ``generate_drive`` keeps the original
canonical profile (the tests pin to it); ``generate_scenario`` builds any of
the named routes in ``SCENARIOS``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_M_PER_DEG = 111_320.0

# Each segment: (duration_s, v_start_mps, v_end_mps, yaw_deg_per_s).
CANONICAL = [
    (5.0, 0.0, 13.4, 0.0),
    (15.0, 13.4, 13.4, 0.0),
    (2.5, 13.4, 0.0, 0.0),     # harsh brake (~0.55g)
    (7.5, 0.0, 0.0, 0.0),      # stop
    (3.0, 0.0, 11.0, 0.0),     # harsh accel (~0.37g)
    (4.0, 11.0, 11.0, 0.0),
    (4.0, 11.0, 11.0, 25.0),   # sharp bend (~0.49g lateral)
    (5.0, 11.0, 11.0, 0.0),
    (4.0, 11.0, 0.0, 0.0),     # gentle stop
]

# A library of believable routes. Speeds are m/s (×2.237 ≈ mph).
SCENARIOS = {
    "harsh_braking_drill": CANONICAL,
    "smooth_suburban": [
        (6, 0, 11, 0), (25, 11, 11, 0), (4, 11, 11, 8), (25, 11, 11, 0), (6, 11, 0, 0),
    ],
    "motorway_cruise": [
        (15, 0, 29, 0), (80, 29, 29, 0), (3, 29, 29, 4), (80, 29, 29, 0), (25, 29, 20, 0),
    ],
    "urban_stop_go": [
        (4, 0, 9, 0), (8, 9, 9, 0), (3, 9, 0, 0), (6, 0, 0, 0),
        (4, 0, 10, 0), (6, 10, 10, 0), (2, 10, 0, 0), (6, 0, 0, 0),   # harsh brake ~0.51g
        (4, 0, 9, 0), (10, 9, 9, 15), (3, 9, 0, 0), (6, 0, 0, 0),     # a corner
    ],
    "roundabouts": [
        (5, 0, 11, 0), (8, 11, 11, 0), (5, 12, 9, 30), (2, 9, 9, 0),
        (6, 9, 11, -28), (8, 11, 11, 0), (5, 11, 9, 32), (6, 9, 0, 0),
    ],
    "test_fail_dangerous": [
        (6, 0, 15, 0), (10, 15, 15, 0), (1.5, 15, 0, 0), (5, 0, 0, 0),  # ~1.0g brake
        (3, 0, 13, 0), (6, 13, 13, 0), (3, 13, 13, 45), (8, 13, 0, 0),  # ~1.0g corner
    ],
    "nervous_learner": [
        (8, 0, 7, 0), (6, 7, 7, 0), (4, 7, 0, 0), (30, 0, 0, 0),        # long hesitation stop
        (7, 0, 8, 0), (10, 8, 8, 0), (3, 8, 3, 0), (5, 3, 3, 0),
        (6, 3, 10, 0), (15, 10, 10, 6), (9, 10, 0, 0),
    ],
    "dual_carriageway_merge": [
        (4, 0, 25, 0), (40, 25, 25, 0), (3, 25, 25, 6), (35, 25, 25, 0),  # ~0.64g merge accel
        (6, 25, 12, 0), (10, 12, 12, 0), (5, 12, 0, 0),
    ],
    "school_run": [
        (5, 0, 9, 0), (8, 9, 9, 0), (4, 9, 9, 15), (8, 9, 9, 0), (3, 9, 0, 0),
        (8, 0, 0, 0), (4, 0, 9, 0), (10, 9, 9, -12), (5, 9, 0, 0),
    ],
    "rush_hour_crawl": [
        seg for _ in range(6) for seg in [(3, 0, 5, 0), (5, 5, 5, 0), (2, 5, 0, 0), (8, 0, 0, 0)]
    ],
    "country_lanes": [
        (5, 0, 12, 0), (6, 12, 12, 0), (5, 12, 10, 18), (4, 10, 10, 0),
        (5, 10, 12, -20), (6, 12, 12, 0), (5, 12, 10, 16), (4, 10, 10, 0),
        (6, 10, 12, -16), (6, 12, 0, 0),
    ],
}


def _expand(segments, dt: float):
    """Expand a segment list into per-sample speed + yaw arrays."""
    speed, yaw = [], []
    for dur, v0, v1, yw in segments:
        n = max(1, int(round(dur / dt)))
        speed.extend(np.linspace(v0, v1, n, endpoint=False))
        yaw.extend([float(yw)] * n)
    return np.asarray(speed, dtype=float), np.asarray(yaw, dtype=float)


def _assemble(speed, yaw, dt, lat0, lon0, noise_m, speed_noise_mps,
              course_noise_deg, seed, include_speed, start_heading=90.0):
    """Integrate a speed+yaw profile into a canonical DataFrame."""
    n = len(speed)
    t = np.arange(n) * dt

    heading = np.zeros(n)
    heading[0] = start_heading
    for i in range(1, n):
        heading[i] = heading[i - 1] + yaw[i] * dt

    hd = np.radians(heading)
    east = np.cumsum(speed * np.sin(hd) * dt)
    north = np.cumsum(speed * np.cos(hd) * dt)
    lat = lat0 + north / _M_PER_DEG
    lon = lon0 + east / (_M_PER_DEG * np.cos(np.radians(lat0)))

    if noise_m > 0:
        rng = np.random.default_rng(seed)
        lat = lat + rng.normal(0, noise_m, n) / _M_PER_DEG
        lon = lon + rng.normal(0, noise_m, n) / (_M_PER_DEG * np.cos(np.radians(lat0)))

    data = {"t": t, "lat": lat, "lon": lon}
    if include_speed:
        spd = speed.copy()
        crs = heading.copy() % 360.0
        if speed_noise_mps > 0:
            spd = np.clip(spd + np.random.default_rng(seed + 1).normal(0, speed_noise_mps, n), 0.0, None)
        if course_noise_deg > 0:
            crs = (crs + np.random.default_rng(seed + 2).normal(0, course_noise_deg, n)) % 360.0
        data["speed"] = spd
        data["course"] = crs
    return pd.DataFrame(data)


def generate_drive(
    dt: float = 1.0,
    lat0: float = 51.5326,
    lon0: float = -0.1050,
    noise_m: float = 0.0,
    speed_noise_mps: float = 0.0,
    course_noise_deg: float = 0.0,
    seed: int = 7,
    include_speed: bool = True,
) -> pd.DataFrame:
    """The canonical drive (the tests pin to this exact profile)."""
    speed, yaw = _expand(CANONICAL, dt)
    return _assemble(speed, yaw, dt, lat0, lon0, noise_m, speed_noise_mps,
                     course_noise_deg, seed, include_speed)


def generate_scenario(
    name: str,
    dt: float = 1.0,
    lat0: float = 51.5326,
    lon0: float = -0.1050,
    noise_m: float = 0.0,
    speed_noise_mps: float = 0.0,
    course_noise_deg: float = 0.0,
    seed: int = 7,
    include_speed: bool = True,
    start_heading: float = 90.0,
) -> pd.DataFrame:
    """Build a named route from ``SCENARIOS``."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; options: {sorted(SCENARIOS)}")
    speed, yaw = _expand(SCENARIOS[name], dt)
    return _assemble(speed, yaw, dt, lat0, lon0, noise_m, speed_noise_mps,
                     course_noise_deg, seed, include_speed, start_heading)
