"""Generate a synthetic practice drive with *known* events.

Because the drive is built from an explicit speed + yaw profile and then
integrated to lat/lon, we know exactly where each event should land. That
makes it both a believable demo and a ground truth for the unit tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_M_PER_DEG = 111_320.0


def _profile(dt: float):
    """Return arrays of target speed (m/s) and yaw rate (deg/s) per sample.

    Timeline (approx):
      0-5s    gentle accel to 30mph          (~0.27g, not flagged)
      5-20s   cruise straight
      20-22.5s HARSH BRAKE to a stop          (~0.55g -> flagged)
      22.5-30s stationary                     (stop)
      30-33s  HARSH ACCEL to ~25mph           (~0.37g -> flagged)
      33-37s  cruise
      37-41s  SHARP LEFT bend                 (~0.39g lateral -> flagged)
      41-46s  cruise
      46-50s  gentle stop                     (~0.28g, not flagged) + stop
    """
    segs = [
        # (duration_s, v_start, v_end, yaw_deg_s)
        (5.0, 0.0, 13.4, 0.0),
        (15.0, 13.4, 13.4, 0.0),
        (2.5, 13.4, 0.0, 0.0),
        (7.5, 0.0, 0.0, 0.0),
        (3.0, 0.0, 11.0, 0.0),
        (4.0, 11.0, 11.0, 0.0),
        (4.0, 11.0, 11.0, 20.0),   # sharp bend
        (5.0, 11.0, 11.0, 0.0),
        (4.0, 11.0, 0.0, 0.0),
    ]
    speed, yaw = [], []
    for dur, v0, v1, yw in segs:
        n = int(round(dur / dt))
        speed.extend(np.linspace(v0, v1, n, endpoint=False))
        yaw.extend([yw] * n)
    return np.asarray(speed), np.asarray(yaw)


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
    """Build a synthetic drive at a realistic 1 Hz GPS cadence.

    ``noise_m`` adds Gaussian position jitter (metres); ``speed_noise_mps``
    jitters the Doppler speed. Keep both 0 for deterministic tests.
    ``include_speed`` writes a velocity column, exactly as phyphox /
    SensorLog do — real logs always carry Doppler speed, which is why the
    pipeline prefers it over differentiated position.
    """
    speed, yaw = _profile(dt)
    n = len(speed)
    t = np.arange(n) * dt

    heading = np.zeros(n)
    heading[0] = 90.0  # heading east
    for i in range(1, n):
        heading[i] = heading[i - 1] + yaw[i] * dt

    hd = np.radians(heading)
    ve = speed * np.sin(hd)  # east
    vn = speed * np.cos(hd)  # north
    east = np.cumsum(ve * dt)
    north = np.cumsum(vn * dt)

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
            rng2 = np.random.default_rng(seed + 1)
            spd = np.clip(spd + rng2.normal(0, speed_noise_mps, n), 0.0, None)
        if course_noise_deg > 0:
            rng3 = np.random.default_rng(seed + 2)
            crs = (crs + rng3.normal(0, course_noise_deg, n)) % 360.0
        data["speed"] = spd
        data["course"] = crs
    return pd.DataFrame(data)
