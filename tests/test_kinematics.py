"""Kinematics maths should recover known physics from position + time."""
import numpy as np
import pandas as pd

from drive_debrief.geo import bearing_deg, haversine_m
from drive_debrief.kinematics import G, build_track


def test_haversine_known_distance():
    # 0.001 deg of latitude is ~111.32 m.
    d = haversine_m(51.5, -0.1, 51.501, -0.1)
    assert 110 < d < 113


def test_bearing_cardinal():
    assert abs(bearing_deg(0, 0, 1, 0) - 0) < 1e-6      # due north
    assert abs(bearing_deg(0, 0, 0, 1) - 90) < 1e-6     # due east


def test_constant_acceleration_recovered():
    # Straight east line, speed ramps 0 -> 10 m/s over 10s => a = 1 m/s^2.
    dt = 0.1
    t = np.arange(0, 10, dt)
    speed = t * 1.0
    lat0 = 51.5
    east = np.cumsum(speed * dt)
    lon = -0.1 + east / (111_320.0 * np.cos(np.radians(lat0)))
    df = pd.DataFrame({"t": t, "lat": lat0 * np.ones_like(t), "lon": lon})

    track = build_track(df)
    # Ignore smoothing edges; the interior should sit near 1 m/s^2.
    interior = track.a_long[10:-10]
    assert abs(np.median(interior) - 1.0) < 0.15
    assert abs(track.a_long[50] / G - (1.0 / G)) < 0.05
