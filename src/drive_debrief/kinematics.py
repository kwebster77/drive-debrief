"""Derive per-sample kinematics from a GPS track.

Design decisions that make this robust on real (noisy, ~1 Hz) GPS:

* **Longitudinal accel comes from *speed*, not differentiated position.**
  Phone loggers report Doppler speed, which is far cleaner than
  position deltas. If speed is absent we fall back to derived speed.
* **Heading is smoothed and speed-gated.** Bearing between two points is
  meaningless when nearly stationary (a 3 m GPS wobble becomes a wild
  heading swing), so we zero the yaw rate below a small speed.
* **Smoothing windows are specified in *seconds*** and converted to
  samples from the detected rate, so the same code works at 1 Hz or
  10 Hz.

We deliberately avoid the raw accelerometer: the phone is not aligned to
the car's axes and calibrating that rotation is a classic time-sink.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import headings_deg, segment_distances_m

G = 9.80665  # m/s^2


@dataclass
class Track:
    t: np.ndarray          # s, starts at 0
    lat: np.ndarray
    lon: np.ndarray
    speed: np.ndarray      # m/s, smoothed
    heading: np.ndarray    # deg, 0..360
    a_long: np.ndarray     # m/s^2  (+ = accelerating, - = braking)
    a_lat: np.ndarray      # m/s^2  (signed by turn direction)
    yaw_rate: np.ndarray   # deg/s
    jerk: np.ndarray       # m/s^3
    dist_cum: np.ndarray   # m
    dt: float              # median sample interval (s)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    @property
    def distance_m(self) -> float:
        return float(self.dist_cum[-1]) if len(self.dist_cum) else 0.0


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean that preserves length and edges."""
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) <= 2:
        return x
    window = min(window, len(x))
    s = pd.Series(x)
    return s.rolling(window, center=True, min_periods=1).mean().to_numpy()


def _cumtrapz(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral, same length as input (starts at 0)."""
    dt = np.diff(t)
    area = (y[:-1] + y[1:]) / 2.0 * dt
    return np.concatenate([[0.0], np.cumsum(area)])


def _win(seconds: float, dt: float) -> int:
    return max(1, int(round(seconds / dt)))


def build_track(
    df: pd.DataFrame,
    speed_smooth_s: float = 2.0,
    heading_smooth_s: float = 3.5,
    accel_smooth_s: float = 2.0,
    yaw_speed_gate_mps: float = 2.5,
) -> Track:
    """Turn a canonical DataFrame (t, lat, lon, [speed]) into a Track."""
    t = df["t"].to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)

    diffs = np.diff(t)
    dt_med = float(np.median(diffs)) if len(diffs) and np.median(diffs) > 0 else 1.0

    has_speed = "speed" in df.columns and bool(df["speed"].notna().all())
    if has_speed:
        speed_raw = np.clip(df["speed"].to_numpy(dtype=float), 0.0, None)
        dist_cum = _cumtrapz(speed_raw, t)  # clean distance from clean speed
    else:
        dist_cum = np.cumsum(segment_distances_m(lat, lon))
        speed_raw = np.clip(np.gradient(dist_cum, t), 0.0, None)

    speed = _rolling_mean(speed_raw, _win(speed_smooth_s, dt_med))

    # Prefer a device-reported course (Doppler, clean) over a bearing
    # differentiated from noisy position — the latter fakes cornering
    # during fast straight-line driving.
    if "course" in df.columns and bool(df["course"].notna().any()):
        heading_raw = df["course"].to_numpy(dtype=float)
        heading = _rolling_mean(heading_raw, _win(1.0, dt_med))
    else:
        # No device course (e.g. KML / bare GPS): a 3 m position wobble fakes a
        # sharp turn at speed, so smooth *position* before taking bearings.
        win = _win(heading_smooth_s, dt_med)
        lat_s = _rolling_mean(lat, win)
        lon_s = _rolling_mean(lon, win)
        heading_raw = headings_deg(lat_s, lon_s)
        heading = _rolling_mean(heading_raw, win)

    a_long = _rolling_mean(np.gradient(speed, t), _win(accel_smooth_s, dt_med))

    yaw_rate = np.gradient(_unwrap_deg(heading), t)     # deg/s
    yaw_rate = np.where(speed < yaw_speed_gate_mps, 0.0, yaw_rate)
    a_lat = _rolling_mean(speed * np.radians(yaw_rate), _win(accel_smooth_s, dt_med))

    jerk = np.gradient(a_long, t)

    for arr in (speed, heading, a_long, a_lat, yaw_rate, jerk):
        np.nan_to_num(arr, copy=False)

    return Track(
        t=t, lat=lat, lon=lon, speed=speed, heading=heading,
        a_long=a_long, a_lat=a_lat, yaw_rate=yaw_rate, jerk=jerk,
        dist_cum=dist_cum, dt=dt_med,
    )


def _unwrap_deg(heading: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(heading)))
