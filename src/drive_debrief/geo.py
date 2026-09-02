"""Geodesic helpers (scalar + vectorised).

Scalar versions exist mainly so the maths is trivially unit-testable;
the pipeline uses the vectorised numpy versions for speed.
"""
from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def segment_distances_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Per-sample distance from the previous sample (first element is 0)."""
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dphi = lat2 - lat1
    dlmb = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlmb / 2) ** 2
    d = 2 * EARTH_RADIUS_M * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return np.concatenate([[0.0], d])


def headings_deg(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Per-sample bearing (first element repeats the second so lengths match)."""
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dl = np.radians(lon[1:] - lon[:-1])
    y = np.sin(dl) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dl)
    brg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    if len(brg) == 0:
        return np.zeros_like(lat)
    return np.concatenate([[brg[0]], brg])
