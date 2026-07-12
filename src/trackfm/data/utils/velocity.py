"""Velocity calculation utilities for AIS data processing."""
from datetime import datetime
from typing import Optional

from .geo import haversine_km


def velocity_knots(
    lat1: float, lon1: float, t1: datetime,
    lat2: float, lon2: float, t2: datetime
) -> float:
    """Calculate velocity in knots between two positions.

    Args:
        lat1: Latitude of first position
        lon1: Longitude of first position
        t1: Timestamp of first position
        lat2: Latitude of second position
        lon2: Longitude of second position
        t2: Timestamp of second position

    Returns:
        Speed in knots. Returns inf if time delta is zero or negative.
    """
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    dt_hours = (t2 - t1).total_seconds() / 3600

    if dt_hours <= 0:
        return float('inf')

    speed_kmh = dist_km / dt_hours
    speed_knots = speed_kmh / 1.852  # 1 knot = 1.852 km/h

    return speed_knots


def calculate_velocities(
    lats: list, lons: list, timestamps: list
) -> list:
    """Calculate velocities between consecutive positions.

    Args:
        lats: List of latitudes
        lons: List of longitudes
        timestamps: List of timestamps

    Returns:
        List of velocities in knots. First element is 0.
    """
    if len(lats) < 2:
        return [0.0] * len(lats)

    velocities = [0.0]  # First position has no velocity

    for i in range(1, len(lats)):
        if lats[i] is None or lons[i] is None or lats[i-1] is None or lons[i-1] is None:
            velocities.append(float('inf'))
        else:
            v = velocity_knots(
                lats[i-1], lons[i-1], timestamps[i-1],
                lats[i], lons[i], timestamps[i]
            )
            velocities.append(v)

    return velocities
