"""FieldStore: interpolation correctness and forecast-leakage enforcement."""
from __future__ import annotations

import numpy as np
import pytest

from trackfm.context.fields import Field3D, FieldStore, LeakageError

T0 = 1_700_000_000


def _linear_field(issued=None):
    """f(lat, lon) = 10*lat + lon at t0, +100 at t0+3600 (exactly bilinear)."""
    lat = np.array([54.0, 55.0, 56.0])
    lon = np.array([10.0, 11.0, 12.0, 13.0])
    base = 10.0 * lat[:, None] + lon[None, :]
    values = np.stack([base, base + 100.0]).astype(np.float32)
    return Field3D("wind", values, np.array([T0, T0 + 3600], dtype=np.int64),
                   lat, lon, issued_time=issued)


def test_bilinear_and_nearest_time_exact():
    st = FieldStore()
    st.add(_linear_field())
    v = st.sample("wind", lat=[54.5, 55.25], lon=[10.5, 12.75],
                  t=[T0 + 10, T0 + 3500])
    # bilinear reproduces a linear surface exactly; t snaps to nearest hour
    np.testing.assert_allclose(v, [10 * 54.5 + 10.5, 10 * 55.25 + 12.75 + 100],
                               rtol=1e-6)


def test_descending_lat_axis_matches_ascending():
    st = FieldStore()
    f = _linear_field()
    flipped = Field3D("wind", f.values[:, ::-1].copy(), f.valid_time,
                      f.lat[::-1].copy(), f.lon)
    st.add(flipped)
    v = st.sample("wind", lat=[54.5], lon=[11.5], t=[T0])
    np.testing.assert_allclose(v, [10 * 54.5 + 11.5], rtol=1e-6)


def test_operational_field_requires_decision_time():
    st = FieldStore()
    st.add(_linear_field(issued=np.array([T0 - 7200, T0 - 7200], dtype=np.int64)))
    with pytest.raises(LeakageError, match="decision_time"):
        st.sample("wind", lat=[55.0], lon=[11.0], t=[T0])


def test_later_issued_forecast_is_invisible():
    """Two slices for the same valid hour: an old forecast and a fresher one
    issued after decision time. The query must get the OLD one."""
    lat = np.array([54.0, 56.0])
    lon = np.array([10.0, 12.0])
    old = np.full((2, 2), 1.0, dtype=np.float32)
    new = np.full((2, 2), 2.0, dtype=np.float32)
    f = Field3D("wind", np.stack([old, new]),
                valid_time=np.array([T0 + 3600, T0 + 3600], dtype=np.int64),
                lat=lat, lon=lon,
                issued_time=np.array([T0 - 3600, T0 + 1800], dtype=np.int64))
    st = FieldStore()
    st.add(f)
    v = st.sample("wind", lat=[55.0], lon=[11.0], t=[T0 + 3600], decision_time=T0)
    np.testing.assert_allclose(v, [1.0])   # fresher run leaked => would be 2.0
    v2 = st.sample("wind", lat=[55.0], lon=[11.0], t=[T0 + 3600],
                   decision_time=T0 + 1800)
    np.testing.assert_allclose(v2, [2.0])  # by then the fresh run existed


def test_no_forecast_available_raises():
    st = FieldStore()
    st.add(_linear_field(issued=np.array([T0, T0], dtype=np.int64)))
    with pytest.raises(LeakageError, match="no forecast issued"):
        st.sample("wind", lat=[55.0], lon=[11.0], t=[T0], decision_time=T0 - 1)
