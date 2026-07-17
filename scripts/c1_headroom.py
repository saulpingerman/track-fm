"""C1 headroom gate: do environmental fields explain the hard tail?

For each val-range window at wall-clock ~2h horizon, compute the DR-residual
difficulty (how far constant-velocity extrapolation missed) and sample the
local wind/current magnitudes at the ORIGIN position/time via the FieldStore.
Report Pearson correlations, per-decile means, and OLS R^2 -- overall and
restricted to the hard decile (the real target).

If fields don't explain measurable variance in the hard tail here, the
conditioning program's headroom is small and the flagship-vs-conditioning
tradeoff shifts.
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime

import numpy as np
import polars as pl
import xarray as xr

sys.path.insert(0, "/home/paul/projects/trackfm-v2/src")
from trackfm.context.fields import Field3D, FieldStore

# Sample 6 clean days spanning the v1 val window (2024-09-22..12-08)
DAYS = ["2024-09-25", "2024-10-05", "2024-10-20", "2024-11-05",
        "2024-11-20", "2024-12-05"]
CLEAN = "/home/paul/data/ais/clean/year={y}/month={m}/day={d}/tracks.parquet"
CMEMS_CUR = "/home/paul/data/context/cmems/BALTICSEA_ANALYSISFORECAST_PHY_003_006/bal_cur_{ym}.nc"
CMEMS_WAV = "/home/paul/data/context/cmems/BALTICSEA_ANALYSISFORECAST_WAV_003_010/bal_wav_{ym}.nc"
ERA5 = "/home/paul/data/context/era5/era5_{ym}.nc"

TAU = 7200.0            # target ~2h horizon
TAU_TOL = 0.15 * TAU
SEQ = 128
MAX_WINDOWS = 60000
KN = 0.514444


def load_month_store(ym: str) -> FieldStore:
    st = FieldStore()
    try:
        c = xr.open_dataset(CMEMS_CUR.format(ym=ym))
        t = c.time.values.astype("datetime64[s]").astype(np.int64)
        for v in ("uo", "vo"):
            st.add(Field3D(v, c[v].isel(depth=0).values.astype(np.float32), t,
                           c.latitude.values.astype(np.float64),
                           c.longitude.values.astype(np.float64)))
    except Exception as ex:
        print(f"  no CMEMS currents for {ym}: {ex}")
    try:
        w = xr.open_dataset(CMEMS_WAV.format(ym=ym))
        t = w.time.values.astype("datetime64[s]").astype(np.int64)
        st.add(Field3D("wave", w["VHM0"].values.astype(np.float32), t,
                       w.latitude.values.astype(np.float64),
                       w.longitude.values.astype(np.float64)))
    except Exception as ex:
        print(f"  no CMEMS waves for {ym}: {ex}")
    e = xr.open_dataset(ERA5.format(ym=ym))
    te = e.valid_time.values.astype("datetime64[s]").astype(np.int64)
    for v in ("u10", "v10"):
        st.add(Field3D(v, e[v].values.astype(np.float32), te,
                       e.latitude.values.astype(np.float64),
                       e.longitude.values.astype(np.float64)))
    return st


def compute_window_signals(day: str) -> dict | None:
    """For one clean day: sliding-window origins + 2h-horizon DR residuals +
    field magnitudes at each origin. Returns None if no fields available."""
    y, m, d = day.split("-")
    p = CLEAN.format(y=y, m=m, d=d)
    df = pl.read_parquet(p, columns=["mmsi", "timestamp", "lat", "lon", "sog",
                                      "cog", "dt_seconds"])
    df = df.sort(["mmsi", "timestamp"])
    st = load_month_store(f"{y}-{m}")
    if "uo" not in st.fields or "u10" not in st.fields:
        return None

    origs_lat, origs_lon, origs_t = [], [], []
    dr_res = []
    for (_,), grp in df.group_by(["mmsi"]):
        n = len(grp)
        if n < SEQ + 40:
            continue
        lat = grp["lat"].to_numpy(); lon = grp["lon"].to_numpy()
        sog = grp["sog"].to_numpy() * KN                # m/s
        cog = np.radians(grp["cog"].to_numpy())
        ts = grp["timestamp"].dt.epoch(time_unit="s").to_numpy().astype(np.int64)

        # sliding window every 32 positions (like materialization stride)
        for start in range(0, n - SEQ - 40, 32):
            o = start + SEQ - 1
            t0 = ts[o]
            # find future index closest to TAU seconds ahead (within tolerance)
            future_t = ts[o + 1 : min(o + 400, n)]
            if len(future_t) == 0:
                continue
            elapsed = future_t - t0
            j = int(np.argmin(np.abs(elapsed - TAU)))
            if abs(elapsed[j] - TAU) > TAU_TOL:
                continue
            tgt_i = o + 1 + j
            elapsed_s = float(elapsed[j])

            # DR extrapolation from last input position at speed sog[o], cog[o]
            v_north = sog[o] * np.cos(cog[o])          # m/s
            v_east = sog[o] * np.sin(cog[o])
            # to degrees: north 1 deg lat = 111320 m; east 1 deg lon = 111320*cos(lat) m
            dlat_dr = (v_north * elapsed_s) / 111320.0
            dlon_dr = (v_east * elapsed_s) / (111320.0 * np.cos(np.radians(lat[o])))
            # true target displacement
            dlat_t = lat[tgt_i] - lat[o]
            dlon_t = lon[tgt_i] - lon[o]
            # residual in km (via lat/lon → km at origin lat)
            r_lat_km = (dlat_t - dlat_dr) * 111.32
            r_lon_km = (dlon_t - dlon_dr) * 111.32 * np.cos(np.radians(lat[o]))
            dr_res.append(float(np.hypot(r_lat_km, r_lon_km)))
            origs_lat.append(float(lat[o]))
            origs_lon.append(float(lon[o]))
            origs_t.append(t0)
            if len(dr_res) >= MAX_WINDOWS:
                break
        if len(dr_res) >= MAX_WINDOWS:
            break

    if not dr_res:
        return None
    lat_a = np.array(origs_lat); lon_a = np.array(origs_lon); t_a = np.array(origs_t, dtype=np.int64)
    residuals = np.array(dr_res, dtype=np.float32)

    # only sample fields where CMEMS Baltic covers (lon > ~9.0)
    m_bal = (lon_a > 9.05) & (lon_a < 16.49) & (lat_a > 54.01) & (lat_a < 58.49)
    if m_bal.sum() < 200:
        return None
    lat_a, lon_a, t_a, residuals = lat_a[m_bal], lon_a[m_bal], t_a[m_bal], residuals[m_bal]

    uo = st.sample("uo", lat_a, lon_a, t_a)
    vo = st.sample("vo", lat_a, lon_a, t_a)
    u10 = st.sample("u10", lat_a, lon_a, t_a)
    v10 = st.sample("v10", lat_a, lon_a, t_a)
    wave = st.sample("wave", lat_a, lon_a, t_a) if "wave" in st.fields \
           else np.full_like(residuals, np.nan)

    return dict(residual_km=residuals,
                wind=np.hypot(u10, v10),         # m/s
                current=np.hypot(uo, vo),        # m/s
                wave=wave,
                sog_kn=None,                     # not needed for headline
                lat=lat_a, lon=lon_a)


def main():
    print(f"C1 headroom gate | tau={TAU/3600:.1f}h target, {len(DAYS)} days sampled")
    parts = []
    for d in DAYS:
        try:
            r = compute_window_signals(d)
        except Exception as ex:
            print(f"  {d}: skipped ({ex})")
            continue
        if r is None:
            print(f"  {d}: no data")
            continue
        print(f"  {d}: n={len(r['residual_km'])}")
        parts.append(r)
    if not parts:
        raise SystemExit("no data — check field files")

    res = np.concatenate([p["residual_km"] for p in parts])
    wind = np.concatenate([p["wind"] for p in parts])
    cur = np.concatenate([p["current"] for p in parts])
    wave = np.concatenate([p["wave"] for p in parts])
    ok = np.isfinite(res) & np.isfinite(wind) & np.isfinite(cur)
    res, wind, cur, wave = res[ok], wind[ok], cur[ok], wave[ok]
    print(f"\npooled n={len(res):,}, DR residual @2h: median {np.median(res):.2f}km, "
          f"p90 {np.percentile(res, 90):.2f}km, p99 {np.percentile(res, 99):.2f}km")

    def pearson(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    def ols_r2(X, y):
        X = np.column_stack([np.ones_like(y), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1 - ss_res / max(ss_tot, 1e-12)

    hard_mask = res >= np.percentile(res, 90)  # top decile: the hard tail
    out = {"n": int(len(res)), "hard_n": int(hard_mask.sum()),
           "tau_hours": TAU / 3600.0}
    print(f"\nPearson correlations (overall | hard-decile only):")
    for name, f in [("wind", wind), ("current", cur), ("wave (may be NaN)", wave)]:
        f_ok = np.isfinite(f)
        r_all = pearson(f[f_ok], res[f_ok])
        m = hard_mask & f_ok
        r_hard = pearson(f[m], res[m]) if m.sum() > 50 else float("nan")
        print(f"  {name:20} {r_all:>+.3f}  |  {r_hard:>+.3f}")
        out[f"pearson_{name.split()[0]}_all"] = round(r_all, 4)
        out[f"pearson_{name.split()[0]}_hard"] = round(r_hard, 4) if not np.isnan(r_hard) else None

    finite = np.isfinite(wave)
    X_all = np.column_stack([wind, cur])
    X_wav = np.column_stack([wind[finite], cur[finite], wave[finite]])
    r2_all = ols_r2(X_all, res)
    r2_wav = ols_r2(X_wav, res[finite]) if finite.sum() > 200 else float("nan")
    r2_hard = ols_r2(X_all[hard_mask], res[hard_mask]) if hard_mask.sum() > 200 else float("nan")
    print(f"\nOLS R^2 (residual_km ~ fields):")
    print(f"  wind+current                overall {r2_all:.3f}   hard-decile {r2_hard:.3f}")
    print(f"  wind+current+wave           overall {r2_wav:.3f}")

    out.update(r2_wind_cur_overall=round(r2_all, 4),
               r2_wind_cur_hard=round(r2_hard, 4) if not np.isnan(r2_hard) else None,
               r2_wind_cur_wave_overall=round(r2_wav, 4) if not np.isnan(r2_wav) else None)

    # decile means
    print(f"\nresidual decile -> mean field magnitudes:")
    print(f"  {'decile':>6} {'n':>6} {'median_res_km':>13} {'wind_m/s':>9} {'curr_m/s':>9}")
    edges = np.percentile(res, np.linspace(0, 100, 11))
    deciles = []
    for i in range(10):
        m = (res >= edges[i]) & (res <= edges[i + 1] if i == 9 else res < edges[i + 1])
        if m.sum() == 0: continue
        row = dict(decile=i, n=int(m.sum()),
                   median_res_km=round(float(np.median(res[m])), 2),
                   mean_wind=round(float(np.mean(wind[m])), 2),
                   mean_current=round(float(np.mean(cur[m])), 3))
        deciles.append(row)
        print(f"  {i:>6} {m.sum():>6} {row['median_res_km']:>13} "
              f"{row['mean_wind']:>9} {row['mean_current']:>9}")
    out["deciles"] = deciles

    out_path = "/home/paul/data/trackfm/c1_headroom.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
