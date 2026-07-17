"""C1-tube: upgrade of the C1 headroom gate that samples fields over the
DR SPACE-TIME TUBE instead of a single origin snapshot.

The storm scenario: a front crossing the vessel's path mid-window causes
the deviation, but is invisible in both the t0 snapshot and the T snapshot.
Point-C1 (c1_headroom.py) regressed difficulty on origin-time point samples
and found ~0% — this rerun tests whether that null survives when the field
is sampled where/when the vessel would actually encounter it:

  for f in {0, .25, .5, .75, 1.0}:  field( DR_position(f*tau), t0 + f*tau )

Tube features per window: max/mean over slices (did anything cross the
path), and max-minus-origin (did conditions WORSEN ahead — the avoidance
trigger). Reports origin-only vs tube R^2 side by side, overall and on the
hard decile.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl
import xarray as xr

sys.path.insert(0, "/home/paul/projects/trackfm-v2/src")
from trackfm.context.fields import Field3D, FieldStore

DAYS = ["2024-09-25", "2024-10-05", "2024-10-20", "2024-11-05",
        "2024-11-20", "2024-12-05"]
CLEAN = "/home/paul/data/ais/clean/year={y}/month={m}/day={d}/tracks.parquet"
CMEMS_CUR = "/home/paul/data/context/cmems/BALTICSEA_ANALYSISFORECAST_PHY_003_006/bal_cur_{ym}.nc"
CMEMS_WAV = "/home/paul/data/context/cmems/BALTICSEA_ANALYSISFORECAST_WAV_003_010/bal_wav_{ym}.nc"
ERA5 = "/home/paul/data/context/era5/era5_{ym}.nc"

TAU = 7200.0
TAU_TOL = 0.15 * TAU
SEQ = 128
MAX_WINDOWS = 60000
KN = 0.514444
FRACS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


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


def compute_day(day: str) -> dict | None:
    y, m, d = day.split("-")
    df = pl.read_parquet(CLEAN.format(y=y, m=m, d=d),
                         columns=["mmsi", "timestamp", "lat", "lon", "sog",
                                  "cog", "dt_seconds"]).sort(["mmsi", "timestamp"])
    st = load_month_store(f"{y}-{m}")
    if "uo" not in st.fields or "u10" not in st.fields:
        return None

    o_lat, o_lon, o_t = [], [], []
    dlat_drs, dlon_drs, taus = [], [], []
    dr_res, sogs = [], []
    for (_,), grp in df.group_by(["mmsi"]):
        n = len(grp)
        if n < SEQ + 40:
            continue
        lat = grp["lat"].to_numpy(); lon = grp["lon"].to_numpy()
        sog = grp["sog"].to_numpy() * KN
        cog = np.radians(grp["cog"].to_numpy())
        ts = grp["timestamp"].dt.epoch(time_unit="s").to_numpy().astype(np.int64)
        for start in range(0, n - SEQ - 40, 32):
            o = start + SEQ - 1
            t0 = ts[o]
            future_t = ts[o + 1: min(o + 400, n)]
            if len(future_t) == 0:
                continue
            elapsed = future_t - t0
            j = int(np.argmin(np.abs(elapsed - TAU)))
            if abs(elapsed[j] - TAU) > TAU_TOL:
                continue
            tgt_i = o + 1 + j
            el = float(elapsed[j])
            v_n = sog[o] * np.cos(cog[o]); v_e = sog[o] * np.sin(cog[o])
            dlat_dr = (v_n * el) / 111320.0
            dlon_dr = (v_e * el) / (111320.0 * np.cos(np.radians(lat[o])))
            r_lat = (lat[tgt_i] - lat[o] - dlat_dr) * 111.32
            r_lon = (lon[tgt_i] - lon[o] - dlon_dr) * 111.32 * np.cos(np.radians(lat[o]))
            dr_res.append(float(np.hypot(r_lat, r_lon)))
            sogs.append(float(sog[o]))
            o_lat.append(float(lat[o])); o_lon.append(float(lon[o])); o_t.append(t0)
            dlat_drs.append(dlat_dr); dlon_drs.append(dlon_dr); taus.append(el)
            if len(dr_res) >= MAX_WINDOWS:
                break
        if len(dr_res) >= MAX_WINDOWS:
            break
    if not dr_res:
        return None

    lat_a = np.array(o_lat); lon_a = np.array(o_lon)
    t_a = np.array(o_t, dtype=np.int64)
    dlat_a = np.array(dlat_drs); dlon_a = np.array(dlon_drs)
    tau_a = np.array(taus); res = np.array(dr_res, dtype=np.float32)
    sog_a = np.array(sogs, dtype=np.float32)

    # keep tube fully inside the CMEMS Baltic box (origin AND DR endpoint)
    end_lat, end_lon = lat_a + dlat_a, lon_a + dlon_a
    m_bal = ((lon_a > 9.05) & (lon_a < 16.49) & (lat_a > 54.01) & (lat_a < 58.49) &
             (end_lon > 9.05) & (end_lon < 16.49) & (end_lat > 54.01) & (end_lat < 58.49))
    if m_bal.sum() < 200:
        return None
    lat_a, lon_a, t_a = lat_a[m_bal], lon_a[m_bal], t_a[m_bal]
    dlat_a, dlon_a, tau_a, res = dlat_a[m_bal], dlon_a[m_bal], tau_a[m_bal], res[m_bal]
    sog_a = sog_a[m_bal]

    # sample each field on the DR tube: position at fraction f, time t0+f*tau
    slices = {}                                     # name -> (n_windows, n_fracs)
    for name in ("u10", "v10", "uo", "vo", "wave"):
        if name not in st.fields:
            continue
        cols = []
        for f in FRACS:
            cols.append(st.sample(name, lat_a + f * dlat_a, lon_a + f * dlon_a,
                                  (t_a + f * tau_a).astype(np.int64)))
        slices[name] = np.column_stack(cols)

    wind_t = np.hypot(slices["u10"], slices["v10"])       # (n, K)
    cur_t = np.hypot(slices["uo"], slices["vo"])
    wave_t = slices.get("wave")
    return dict(res=res, sog=sog_a, wind_t=wind_t, cur_t=cur_t, wave_t=wave_t)


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def ols_r2(X, y):
    X = np.column_stack([np.ones_like(y), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_res = float(np.sum((y - X @ beta) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / max(ss_tot, 1e-12)


def main():
    print(f"C1-TUBE | tau={TAU/3600:.1f}h, fracs={FRACS.tolist()}, {len(DAYS)} days")
    parts = []
    for d in DAYS:
        try:
            r = compute_day(d)
        except Exception as ex:
            print(f"  {d}: skipped ({ex})")
            continue
        if r is None:
            print(f"  {d}: no data")
            continue
        print(f"  {d}: n={len(r['res'])}", flush=True)
        parts.append(r)
    if not parts:
        raise SystemExit("no data")

    res = np.concatenate([p["res"] for p in parts])
    sog = np.concatenate([p["sog"] for p in parts])
    wind = np.vstack([p["wind_t"] for p in parts])
    cur = np.vstack([p["cur_t"] for p in parts])
    have_wave = all(p["wave_t"] is not None for p in parts)
    wave = np.vstack([p["wave_t"] for p in parts]) if have_wave else None

    # SANE-RANGE mask, not just isfinite: netCDF fill values (~1e37/1e20)
    # pass isfinite and destroy OLS numerics. Physical bounds: wind<60 m/s,
    # current<5 m/s, wave<20 m.
    ok = (np.isfinite(res) &
          np.isfinite(wind).all(1) & (np.abs(wind) < 60).all(1) &
          np.isfinite(cur).all(1) & (np.abs(cur) < 5).all(1))
    if have_wave:
        ok &= np.isfinite(wave).all(1) & (np.abs(wave) < 20).all(1)
    n_dropped = int((~ok).sum())
    res, sog, wind, cur = res[ok], sog[ok], wind[ok], cur[ok]
    wave = wave[ok] if have_wave else None
    print(f"\npooled n={len(res):,} (dropped {n_dropped} fill/out-of-range) | "
          f"residual@2h median {np.median(res):.2f} "
          f"p90 {np.percentile(res, 90):.2f} p99 {np.percentile(res, 99):.2f} km")

    np.savez("/home/paul/data/trackfm/c1_tube_arrays.npz", res=res, sog=sog,
             wind=wind, cur=cur, **({"wave": wave} if have_wave else {}))

    hard = res >= np.percentile(res, 90)

    def tube_feats(F):
        """origin, max-over-tube, mean-over-tube, worsening (max - origin)"""
        return dict(origin=F[:, 0], tmax=F.max(1), tmean=F.mean(1),
                    worsen=F.max(1) - F[:, 0])

    fields = {"wind": tube_feats(wind), "current": tube_feats(cur)}
    if have_wave:
        fields["wave"] = tube_feats(wave)

    def partial_r(x, y, z):
        """Pearson(x,y) after regressing z (confounders) out of both."""
        Z = np.column_stack([np.ones_like(y), z])
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return pearson(rx, ry)

    # CONFOUND: tube length = sog * tau. 'worsen' grows mechanically with a
    # longer tube (more spatial variation sampled), and residual km also
    # grows with speed. Report raw AND sog-partialed correlations.
    print(f"\nPearson r vs residual (raw overall | sog-partialed overall | sog-partialed hard):")
    out = {"n": int(len(res)), "fracs": FRACS.tolist()}
    conf = np.column_stack([sog, sog ** 2])
    for fname, feats in fields.items():
        for kind, x in feats.items():
            r_raw = pearson(x, res)
            r_part = partial_r(x, res, conf)
            r_ph = partial_r(x[hard], res[hard], conf[hard]) if hard.sum() > 50 else float("nan")
            print(f"  {fname:8}{kind:8} {r_raw:+.3f} | {r_part:+.3f} | {r_ph:+.3f}")
            out[f"r_{fname}_{kind}"] = [round(r_raw, 4), round(r_part, 4), round(r_ph, 4)]

    # head-to-head R^2, z-scored features (fill-value bug fixed by sane-range
    # mask; z-scoring guards the conditioning)
    def zs(X):
        return (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)

    X_origin = zs(np.column_stack([f["origin"] for f in fields.values()]))
    X_tube = zs(np.column_stack(sum([[f["origin"], f["tmax"], f["worsen"]]
                                     for f in fields.values()], [])))
    X_sog = zs(np.column_stack([sog, sog ** 2]))
    X_tube_sog = np.column_stack([X_tube, X_sog])
    X_origin_sog = np.column_stack([X_origin, X_sog])
    for label, X in [("sog only (confound base)", X_sog),
                     ("origin-only (point-C1)", X_origin),
                     ("tube", X_tube),
                     ("sog + origin", X_origin_sog),
                     ("sog + tube", X_tube_sog)]:
        r2a = ols_r2(X, res)
        r2h = ols_r2(X[hard], res[hard]) if hard.sum() > 200 else float("nan")
        print(f"OLS R^2 {label:28} overall {r2a:.4f}  hard {r2h:.4f}")
        out[f"r2_{label.replace(' ', '_')}"] = [round(r2a, 4), round(r2h, 4)]

    # storm-crossing indicator with SPEED-MATCHED control: within each sog
    # quintile, compare hard-rate of crossing vs non-crossing windows.
    print(f"\nstorm-crossing indicator (tube max >> origin), speed-matched:")
    for fname, thresh in [("wind", 5.0), ("wave", 1.0)] if have_wave else [("wind", 5.0)]:
        crossing = fields[fname]["worsen"] > thresh
        if crossing.sum() < 30:
            print(f"  {fname}: too few crossing events ({int(crossing.sum())})")
            continue
        lifts, ns = [], []
        edges = np.percentile(sog, [0, 20, 40, 60, 80, 100])
        for i in range(5):
            mq = (sog >= edges[i]) & (sog <= edges[i + 1])
            nc = crossing & mq
            if nc.sum() < 10 or (~crossing & mq).sum() < 100:
                continue
            rate_c = float(hard[nc].mean())
            rate_n = float(hard[~crossing & mq].mean())
            if rate_n > 0:
                lifts.append(rate_c / rate_n); ns.append(int(nc.sum()))
        raw_lift = float(hard[crossing].mean() / max(hard[~crossing].mean(), 1e-9))
        if lifts:
            wl = float(np.average(lifts, weights=ns))
            print(f"  {fname}: n_crossing={int(crossing.sum()):,}  raw lift "
                  f"{raw_lift:.2f}x  speed-matched lift {wl:.2f}x "
                  f"({len(lifts)} quintiles)")
            out[f"crossing_{fname}"] = dict(n=int(crossing.sum()),
                                            raw_lift=round(raw_lift, 3),
                                            matched_lift=round(wl, 3))
        else:
            print(f"  {fname}: crossing events too concentrated for speed match "
                  f"(n={int(crossing.sum())}, raw lift {raw_lift:.2f}x)")

    json.dump(out, open("/home/paul/data/trackfm/c1_tube.json", "w"), indent=2)
    print("\nwritten /home/paul/data/trackfm/c1_tube.json")


if __name__ == "__main__":
    main()
