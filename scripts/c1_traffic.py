"""C1-traffic: do OTHER VESSELS explain the hard tail?

Same gate design as c1_tube.py (2h-horizon DR residuals over val-window
days, sog-controlled), but the conditioning signal is dynamic TRAFFIC
computed from the AIS data itself at each window's origin time:

  - local density (other vessels within 2/5/10 km at t0)
  - distance to nearest vessel
  - encounter geometry: min CPA over neighbors with TCPA in [0, 30 min]
    (COLREGS-relevant: someone will pass close -> give-way maneuvering)
  - crossing indicator: such a neighbor on a crossing course (30-150 deg)
  - anchored-vessel count within 5 km (queue/anchorage pressure)

Hypothesis (from tube-C1's verdict): the bulk of the hard tail is
decisions, and the most common decision-driver is other traffic. If this
gate comes back much stronger than weather did, dynamic traffic fields
jump the priority queue for Tier-3 conditioning.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import polars as pl

DAYS = ["2024-09-25", "2024-10-05", "2024-10-20", "2024-11-05",
        "2024-11-20", "2024-12-05"]
CLEAN = "/home/paul/data/ais/clean/year={y}/month={m}/day={d}/tracks.parquet"

TAU = 7200.0
TAU_TOL = 0.15 * TAU
SEQ = 128
MAX_WINDOWS = 60000
KN = 0.514444
KM_PER_DEG = 111.32
SLAB = 300                     # s, time-slab for neighbor lookup
CELL = 0.15                    # deg lat (~16.7 km) spatial bin


def build_index(ts, lat, lon):
    """(slab, ix, iy) -> row indices, for +-1-cell neighborhood lookups."""
    slab = (ts // SLAB).astype(np.int64)
    ix = np.floor(lon / (CELL * 2)).astype(np.int64)   # lon cells ~2x wider
    iy = np.floor(lat / CELL).astype(np.int64)
    idx = defaultdict(list)
    for i in range(len(ts)):
        idx[(slab[i], ix[i], iy[i])].append(i)
    return {k: np.asarray(v) for k, v in idx.items()}, slab, ix, iy


def neighbors_at(index, t0, lat0, lon0):
    """candidate row indices in the 2 nearest slabs x 3x3 cells."""
    s0 = t0 // SLAB
    ix0 = int(np.floor(lon0 / (CELL * 2)))
    iy0 = int(np.floor(lat0 / CELL))
    out = []
    for s in (s0, s0 - 1, s0 + 1):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                a = index.get((s, ix0 + dx, iy0 + dy))
                if a is not None:
                    out.append(a)
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def compute_day(day: str) -> dict | None:
    y, m, d = day.split("-")
    df = pl.read_parquet(CLEAN.format(y=y, m=m, d=d),
                         columns=["mmsi", "timestamp", "lat", "lon", "sog",
                                  "cog"]).sort(["mmsi", "timestamp"])
    mmsi_all = df["mmsi"].to_numpy()
    ts_all = df["timestamp"].dt.epoch(time_unit="s").to_numpy().astype(np.int64)
    lat_all = df["lat"].to_numpy(); lon_all = df["lon"].to_numpy()
    sog_all = df["sog"].to_numpy(); cog_all = np.radians(df["cog"].to_numpy())
    index, *_ = build_index(ts_all, lat_all, lon_all)

    # window origins + DR residuals (same recipe as c1_tube)
    o_rows = []                                  # (row_i, tgt_row, elapsed)
    df_idx = np.arange(len(df))
    starts = {}
    for (mm,), grp in df.group_by(["mmsi"], maintain_order=True):
        pass                                     # placeholder; use array walk below
    # array walk per mmsi (faster than group_by materialization)
    order = np.lexsort((ts_all, mmsi_all))
    mm_s, ts_s = mmsi_all[order], ts_all[order]
    bounds = np.flatnonzero(np.r_[True, mm_s[1:] != mm_s[:-1], True])
    res, sogs = [], []
    feat = defaultdict(list)
    rng = np.random.default_rng(17)
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        n = hi - lo
        if n < SEQ + 40:
            continue
        rows = order[lo:hi]
        lat = lat_all[rows]; lon = lon_all[rows]
        sog = sog_all[rows] * KN; cog = cog_all[rows]
        ts = ts_all[rows]; mm = mm_s[lo]
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
            r_lat = (lat[tgt_i] - lat[o] - dlat_dr) * KM_PER_DEG
            r_lon = (lon[tgt_i] - lon[o] - dlon_dr) * KM_PER_DEG * np.cos(np.radians(lat[o]))

            # ---- traffic features at (lat[o], lon[o], t0) ----
            cand = neighbors_at(index, t0, lat[o], lon[o])
            if len(cand):
                cand = cand[mmsi_all[cand] != mm]
            if len(cand):
                # nearest-in-time posit per neighbor mmsi within +-SLAB
                dt_c = np.abs(ts_all[cand] - t0)
                keep = dt_c <= SLAB
                cand, dt_c = cand[keep], dt_c[keep]
            if len(cand):
                o_ = np.argsort(dt_c)
                cand = cand[o_]
                _, first = np.unique(mmsi_all[cand], return_index=True)
                cand = cand[first]
            cos0 = np.cos(np.radians(lat[o]))
            if len(cand):
                dy = (lat_all[cand] - lat[o]) * KM_PER_DEG
                dx = (lon_all[cand] - lon[o]) * KM_PER_DEG * cos0
                dist = np.hypot(dx, dy)
                inr = dist < 12.0
                cand, dx, dy, dist = cand[inr], dx[inr], dy[inr], dist[inr]
            if len(cand):
                n2 = int((dist < 2).sum()); n5 = int((dist < 5).sum())
                n10 = int((dist < 10).sum())
                d_nn = float(dist.min())
                anch5 = int(((sog_all[cand] < 0.5) & (dist < 5)).sum())
                # CPA/TCPA vs each neighbor (km, hours)
                sp_n = sog_all[cand] * KN * 3.6            # km/h
                cg_n = cog_all[cand]
                vnx = sp_n * np.sin(cg_n); vny = sp_n * np.cos(cg_n)
                v0x = sog[o] * 3.6 * np.sin(cog[o]); v0y = sog[o] * 3.6 * np.cos(cog[o])
                rvx, rvy = vnx - v0x, vny - v0y
                rv2 = rvx ** 2 + rvy ** 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    tcpa = np.where(rv2 > 1e-6, -(dx * rvx + dy * rvy) / rv2, 0.0)
                tcpa = np.clip(tcpa, 0.0, 0.5)             # within 30 min
                cpa = np.hypot(dx + rvx * tcpa, dy + rvy * tcpa)
                approaching = (tcpa > 1e-6) & (tcpa < 0.5)
                min_cpa = float(cpa[approaching].min()) if approaching.any() else 99.0
                # crossing geometry for the min-CPA approacher
                crossing = 0
                if approaching.any():
                    k = np.flatnonzero(approaching)[int(np.argmin(cpa[approaching]))]
                    dang = np.degrees(np.abs(((cg_n[k] - cog[o]) + np.pi) %
                                              (2 * np.pi) - np.pi))
                    crossing = int(min_cpa < 1.0 and 30.0 <= dang <= 150.0)
            else:
                n2 = n5 = n10 = anch5 = 0
                d_nn, min_cpa, crossing = 30.0, 99.0, 0

            res.append(float(np.hypot(r_lat, r_lon)))
            sogs.append(float(sog[o]))
            feat["n2"].append(n2); feat["n5"].append(n5); feat["n10"].append(n10)
            feat["anch5"].append(anch5); feat["d_nn"].append(d_nn)
            feat["min_cpa"].append(min_cpa); feat["crossing"].append(crossing)
            if len(res) >= MAX_WINDOWS:
                break
        if len(res) >= MAX_WINDOWS:
            break
    if not res:
        return None
    out = dict(res=np.array(res, dtype=np.float32),
               sog=np.array(sogs, dtype=np.float32))
    for k, v in feat.items():
        out[k] = np.array(v, dtype=np.float32)
    return out


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def ols_r2(X, y):
    X = np.column_stack([np.ones_like(y), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_res = float(np.sum((y - X @ beta) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / max(ss_tot, 1e-12)


def main():
    print(f"C1-TRAFFIC | tau={TAU/3600:.1f}h, {len(DAYS)} days")
    parts = []
    for d in DAYS:
        try:
            r = compute_day(d)
        except Exception as ex:
            print(f"  {d}: skipped ({ex!r})")
            continue
        if r is None:
            print(f"  {d}: no data")
            continue
        print(f"  {d}: n={len(r['res'])}", flush=True)
        parts.append(r)
    if not parts:
        raise SystemExit("no data")

    keys = ["res", "sog", "n2", "n5", "n10", "anch5", "d_nn", "min_cpa", "crossing"]
    D = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    res, sog = D["res"], D["sog"]
    ok = np.isfinite(res)
    for k in keys:
        D[k] = D[k][ok]
    res, sog = D["res"], D["sog"]
    print(f"\npooled n={len(res):,} | residual@2h median {np.median(res):.2f} "
          f"p90 {np.percentile(res, 90):.2f} km")
    np.savez("/home/paul/data/trackfm/c1_traffic_arrays.npz", **D)

    hard = res >= np.percentile(res, 90)
    conf = np.column_stack([sog, sog ** 2])

    def partial_r(x, y, z):
        Z = np.column_stack([np.ones_like(y), z])
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return pearson(rx, ry)

    # transform skewed features for regression sanity
    X_feats = {
        "log1p_n5": np.log1p(D["n5"]),
        "log1p_n10": np.log1p(D["n10"]),
        "log1p_anch5": np.log1p(D["anch5"]),
        "inv_dnn": 1.0 / (1.0 + D["d_nn"]),
        "inv_cpa": 1.0 / (1.0 + D["min_cpa"]),
        "crossing": D["crossing"],
    }
    print(f"\nPearson r vs residual (raw | sog-partialed | sog-partialed hard):")
    out = {"n": int(len(res))}
    for name, x in X_feats.items():
        r_raw = pearson(x, res)
        r_p = partial_r(x, res, conf)
        r_ph = partial_r(x[hard], res[hard], conf[hard])
        print(f"  {name:12} {r_raw:+.3f} | {r_p:+.3f} | {r_ph:+.3f}")
        out[f"r_{name}"] = [round(r_raw, 4), round(r_p, 4), round(r_ph, 4)]

    def zs(X):
        return (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)

    X_tr = zs(np.column_stack(list(X_feats.values())))
    X_sog = zs(conf)
    for label, X in [("sog only", X_sog),
                     ("traffic only", X_tr),
                     ("sog + traffic", np.column_stack([X_sog, X_tr]))]:
        r2a = ols_r2(X, res)
        r2h = ols_r2(X[hard], res[hard])
        print(f"OLS R^2 {label:16} overall {r2a:.4f}  hard {r2h:.4f}")
        out[f"r2_{label.replace(' ', '_').replace('+', 'p')}"] = [round(r2a, 4), round(r2h, 4)]

    # encounter indicator, speed-matched (mirror of tube-C1 storm-crossing)
    print(f"\nclose-encounter indicator (min_cpa < 1 km, approaching), speed-matched:")
    enc = (D["min_cpa"] < 1.0)
    for label, ind in [("encounter(any)", enc),
                       ("encounter(crossing)", D["crossing"] > 0.5)]:
        if ind.sum() < 30:
            print(f"  {label}: too few events ({int(ind.sum())})")
            continue
        raw_lift = float(hard[ind].mean() / max(hard[~ind].mean(), 1e-9))
        lifts, ns = [], []
        edges = np.percentile(sog, [0, 20, 40, 60, 80, 100])
        for i in range(5):
            mq = (sog >= edges[i]) & (sog <= edges[i + 1])
            nc = ind & mq
            if nc.sum() < 10 or ((~ind) & mq).sum() < 100:
                continue
            rn = float(hard[(~ind) & mq].mean())
            if rn > 0:
                lifts.append(float(hard[nc].mean()) / rn); ns.append(int(nc.sum()))
        wl = float(np.average(lifts, weights=ns)) if lifts else float("nan")
        print(f"  {label}: n={int(ind.sum()):,} ({ind.mean()*100:.1f}%)  "
              f"raw lift {raw_lift:.2f}x  speed-matched {wl:.2f}x ({len(lifts)} q)")
        out[label] = dict(n=int(ind.sum()), raw=round(raw_lift, 3),
                          matched=round(wl, 3) if lifts else None)

    json.dump(out, open("/home/paul/data/trackfm/c1_traffic.json", "w"), indent=2)
    print("\nwritten /home/paul/data/trackfm/c1_traffic.json")


if __name__ == "__main__":
    main()
