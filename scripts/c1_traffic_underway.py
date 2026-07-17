"""Re-analyze C1-traffic on UNDERWAY vessels only (sog >= 2 kn).
Moored/anchored windows (residual ~ 0) dominate the pooled set and
flip/dilute every statistic."""
import json
import numpy as np

D = dict(np.load("/home/paul/data/trackfm/c1_traffic_arrays.npz"))
KN = 0.514444
uw = D["sog"] >= 2 * KN                       # sog stored in m/s
print(f"underway windows: {uw.sum():,} / {len(uw):,} ({uw.mean()*100:.0f}%)")
for k in D:
    D[k] = D[k][uw]
res, sog = D["res"], D["sog"]
print(f"residual@2h: median {np.median(res):.2f}  p90 {np.percentile(res,90):.2f}  "
      f"p99 {np.percentile(res,99):.2f} km")

hard = res >= np.percentile(res, 90)
conf = np.column_stack([sog, sog ** 2])


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def partial_r(x, y, z):
    Z = np.column_stack([np.ones_like(y), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearson(rx, ry)


def ols_r2(X, y):
    X = np.column_stack([np.ones_like(y), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss = float(np.sum((y - X @ beta) ** 2))
    st = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss / max(st, 1e-12)


feats = {
    "log1p_n5": np.log1p(D["n5"]),
    "log1p_n10": np.log1p(D["n10"]),
    "log1p_anch5": np.log1p(D["anch5"]),
    "inv_dnn": 1.0 / (1.0 + D["d_nn"]),
    "inv_cpa": 1.0 / (1.0 + D["min_cpa"]),
    "crossing": D["crossing"],
}
print("\nPearson r (raw | sog-partialed | sog-partialed hard) — UNDERWAY only:")
for name, x in feats.items():
    print(f"  {name:12} {pearson(x, res):+.3f} | {partial_r(x, res, conf):+.3f} | "
          f"{partial_r(x[hard], res[hard], conf[hard]):+.3f}")


def zs(X):
    return (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)


X_tr = zs(np.column_stack(list(feats.values())))
X_sog = zs(conf)
for label, X in [("sog only", X_sog), ("traffic only", X_tr),
                 ("sog + traffic", np.column_stack([X_sog, X_tr]))]:
    print(f"OLS R^2 {label:14} overall {ols_r2(X, res):.4f}  "
          f"hard {ols_r2(X[hard], res[hard]):.4f}")

print("\nencounter indicators, speed-matched within underway:")
for label, ind in [("cpa<1km", D["min_cpa"] < 1.0),
                   ("cpa<0.5km", D["min_cpa"] < 0.5),
                   ("crossing", D["crossing"] > 0.5)]:
    raw = float(hard[ind].mean() / max(hard[~ind].mean(), 1e-9))
    lifts, ns = [], []
    edges = np.percentile(sog, [0, 20, 40, 60, 80, 100])
    for i in range(5):
        mq = (sog >= edges[i]) & (sog <= edges[i + 1])
        nc = ind & mq
        if nc.sum() < 10 or ((~ind) & mq).sum() < 100:
            continue
        rn = float(hard[(~ind) & mq].mean())
        if rn > 0:
            lifts.append(float(hard[nc].mean()) / rn)
            ns.append(int(nc.sum()))
    wl = float(np.average(lifts, weights=ns)) if lifts else float("nan")
    print(f"  {label:10} n={int(ind.sum()):,} ({ind.mean()*100:.1f}%)  "
          f"raw {raw:.2f}x  matched {wl:.2f}x")

# decile profile: does traffic intensity rise with difficulty?
print("\nresidual decile -> mean traffic (underway):")
edges = np.percentile(res, np.linspace(0, 100, 11))
print(f"  {'dec':>3} {'med_res':>8} {'n5':>5} {'d_nn':>6} {'cpa<1k%':>8} {'anch5':>6}")
for i in range(10):
    m = (res >= edges[i]) & (res < edges[i + 1]) if i < 9 else (res >= edges[i])
    if m.sum() == 0:
        continue
    print(f"  {i:>3} {np.median(res[m]):>8.2f} {D['n5'][m].mean():>5.1f} "
          f"{D['d_nn'][m].mean():>6.2f} {(D['min_cpa'][m] < 1).mean()*100:>7.1f}% "
          f"{D['anch5'][m].mean():>6.2f}")
