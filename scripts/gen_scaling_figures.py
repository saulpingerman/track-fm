"""Regenerate the fixed-series scaling figures (MLflow + rescore_v2.json).

Fig A  docs/figures/flops_vs_loss_gpt3style.png
    Smoothed train-loss curves vs cumulative training compute (PF-days),
    GPT-3-style viridis spectrum by model size, dashed power-law frontier.
Fig B  docs/figures/scaling_ce_vs_search2h_7pt.png
    Best val CE vs v2 fixgrid p90 2h search cells (uniform rescore_v2
    numbers — NEVER mix trainer-logged pre-v2 values into this figure).

Series: the 8 FIXED-geometry runs nano..xlarge. Cone runs are excluded by
construction: val CE is not comparable across geometries (different
targets/canvas), containment tables are the only cross-geometry venue.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trackfm.config import PretrainConfig, load_config  # noqa: E402
from trackfm.models.factory import build_model  # noqa: E402
from trackfm.training.flops import train_flops_per_sample  # noqa: E402

DB = "/home/paul/data/trackfm/mlflow/mlflow.db"
RESCORE = "/home/paul/data/trackfm/rescore_v2.json"
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

#         run name             config yaml              short  params label
SERIES = [
    ("scaling-nano-50M",   "scaling_nano_50M",   "nano",   "25k"),
    ("scaling-micro-50M",  "scaling_micro_50M",  "micro",  "70k"),
    ("scaling-mini-50M",   "scaling_mini_50M",   "mini",   "244k"),
    ("scaling-tiny-50M",   "scaling_tiny_50M",   "tiny",   "485k"),
    ("scaling-small-50M",  "scaling_small_50M",  "small",  "1.0M"),
    ("scaling-medium-50M", "scaling_medium_50M", "medium", "5.3M"),
    ("scaling-large-50M",  "scaling_large_50M",  "large",  "18.3M"),
    ("scaling-xlarge-50M", "scaling_xlarge_50M", "xlarge", "116M"),
]


def _run_id(db, name):
    row = db.execute(
        'SELECT run_uuid FROM runs r WHERE (SELECT value FROM tags WHERE '
        'run_uuid=r.run_uuid AND key="mlflow.runName")=? '
        'AND r.status="FINISHED" ORDER BY r.start_time DESC', (name,)).fetchone()
    if row is None:
        raise SystemExit(f"no FINISHED run named {name}")
    return row[0]


def _metric(db, rid, key):
    return db.execute(
        "SELECT step, value FROM metrics WHERE run_uuid=? AND key=? "
        "ORDER BY step", (rid, key)).fetchall()


def _ema(vals, beta=0.98):
    out, m = [], None
    for v in vals:
        m = v if m is None else beta * m + (1 - beta) * v
        out.append(m)
    return out


def fig_flops_vs_loss():
    db = sqlite3.connect(DB)
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    cmap = plt.get_cmap("viridis")
    logp = []
    curves = []
    for name, cfg_name, _, label in SERIES:
        cfg = load_config(f"configs/pretrain/{cfg_name}.yaml", PretrainConfig)
        fps = train_flops_per_sample(cfg.model, cfg.train.num_horizon_samples)
        bs = cfg.train.batch_size
        pts = _metric(db, _run_id(db, name), "train_loss")
        steps = np.array([s for s, _ in pts], dtype=float)
        loss = _ema([v for _, v in pts])
        pf_days = fps * bs * steps / 1e15 / 86400.0
        n = sum(p.numel()
                for p in build_model(cfg.model, cfg.normalization).parameters())
        logp.append(np.log10(n))
        curves.append((pf_days, np.array(loss), label))

    lo, hi = min(logp), max(logp)
    label_off = {"485k": (10, 12), "1.0M": (-52, -14), "5.3M": (10, 2),
                 "18.3M": (10, -14)}
    all_x, all_y = [], []
    for (x, y, label), lp in zip(curves, logp):
        keep = x > 0
        x, y = x[keep], y[keep]
        c = cmap(0.9 - 0.85 * (lp - lo) / (hi - lo))
        ax.plot(x, y, color=c, lw=1.6)
        dx, dy = label_off.get(label, (8, 0))
        ax.annotate(label, (x[-1], y[-1]), textcoords="offset points",
                    xytext=(dx, dy), color=c, fontsize=12, va="center")
        all_x.append(x); all_y.append(y)

    # dashed power-law frontier: running minimum over all curves = the
    # compute-efficient frontier; fit only where it IMPROVES (excludes
    # warmup transients and each model's saturation shelf, which
    # otherwise flatten the slope)
    X = np.concatenate(all_x); Y = np.concatenate(all_y)
    order = np.argsort(X)
    Xs, Ys = X[order], Y[order]
    run_min = np.minimum.accumulate(Ys)
    sup = np.diff(run_min, prepend=np.inf) < 0
    sup &= run_min < 2.8                       # drop the earliest transient
    bx, by = np.log10(Xs[sup]), np.log10(run_min[sup])
    slope, icpt = np.polyfit(bx, by, 1)
    # x range: from where the frontier first drops below the y-cap, so
    # nano's ultra-cheap early steps don't drag the axis to 1e-5
    x_lo = Xs[sup][np.argmax(run_min[sup] < 3.0)] * 0.8
    x_hi = X.max() * 1.15
    fx = np.logspace(np.log10(x_lo), np.log10(x_hi), 50)
    ax.plot(fx, 10 ** (icpt + slope * np.log10(fx)), "k--", lw=1.4)
    ax.annotate(f"L ∝ C^{slope:.2f}",
                (fx[36], 10 ** (icpt + slope * np.log10(fx[36])) * 0.96),
                fontsize=14, va="top")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi * 2.2)              # right margin for end labels
    ax.set_ylim(1.4, 3.0)
    ax.set_yticks([1.5, 2, 2.5, 3])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("compute (PetaFLOP/s-days)", fontsize=14)
    ax.set_ylabel("training loss (soft-target CE, smoothed)", fontsize=14)
    ax.set_title("Fixed-geometry scaling, 25k → 116M params\n"
                 "same data, same 50M-sample budget", fontsize=15, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fafafa")
    fig.tight_layout()
    fig.savefig(OUT / "flops_vs_loss_gpt3style.png", dpi=130)
    print("wrote flops_vs_loss_gpt3style.png  slope", round(slope, 3))


def fig_ce_vs_search2h():
    db = sqlite3.connect(DB)
    cells = {r["run"]: r["buckets"]["2h"]["fixgrid_p90_rank"]
             for r in json.load(open(RESCORE))}
    xs, ys, names = [], [], []
    for name, _, short, _ in SERIES:
        rid = _run_id(db, name)
        ce = db.execute('SELECT MIN(value) FROM metrics WHERE run_uuid=? AND '
                        'key="val_loss"', (rid,)).fetchone()[0]
        xs.append(ce); ys.append(cells[name]); names.append(short)

    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.plot(xs, ys, "-", color="#a8c8f0", lw=3, zorder=1)
    ax.scatter(xs, ys, s=110, color="#2f7fd4", zorder=2)
    offsets = {"nano": (14, 4), "xlarge": (-6, -22)}
    for x, y, s in zip(xs, ys, names):
        dx, dy = offsets.get(s, (10, 10))
        ax.annotate(f"{s} · {y:.0f}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=13)
    ax.invert_xaxis()
    ax.set_ylim(0, max(ys) * 1.12)
    span = f"{ys[0]:.0f} → {ys[-1]:.0f} cells ({ys[0] / ys[-1]:.1f}×)"
    ax.annotate(f"25k → 116M params: {span}\nsame data, same "
                "50M-sample budget (metrics v2, uniform rescore)",
                xy=(0.42, 0.86), xycoords="axes fraction", fontsize=13,
                ha="center", color="#555555")
    ax.set_xlabel("validation cross-entropy (lower = better forecasts)",
                  fontsize=14)
    ax.set_ylabel("grid cells searched to contain 90% of vessels\n"
                  "(2 h forecast, 0.6 km² cells, model-priority order)",
                  fontsize=14)
    ax.set_title("Forecast quality compounds into search efficiency",
                 fontsize=16, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fafafa")
    fig.tight_layout()
    fig.savefig(OUT / "scaling_ce_vs_search2h_7pt.png", dpi=130)
    print("wrote scaling_ce_vs_search2h_7pt.png")
    for n, x, y in zip(names, xs, ys):
        print(f"  {n:<8} CE={x:.3f}  2h cells={y:.0f}")


if __name__ == "__main__":
    fig_flops_vs_loss()
    fig_ce_vs_search2h()
