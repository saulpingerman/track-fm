"""Grouped bar chart: old-era models vs the campaign's best, fixgrid p90.

Reads rescore_v2.json (uniform v2 harness). Log y — the span is 4 to
2,532 cells. Old era muted, campaign accented; coverage noted in labels.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUCKETS = ["15m", "30m", "1h", "2h"]
MODELS = [   # (run key, label, color)
    ("recovered-exp11-116M",
     "exp-11 116M (draft model, 69d, h400)", "#b0b0b8"),
    ("recovered-exp14-18M-grid12",
     "exp-14 18M ±1.2° (1yr)", "#8899aa"),
    ("recovered-exp14-100M-h800",
     "exp-14 100M ±0.6° (1yr)", "#5f7a99"),
    ("scaling-xlarge-50M",
     "NEW xlarge-fixed 116M (81% cov)", "#2f7fd4"),
    ("scaling-large-cone-mlp-50M",
     "NEW large-cone-mlp 18M (100% cov)", "#d4762f"),
]

data = {r["run"]: r for r in
        json.load(open("/home/paul/data/trackfm/rescore_v2.json"))}

def draw(models, log, out, ylab_extra, title):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(BUCKETS))
    w = 0.8 / len(models)
    for i, (run, label, color) in enumerate(models):
        vals = [data[run]["buckets"][b]["fixgrid_p90_rank"] for b in BUCKETS]
        pos = x + (i - (len(models) - 1) / 2) * w
        ax.bar(pos, vals, w, label=label, color=color, zorder=2)
        for p, v in zip(pos, vals):
            ax.annotate(f"{v:.0f}", (p, v), ha="center", va="bottom",
                        fontsize=10, xytext=(0, 2), textcoords="offset points")
    if log:
        ax.set_yscale("log")
        ax.set_ylim(3, 6000)
    else:
        ax.set_ylim(0, None)
    ax.set_xticks(x, [f"{b} forecast" for b in BUCKETS], fontsize=13)
    ax.set_ylabel("grid cells searched to contain 90% of vessels\n"
                  f"(0.6 km² cells, ±0.3° population{ylab_extra})", fontsize=13)
    ax.set_title(title, fontsize=15, loc="left")
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", which="both", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fafafa")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


draw(MODELS, log=True, out="docs/figures/old_vs_new_containment.png",
     ylab_extra=", log scale",
     title="Containment, project history vs current best "
           "(uniform metrics-v2 rescore, val)")
draw(MODELS[1:], log=False,
     out="docs/figures/old_vs_new_containment_linear.png",
     ylab_extra="",
     title="Containment, exp-14 era vs current best "
           "(uniform metrics-v2 rescore, val)")
