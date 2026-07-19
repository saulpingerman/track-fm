"""The coverage story at 2h: what happens when you stop censoring.

One bucket (2h), three evaluation frames of the same 0.6 km² cells,
widening the counted vessel population. Cone stays flat; every fixed
box cliffs when its escapees enter the denominator. Linear scale on
purpose — the 970 should look as brutal as it is.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

wide = json.load(open("/home/paul/data/trackfm/widegrid_compare.json"))
base = {r["run"]: r["buckets"]["2h"]["fixgrid_p90_rank"]
        for r in json.load(open("/home/paul/data/trackfm/rescore_v2.json"))}

MODELS = [  # (run, label, color)
    ("scaling-large-cone-mlp-50M", "cone-mlp 18M (full coverage)", "#d4762f"),
    ("scaling-xlarge-cone-50M", "cone 116M, linear head", "#e3a76f"),
    ("recovered-exp14-100M-grid12", "exp-14 100M, ±1.2° box", "#8899aa"),
    ("recovered-exp14-100M-h800", "exp-14 100M, ±0.6° box", "#5f7a99"),
    ("scaling-xlarge-50M", "xlarge-fixed 116M, ±0.3° box", "#2f7fd4"),
]
FRAMES = [("0.3", "±0.3° frame\n(81% of vessels)"),
          ("0.6", "±0.6° frame\n(wider population)"),
          ("1.2", "±1.2° frame\n(~all vessels)")]

fig, ax = plt.subplots(figsize=(11.5, 7))
x = np.arange(len(FRAMES))
w = 0.8 / len(MODELS)
for i, (run, label, color) in enumerate(MODELS):
    vals = [base[run] if f == "0.3" else wide[f][run]["2h"]
            for f, _ in FRAMES]
    pos = x + (i - (len(MODELS) - 1) / 2) * w
    ax.bar(pos, vals, w, label=label, color=color, zorder=2)
    for p, v in zip(pos, vals):
        ax.annotate(f"{v:.0f}", (p, v), ha="center", va="bottom",
                    fontsize=10.5, xytext=(0, 2), textcoords="offset points")

ax.set_ylim(0, 1080)
ax.set_xticks(x, [lab for _, lab in FRAMES], fontsize=13)
ax.set_ylabel("grid cells searched to contain 90% of vessels\n"
              "at 2 h (0.6 km² cells)", fontsize=13)
ax.set_title("2-hour containment as the counted population widens:\n"
             "cone holds, every fixed box hits its wall",
             fontsize=15, loc="left")
ax.legend(fontsize=11.5, loc="upper left", framealpha=0.95)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.set_facecolor("#fafafa")

# narrative callouts
ax.annotate("full coverage: barely moves\n(58 → 92 → 98)",
            xy=(2 - 2 * w, 110), xytext=(0.72, 470), fontsize=11.5,
            color="#b85f1a", ha="center",
            arrowprops=dict(arrowstyle="->", color="#b85f1a",
                            connectionstyle="arc3,rad=-0.15"))
ax.annotate("best model on the censored metric,\n10× worse than cone on everyone",
            xy=(2 + 2 * w, 970), xytext=(1.62, 840), fontsize=11.5,
            color="#1d5fa8", ha="right",
            arrowprops=dict(arrowstyle="->", color="#1d5fa8"))
fig.tight_layout()
out = "docs/figures/coverage_story_2h.png"
fig.savefig(out, dpi=130)
print("wrote", out)
