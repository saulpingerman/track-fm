"""Head/backbone dissociation scatter: forecasting containment vs transfer.

X = 2h fixgrid p90 containment (0.6 km2 cells, log scale, lower=better);
Y = vessel-type-v2 linear-probe f1_macro (higher=better). If containment
quality and representation quality were one axis, points would fall on a
monotone curve; the spectrum point (worst-of-modern containment, tied-best
transfer) and ctx-geotraffic (good containment, poor transfer) break it.

Open markers = fixed +/-0.3 canvases whose 2h containment is computed on
the 81% of vessels they cover (home metric); filled = full population.

Usage: python scripts/gen_dissociation_scatter.py
Writes docs/figures/dissociation_scatter.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORANGE, BLUE, TEAL, SLATE = "#d9772f", "#2878c8", "#1f9e89", "#8494a8"

#      name, 2h p90 (cells), vessel f1, color, censored(open marker)
PTS = [
    ("large-fixed-mlp", 39, .3825, ORANGE, True),
    ("cone-spectrum", 67, .3824, TEAL, False),
    ("xlarge-fixed", 52, .378, BLUE, True),
    ("large-cone-mlp", 58, .366, ORANGE, False),
    ("small-cone", 120, .347, BLUE, False),
    ("golden-large", 86, .346, SLATE, True),
    ("exp14-100M", 129, .341, SLATE, True),
    ("large-cone", 87, .335, BLUE, False),
    ("exp14-18M", 133, .334, SLATE, True),
    ("large-fixed", 64, .322, BLUE, True),
    ("ctx-geotraffic", 102, .310, BLUE, False),
]
OFFS = {  # label offsets (pts) to avoid collisions
    "large-fixed-mlp": (8, 2), "cone-spectrum": (8, 2),
    "xlarge-fixed": (8, -3), "large-cone-mlp": (8, 2),
    "small-cone": (8, 2), "golden-large": (-8, 6),
    "exp14-100M": (8, -2), "large-cone": (8, -8),
    "exp14-18M": (8, 4), "large-fixed": (8, -4),
    "ctx-geotraffic": (-8, -12),
}


def main():
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for name, x, y, c, censored in PTS:
        ax.scatter(x, y, s=110, color="white" if censored else c,
                   edgecolor=c, linewidth=2.2, zorder=3)
        dx, dy = OFFS[name]
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=10,
                    ha="left" if dx > 0 else "right")
    ax.set_xscale("log")
    ax.set_xticks([40, 60, 80, 100, 130])
    ax.set_xticklabels(["40", "60", "80", "100", "130"])
    ax.invert_xaxis()                      # right = better forecasting
    ax.set_xlabel("2h containment: p90 rank, 0.6 km$^2$ cells "
                  "(log, lower/right = better) — open markers scored on "
                  "censored 81% population")
    ax.set_ylabel("feature quality: ship-type classification from FROZEN\n"
                  "features (linear-probe f1_macro, 15 classes; higher = better)")
    ax.set_title("A model's forecast quality does not predict its feature quality",
                 loc="left", fontsize=14)
    ax.text(0.02, 0.975,
            "Each point = one pretrained encoder.\n"
            "x: how small an area its 2h forecasts pin a vessel to (its actual job).\n"
            "y: how well a linear classifier reads SHIP TYPE off its frozen features\n"
            "    (a task it was never trained for — the foundation-model test).",
            transform=ax.transAxes, va="top", fontsize=9.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#ccc"))
    # the two story arrows
    ax.annotate("spectrum: worst modern containment,\ntied-best transfer",
                xy=(68.5, .3805), xytext=(122, .3625), fontsize=10, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4,
                                connectionstyle="arc3,rad=0.12", shrinkB=3))
    ax.annotate("conditioned pretrain: good containment,\nworst modern transfer",
                xy=(101, .3125), xytext=(97, .3225), fontsize=10, color=BLUE,
                ha="left",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4,
                                shrinkB=4))
    # marker-style legend (was only explained in the axis label)
    filled = plt.Line2D([], [], marker="o", ls="", color="#666",
                        markersize=10, label="full population (100%)")
    open_m = plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                        markeredgecolor="#666", markeredgewidth=2,
                        markersize=10,
                        label="censored home metric (81%, fixed ±0.3°)")
    ax.legend(handles=[filled, open_m], loc="lower right", fontsize=9.5,
              framealpha=0.9)
    ax.grid(alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/figures/dissociation_scatter.png", dpi=150)
    print("wrote dissociation_scatter.png")


if __name__ == "__main__":
    main()
