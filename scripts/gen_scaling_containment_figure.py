"""Scaling figure: 2h containment vs encoder params, 50M-sample budget.

Story: the fixed-geometry linear-head series saturates by 18M, and the
projector-head intervention at 18M is worth more than the 18M->116M
scale step. Fixed +/-0.3 series is scored on its 81%-covered home
metric (consistent within series); cone points cover 100% of vessels.
Numbers: paper/results_master.md (rescore_v2 harness).

Usage: python scripts/gen_scaling_containment_figure.py
Writes docs/figures/scaling_containment.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORANGE, BLUE, SLATE = "#d9772f", "#2878c8", "#8494a8"

FIXED = [(0.004224, 335, "nano"), (0.84, 109, "small"), (4.94, 75, "medium"),
         (17.79, 64, "large"), (115.18, 52, "xlarge")]
CONE = [(0.84, 120, "small-cone"), (17.79, 87, "large-cone"),
        (115.18, 85, "xlarge-cone")]


def main():
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fx = [p for p, _, _ in FIXED]
    fy = [v for _, v, _ in FIXED]
    ax.plot(fx, fy, "o-", color=SLATE, lw=2, ms=9,
            label="fixed ±0.3° geometry, linear head (81% home metric)")
    for p, v, n in FIXED:
        ax.annotate(f"{n}  ({v})", (p, v), textcoords="offset points",
                    xytext=(8, 6), fontsize=9.5, color=SLATE)
    cx = [p for p, _, _ in CONE]
    cy = [v for _, v, _ in CONE]
    ax.plot(cx, cy, "s--", color=BLUE, lw=1.8, ms=9,
            label="cone geometry, linear head (100% population)")
    for p, v, n in CONE:
        ax.annotate(f"{n}  ({v})", (p, v), textcoords="offset points",
                    xytext=(8, 6), fontsize=9.5, color=BLUE)

    # projector interventions: vertical arrows at fixed x
    for x0, y0, y1, color, lab in [
            (17.79, 64, 39, ORANGE, "+projector: 64→39"),
            (17.79, 87, 58, ORANGE, "+projector: 87→58"),
            (0.84, 120, 121, SLATE, "+projector: null at small")]:
        ax.annotate("", xy=(x0, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
    ax.scatter([17.79], [39], marker="*", s=280, color=ORANGE, zorder=4,
               label="18M + projector head (the champion recipe)")
    ax.scatter([17.79], [58], marker="*", s=280, color=BLUE, zorder=4,
               edgecolor="white")
    ax.annotate("fixed+projector (39)\ncensored 81%", (17.79, 39),
                textcoords="offset points", xytext=(10, -16), fontsize=9.5,
                color=ORANGE)
    ax.annotate("cone+projector (58)\nfull population", (17.79, 58),
                textcoords="offset points", xytext=(-125, -6), fontsize=9.5,
                color=BLUE)
    ax.annotate("one hidden layer at 18M beats\nthe entire 18M→116M scale step",
                xy=(115.18, 52), xytext=(6.5, 42), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1))
    ax.annotate("+projector: null at small (121)", (0.84, 120),
                textcoords="offset points", xytext=(10, -14), fontsize=9,
                color=SLATE, style="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([40, 60, 100, 200, 300])
    ax.set_yticklabels(["40", "60", "100", "200", "300"])
    ax.set_xlabel("encoder parameters (M, log)")
    ax.set_ylabel("2h containment: p90 rank, 0.6 km$^2$ cells (log, lower = better)")
    ax.set_title("Scale saturates by 18M at the 50M-sample budget; "
                 "the head, not width, is the lever", loc="left", fontsize=13)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/figures/scaling_containment.png", dpi=150)
    print("wrote scaling_containment.png")


if __name__ == "__main__":
    main()
