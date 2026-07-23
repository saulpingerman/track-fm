"""Round-2 fine-tuning figures (tier-B strategies, vessel-v2, ETA).

Same visual language as ft_sweep_a_summary.png: orange = modern
MLP-head co-trained, blue = modern linear-head, teal = spectrum head,
slate = older eras/replicas, light gray = controls/baselines; dashed
reference lines for dumb baselines; value labels at bar ends.

Numbers are the canonical ones from paper/results_master.md.

Usage: python scripts/gen_ft_round2_charts.py
Writes docs/figures/{ft_strategy_progression,ft_vessel_v2_leaderboard,ft_eta_mae}.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORANGE, BLUE, TEAL = "#d9772f", "#2878c8", "#1f9e89"
SLATE, LIGHT = "#8494a8", "#c0c6cc"
OUT = "docs/figures"


def strategy_progression():
    encoders = ["large-fixed-mlp", "large-cone-mlp", "xlarge-fixed"]
    lp = [0.254, 0.248, 0.245]
    lpft = [0.346, 0.335, 0.330]
    full = [0.364, 0.363, None]
    shades = {  # one hue per encoder family, darkening with strategy
        "large-fixed-mlp": ["#f2c09a", "#e59a56", ORANGE],
        "large-cone-mlp": ["#a9cbee", "#5f9dd8", BLUE],
        "xlarge-fixed": ["#c3ccd8", "#9fadc0", SLATE],
    }
    fig, ax = plt.subplots(figsize=(10, 6.4))
    width, gap = 0.26, 0.06
    for i, enc in enumerate(encoders):
        vals = [lp[i], lpft[i], full[i]]
        for j, (v, lab) in enumerate(zip(vals, ["LP", "LP-FT", "full FT"])):
            if v is None:
                continue
            x = i + (j - 1) * (width + gap)
            ax.bar(x, v, width, color=shades[enc][j],
                   label=lab if i == 0 else None)
            ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(0.411, ls="--", color="purple", lw=1.5)
    ax.text(2.42, 0.415, "dest-given-origin lookup (.411)\nuses privileged origin label",
            color="purple", fontsize=9, ha="right")
    ax.axhline(0.095, ls="--", color="gray", lw=1.5)
    ax.text(2.42, 0.099, "random-init LP (.095)", color="gray", fontsize=9,
            ha="right")
    ax.annotate("full FT closes the\ngeometry gap: .364 vs .363",
                xy=(0.68, 0.363), xytext=(1.25, 0.30), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.set_xticks(range(3))
    ax.set_xticklabels(encoders, fontsize=11)
    ax.set_ylabel("destination f1_macro (811 classes, test)")
    ax.set_ylim(0, 0.46)
    ax.set_title("Port-destination: fine-tuning strategy ladder — "
                 "LP → LP-FT (+35%) → full FT (encoders converge)",
                 loc="left", fontsize=13)
    ax.legend(title="strategy (bar shade)", loc="upper left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ft_strategy_progression.png", dpi=150)
    print("wrote ft_strategy_progression.png")


def vessel_leaderboard():
    rows = [  # (name, f1, top3, color)
        ("large-fixed-mlp", .3825, .707, ORANGE),
        ("cone-spectrum", .3824, .700, TEAL),
        ("xlarge-fixed", .378, .679, BLUE),
        ("large-cone-mlp", .366, .694, ORANGE),
        ("small-cone", .347, .683, BLUE),
        ("golden-large", .346, .657, SLATE),
        ("exp14-100M", .341, .667, SLATE),
        ("exp11-116M", .341, .644, SLATE),
        ("large-cone", .335, .680, BLUE),
        ("exp14-18M", .334, .655, SLATE),
        ("large-fixed", .322, .667, BLUE),
        ("ctx-geotraffic", .310, .654, BLUE),
        ("exp10-large", .269, .614, SLATE),
        ("random-init", .200, .516, LIGHT),
    ]
    names = [r[0] for r in rows][::-1]
    f1 = [r[1] for r in rows][::-1]
    top3 = [r[2] for r in rows][::-1]
    colors = [r[3] for r in rows][::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7.6), sharey=True)
    a1.barh(names, f1, color=colors)
    for i, v in enumerate(f1):
        a1.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=10)
    a1.axvline(0.151, ls="--", color="gray", lw=1.5)
    a1.set_ylim(-0.7, 14.9)
    a1.text(0.155, 14.0, "kinematic-stats baseline (.151)", color="gray",
            fontsize=9)
    a1.set_xlabel("vessel-type f1_macro (15 classes, test)")
    a1.set_title("Feature quality: linear probe on frozen backbones",
                 loc="left", fontsize=12)
    a1.set_xlim(0, 0.44)
    a2.barh(names, top3, color=colors)
    for i, v in enumerate(top3):
        a2.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=10)
    a2.axvline(0.494, ls="--", color="gray", lw=1.5)
    a2.text(0.498, 14.0, "kinstats (.494)", color="gray", fontsize=9)
    a2.set_xlabel("top-3 accuracy")
    a2.set_title("Deployment view: top-3 accuracy", loc="left", fontsize=12)
    a2.set_xlim(0.45, 0.78)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    a1.annotate("spectrum ties #1 on transfer\n(despite losing containment)",
                xy=(0.388, 12.0), xytext=(0.315, 0.35), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4,
                                connectionstyle="arc3,rad=-0.15"),
                color=TEAL)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (ORANGE, TEAL, BLUE, SLATE, LIGHT)]
    fig.legend(handles, ["modern, MLP-head co-trained", "spectrum head",
                         "modern, linear-head co-trained",
                         "older eras / replicas", "random-init control"],
               loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.suptitle("Vessel-type classification v2: 15 classes, 677k windows, "
                 "vessel-disjoint splits (majority baseline f1 = .009)",
                 x=0.02, ha="left", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(f"{OUT}/ft_vessel_v2_leaderboard.png", dpi=150)
    print("wrote ft_vessel_v2_leaderboard.png")


def eta_chart():
    rows = [("xlarge-fixed", 505, 191, BLUE),
            ("large-cone-mlp", 517, 206, ORANGE),
            ("large-fixed-mlp", 524, 211, ORANGE),
            ("cone-spectrum", 544, 213, TEAL)]
    names = [r[0] for r in rows][::-1]
    mae = [r[1] for r in rows][::-1]
    med = [r[2] for r in rows][::-1]
    colors = [r[3] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.barh(names, mae, color=colors)
    for i, (v, m) in enumerate(zip(mae, med)):
        ax.text(v + 4, i, f"{v} min  (median {m})", va="center", fontsize=10)
    ax.set_xlabel("ETA mean absolute error, minutes (linear probe, test) — lower is better")
    ax.set_xlim(0, 640)
    ax.set_title("ETA regression: ordering REVERSES vs classification — "
                 "capacity wins (116M best)", loc="left", fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ft_eta_mae.png", dpi=150)
    print("wrote ft_eta_mae.png")


if __name__ == "__main__":
    strategy_progression()
    vessel_leaderboard()
    eta_chart()
