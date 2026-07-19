"""Cone vs exp-14 on the exp-14 models' NATIVE reference frames.

Scores selected checkpoints on ±0.6° and ±1.2° evaluation boxes with the
SAME 0.6 km² physical cells as the standard ±0.3° fixgrid (128² and 256²
cells) — ranks stay area-comparable across frames. Wider frames admit
the faster-vessel population the ±0.3° metric censors: coverage becomes
visible instead of assumed. A ±0.3°-canvas fixed model scored on a wider
frame shows its structural wall (no density exists outside its box —
border-replicated values, honest and terrible).

Writes ~/data/trackfm/widegrid_compare.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trackfm.config import PretrainConfig, load_config  # noqa: E402
from trackfm.eval.xgeometry import score_geometry  # noqa: E402

CKPT = "/home/paul/data/trackfm/checkpoints"
MODELS = [
    ("recovered_exp14_18M_h800", "recovered-exp14-18M-h800"),
    ("recovered_exp14_18M_grid12", "recovered-exp14-18M-grid12"),
    ("recovered_exp14_100M_h800", "recovered-exp14-100M-h800"),
    ("recovered_exp14_100M_grid12", "recovered-exp14-100M-grid12"),
    ("scaling_large_cone_mlp_50M", "scaling-large-cone-mlp-50M"),
    ("scaling_xlarge_cone_50M", "scaling-xlarge-cone-50M"),
    ("scaling_xlarge_50M", "scaling-xlarge-50M"),      # the ±0.3 wall exhibit
]
FRAMES = (0.6, 1.2)
BUCKETS = ("15m", "30m", "1h", "2h")


def main():
    out = {}
    for frame in FRAMES:
        out[str(frame)] = {}
        for cfg_name, run in MODELS:
            cfg = load_config(f"configs/pretrain/{cfg_name}.yaml",
                              PretrainConfig)
            r = score_geometry(f"{CKPT}/{run}/best.pt", cfg, split="val",
                               max_batches=120, fixgrid_restrict_deg=frame)
            row = {b: r["buckets"][b].get("fixgrid_p90_rank")
                   for b in BUCKETS}
            row["avail_2h"] = r["buckets"]["2h"].get("n")
            out[str(frame)][run] = row
            print(f"±{frame}° {run:<34}" + " ".join(
                f"{row[b]:>8.1f}" if row[b] else f"{'UNRCH':>8}"
                for b in BUCKETS), flush=True)
    json.dump(out, open("/home/paul/data/trackfm/widegrid_compare.json", "w"),
              indent=2)
    print("written ~/data/trackfm/widegrid_compare.json")


if __name__ == "__main__":
    main()
