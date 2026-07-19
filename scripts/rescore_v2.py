"""One-command v2 rescoring: every checkpoint, one metric version, val split.

Metrics v2 changed rank/area semantics (audit F2/F3/F5/F6/F7), so any
cross-run table must come from ONE harness pass — never from mixing
MLflow curves logged by different code versions. This script batch-runs
score_geometry (v2) over all named checkpoints on VAL (test is retired
from selection, audit F1) and adds split-conformal calibrated area@90
per bucket (alpha=0.1; calibration on even batches, evaluation on odd).

Usage: python scripts/rescore_v2.py [max_batches]   (GPU required)
Writes ~/data/trackfm/rescore_v2.json + prints the ranked table with the
DR null gate rows for reference.
"""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "/home/paul/projects/trackfm-v2/src")
from trackfm.config import PretrainConfig, load_config
from trackfm.eval.xgeometry import score_geometry

CKPT = "/home/paul/data/trackfm/checkpoints"
RUNS = [
    # (config, checkpoint dir) — extend as the chain drains
    # golden-* = paper-configuration replicas (69-day Jan-Feb 2025 slice,
    # new code) standing in for the lost exp-11 checkpoints; v1 val is
    # clean for them (their window overlaps the RETIRED test split only)
    ("recovered_exp10_large", "recovered-exp10-large"),
    ("recovered_exp10_small", "recovered-exp10-small"),
    ("golden_medium", "golden-medium"),
    ("golden_large", "golden-large"),
    ("scaling_nano_50M", "scaling-nano-50M"),
    ("scaling_micro_50M", "scaling-micro-50M"),
    ("scaling_mini_50M", "scaling-mini-50M"),
    ("scaling_tiny_50M", "scaling-tiny-50M"),
    ("scaling_medium_50M", "scaling-medium-50M"),
    ("scaling_small_cone_50M", "scaling-small-cone-50M"),
    ("scaling_small_cone_F18_50M", "scaling-small-cone-F18-50M"),
    ("scaling_small_cone_F24_50M", "scaling-small-cone-F24-50M"),
    ("scaling_small_cone_mlp_50M", "scaling-small-cone-mlp-50M"),
    ("scaling_small_cone_direct_mlp_50M", "scaling-small-cone-direct-mlp-50M"),
    ("scaling_large_cone_50M", "scaling-large-cone-50M"),
    ("scaling_large_cone_mlp_50M", "scaling-large-cone-mlp-50M"),
    ("scaling_large_50M", "scaling-large-50M"),
    ("scaling_small_50M", "scaling-small-50M"),
    ("scaling_small_direct_50M", "scaling-small-direct-50M"),
    ("scaling_small_cone_direct_50M", "scaling-small-cone-direct-50M"),
    ("scaling_xlarge_cone_50M", "scaling-xlarge-cone-50M"),
    ("scaling_xlarge_50M", "scaling-xlarge-50M"),
    ("scaling_small_cone_ctx_geo_50M", "scaling-small-cone-ctx-geo-50M"),
    ("scaling_small_cone_ctx_geotraffic_50M", "scaling-small-cone-ctx-geotraffic-50M"),
    ("scaling_small_cone_sig10_50M", "scaling-small-cone-sig10-50M"),
    ("scaling_small_cone_sig05_50M", "scaling-small-cone-sig05-50M"),
    ("scaling_small_cone_bs1024_50M", "scaling-small-cone-bs1024-50M"),
    ("scaling_small_cone_trope_50M", "scaling-small-cone-trope-50M"),
    ("scaling_large_fixed_R125_50M", "scaling-large-fixed-R125-50M"),
    ("scaling_small_cone_mdn_50M", "scaling-small-cone-mdn-50M"),
    ("scaling_large_fixed_R125_mlp_50M", "scaling-large-fixed-R125-mlp-50M"),
    ("scaling_large_mlp_50M", "scaling-large-mlp-50M"),
]

MAXB = int(sys.argv[1]) if len(sys.argv) > 1 else 120
# optional argv[2]: comma-separated run names to (re)score; others keep
# their existing JSON entries (merge, never clobber)
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None


def main():
    out_path = "/home/paul/data/trackfm/rescore_v2.json"
    try:
        merged = {r["run"]: r for r in json.load(open(out_path))}
    except FileNotFoundError:
        merged = {}
    results = []
    for cfg_name, run in RUNS:
        if ONLY is not None and run not in ONLY:
            continue
        ckpt = f"{CKPT}/{run}/best.pt"
        try:
            cfg = load_config(f"configs/pretrain/{cfg_name}.yaml", PretrainConfig)
            r = score_geometry(ckpt, cfg, split="val", max_batches=MAXB)
            r["run"] = run
            results.append(r)
            print(f"scored {run}")
        except FileNotFoundError:
            print(f"skip {run}: no checkpoint yet")
        except Exception as e:
            print(f"SKIP {run}: {e!r}")

    merged.update({r["run"]: r for r in results})
    # table + file in RUNS order, keeping entries scored in prior passes
    results = [merged[run] for _, run in RUNS if run in merged]
    json.dump(results, open(out_path, "w"), indent=2)

    try:
        null = json.load(open("/home/paul/data/trackfm/dr_null_gate.json"))
    except FileNotFoundError:
        null = {}
    print(f"\n=== METRICS V2, val, {MAXB} batches | fixgrid p90 (0.6km2, ±0.3°) ===")
    print(f"{'run':<38} {'15m':>7} {'30m':>7} {'1h':>7} {'2h':>7}")
    if null:
        bar = [null[b].get('fixgrid_p90') for b in ('15m', '30m', '1h', '2h')]
        print(f"{'DR NULL GATE (must beat)':<38} " +
              " ".join(f"{v if v else 'UNRCH':>7}" for v in bar))
    for r in results:
        row = [r["buckets"][b].get("fixgrid_p90_rank") for b in ("15m", "30m", "1h", "2h")]
        print(f"{r['run']:<38} " +
              " ".join(f"{v:>7.1f}" if v else f"{'UNRCH':>7}" for v in row))
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
