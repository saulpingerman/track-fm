"""Score the 2x2 {geometry} x {head} matrix on ONE physical basis.

Containment in km^2-to-capture-90% per time bucket (Paul's metric), with
ceilings reported separately so a geometry that clips long-horizon targets
reads as "unreachable", not as a good number. Long horizons are the point.
"""
import json
import sys

from trackfm.config import PretrainConfig, load_config
from trackfm.eval.xgeometry import score_geometry

CKPT = "/home/paul/data/trackfm/checkpoints"
CELLS = [
    ("fixed", "fourier", "scaling_small_50M", "scaling-small-50M"),
    ("fixed", "direct", "scaling_small_direct_50M", "scaling-small-direct-50M"),
    ("cone", "fourier", "scaling_small_cone_50M", "scaling-small-cone-50M"),
    ("cone", "direct", "scaling_small_cone_direct_50M", "scaling-small-cone-direct-50M"),
]
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"

results = []
for geom, head, cfg_name, run in CELLS:
    cfg = load_config(f"configs/pretrain/{cfg_name}.yaml", PretrainConfig)
    try:
        r = score_geometry(f"{CKPT}/{run}/best.pt", cfg, split=SPLIT, max_batches=120)
        results.append(r)
    except Exception as e:
        print(f"SKIP {geom}+{head}: {e}")

out_path = "/home/paul/data/trackfm/xgeom_matrix.json"
json.dump(results, open(out_path, "w"), indent=2)

print(f"\n{'geom+head':<16} {'bucket':>6} {'ceiling':>8} {'km2@90':>10} {'medrank':>8} {'cap@10':>7}")
for r in results:
    tag = f"{r['geometry']}+{r['head']}"
    for b, d in r["buckets"].items():
        km2 = d["km2_to_capture90"]
        km2s = f"{km2:.0f}" if km2 is not None else "UNREACH"
        cap = d.get("capture@10")
        print(f"{tag:<16} {b:>6} {d['ceiling']:>8.3f} {km2s:>10} "
              f"{str(d['median_rank']):>8} {cap if cap is None else round(cap,2):>7}")
print(f"\nwritten {out_path}")
