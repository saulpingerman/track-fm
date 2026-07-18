"""Drain-time unified results table: rescore + conformal + null gate.

Joins ~/data/trackfm/rescore_v2.json (fixgrid p90 per bucket),
conformal_v2.json (calibrated area@90 + coverage), and
dr_null_gate.json (the must-beat floor) into one markdown table in
rescore_v2.py RUNS order — the evidence table for the flagship
recommendation package. Read-only; run anytime.

Usage: python scripts/unified_table.py [out.md]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "rescore_v2", Path(__file__).parent / "rescore_v2.py")
_rescore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rescore)

DATA = Path("/home/paul/data/trackfm")
BUCKETS = ("15m", "30m", "1h", "2h")


def _load(name):
    try:
        return json.load(open(DATA / name))
    except FileNotFoundError:
        return None


def main():
    rescore = _load("rescore_v2.json") or []
    rescore = {r["run"]: r for r in rescore}
    conf = _load("conformal_v2.json") or {}
    null = _load("dr_null_gate.json") or {}

    lines = ["| run | " + " | ".join(f"p90 {b}" for b in BUCKETS)
             + " | " + " | ".join(f"cal90 {b}" for b in BUCKETS)
             + " | cov 2h |",
             "|---|" + "---|" * (2 * len(BUCKETS) + 1)]
    if null:
        row = [str(null[b].get("fixgrid_p90") or "UNRCH") for b in BUCKETS]
        lines.append("| DR NULL (must beat) | " + " | ".join(row)
                     + " |" + " — |" * len(BUCKETS) + " — |")
    for _, run in _rescore.RUNS:
        r, c = rescore.get(run), conf.get(run, {})
        if r is None and not c:
            continue
        p90 = [(r or {}).get("buckets", {}).get(b, {}).get("fixgrid_p90_rank")
               for b in BUCKETS]
        cal = [c.get(b, {}).get("calib_area90_mean_km2") for b in BUCKETS]
        cov = c.get("2h", {}).get("coverage_eval")
        fmt = lambda v: f"{v:.0f}" if isinstance(v, (int, float)) else "—"
        lines.append(f"| {run} | " + " | ".join(fmt(v) for v in p90)
                     + " | " + " | ".join(fmt(v) for v in cal)
                     + f" | {cov if cov is not None else '—'} |")
    md = "\n".join(lines)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(md + "\n")
        print(f"written {sys.argv[1]}")
    print(md)


if __name__ == "__main__":
    main()
