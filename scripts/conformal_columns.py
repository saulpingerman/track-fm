"""Split-conformal calibrated area@90 per checkpoint per bucket.

The drain-time companion to rescore_v2.py: for every checkpoint in its
RUNS list, computes the calibrated search area (km^2) of the tau-mass
HDR at alpha=0.1, with calibration on EVEN val batches and evaluation
on ODD ones (disjoint by construction; same loader order as rescore so
the split is deterministic). Reports empirical coverage on the eval
half — the conformal guarantee says >= 0.9 up to finite-sample noise;
a large shortfall means non-exchangeability (drift), not a bug.

Cone canvases have PER-SAMPLE cell areas (cell_deg = 2 R(t)/G), so the
area is computed per sample before averaging; fixed grids use the
constant cell. Off-canvas truths are dropped from both halves and
reported as ceiling (same convention as rescore).

Usage: python scripts/conformal_columns.py [max_batches] [batch_size] [only,names]
Writes ~/data/trackfm/conformal_v2.json (merge, never clobber).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/paul/projects/trackfm-v2/src")
from trackfm.config import PretrainConfig, load_config
from trackfm.datasets.loaders import ShardedWindowDataset
from trackfm.eval.conformal import (calibrated_coverage, conformal_threshold,
                                    hdr_mass_at_truth, hdr_region_cells)
from trackfm.eval.horizons import TIME_BUCKETS, time_bucket_indices
from trackfm.models.factory import build_model
from trackfm.training.losses import cone_elapsed_seconds, cone_ranges

LAT0 = 56.25
KM_PER_DEG = 111.32

_spec = importlib.util.spec_from_file_location(
    "rescore_v2", Path(__file__).parent / "rescore_v2.py")
_rescore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rescore)
RUNS, CKPT = _rescore.RUNS, _rescore.CKPT


@torch.no_grad()
def conformal_for(cfg_name: str, ckpt: str, max_batches: int,
                  batch_size: int | None, alpha: float = 0.1) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(f"configs/pretrain/{cfg_name}.yaml", PretrainConfig)
    m, t = cfg.model, cfg.train
    cone = m.grid_mode == "cone"
    G = m.grid_size
    state = torch.load(Path(ckpt).expanduser(), map_location="cpu",
                       weights_only=False)
    model = build_model(m, cfg.normalization, t.max_horizon,
                        len(TIME_BUCKETS)).to(device).eval()
    model.load_state_dict(state["model"])

    ds = ShardedWindowDataset(cfg.data_dir / "val",
                              batch_size=batch_size or t.batch_size,
                              shuffle_shards=False)
    loader = DataLoader(ds, batch_size=None, num_workers=2,
                        pin_memory=device.type == "cuda")

    buckets = list(TIME_BUCKETS)
    acc = {b: {"cal": [], "ev": [], "area": [], "n_all": 0, "n_on": 0}
           for b in buckets}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        step_idx, valid = time_bucket_indices(batch, cfg.normalization,
                                              m.max_seq_len)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            ld, tgt = model.forward_at_indices(batch, step_idx)
        ld, tgt = ld.float(), tgt.float()
        if cone:
            el = cone_elapsed_seconds(batch, step_idx, m.max_seq_len,
                                      cfg.normalization.dt_scale, causal=False)
            R = cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)   # (B, H)
            tgt_canvas = tgt / R
        else:
            R = torch.full_like(tgt[..., 0], m.grid_range)
            tgt_canvas = tgt / m.grid_range

        for k, b in enumerate(buckets):
            on = tgt_canvas[:, k].abs().amax(-1) <= 1.0
            keep = valid[:, k] & on
            acc[b]["n_all"] += int(valid[:, k].sum())
            acc[b]["n_on"] += int(keep.sum())
            if keep.sum() == 0:
                continue
            cell = ((tgt_canvas[keep, k] + 1) / 2 * (G - 1)).round().long()
            cell = cell.clamp(0, G - 1)
            scores = hdr_mass_at_truth(ld[keep, k], cell)
            if i % 2 == 0:
                acc[b]["cal"].append(scores.cpu())
            else:
                acc[b]["ev"].append(scores.cpu())
                # per-sample cell areas (cone canvases scale with R(t))
                # cone_ranges returns (B, H, 1) for target broadcasting —
                # flatten so per-sample areas stay 1-D
                cell_deg = 2.0 * R[keep, k].reshape(-1) / G
                cell_km2 = (cell_deg * KM_PER_DEG) ** 2 \
                    * math.cos(math.radians(LAT0))
                acc[b]["area"].append(cell_km2.cpu())
                acc[b].setdefault("ev_ld", []).append(ld[keep, k].cpu())

    out = {}
    for b in buckets:
        cal = torch.cat(acc[b]["cal"]) if acc[b]["cal"] else torch.zeros(0)
        ev = torch.cat(acc[b]["ev"]) if acc[b]["ev"] else torch.zeros(0)
        if cal.numel() < 50 or ev.numel() < 50:
            out[b] = {"n_cal": int(cal.numel()), "n_eval": int(ev.numel())}
            continue
        tau = conformal_threshold(cal, alpha)
        areas = []
        for ld_chunk, ck2 in zip(acc[b]["ev_ld"], acc[b]["area"]):
            cells = hdr_region_cells(ld_chunk, tau).float()
            areas.append(cells * ck2)
        area = torch.cat(areas)
        out[b] = {
            "n_cal": int(cal.numel()), "n_eval": int(ev.numel()),
            "ceiling": round(acc[b]["n_on"] / max(acc[b]["n_all"], 1), 4),
            "tau": round(tau, 4),
            "coverage_eval": round(calibrated_coverage(ev, tau), 4),
            "calib_area90_mean_km2": round(area.mean().item(), 1),
            "calib_area90_median_km2": round(area.median().item(), 1),
            "calib_area90_p90_km2": round(area.quantile(0.9).item(), 1),
        }
    return out


def main():
    maxb = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    out_path = "/home/paul/data/trackfm/conformal_v2.json"
    try:
        merged = json.load(open(out_path))
    except FileNotFoundError:
        merged = {}
    for cfg_name, run in RUNS:
        if only is not None and run not in only:
            continue
        ckpt = f"{CKPT}/{run}/best.pt"
        try:
            merged[run] = conformal_for(cfg_name, ckpt, maxb, bs)
            print(f"scored {run}")
        except FileNotFoundError:
            print(f"skip {run}: no checkpoint yet")
        except Exception as e:
            print(f"SKIP {run}: {e!r}")
    json.dump(merged, open(out_path, "w"), indent=2)

    print(f"\n=== CONFORMAL alpha=0.1, val | calibrated area@90 km^2 (mean) ===")
    print(f"{'run':<38} {'15m':>8} {'30m':>8} {'1h':>8} {'2h':>8} {'cov2h':>6}")
    for _, run in RUNS:
        if run not in merged:
            continue
        row = [merged[run].get(b, {}).get("calib_area90_mean_km2")
               for b in ("15m", "30m", "1h", "2h")]
        cov = merged[run].get("2h", {}).get("coverage_eval")
        print(f"{run:<38} " +
              " ".join(f"{v:>8.1f}" if v is not None else f"{'—':>8}" for v in row) +
              (f" {cov:>6.3f}" if cov is not None else f" {'—':>6}"))
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
