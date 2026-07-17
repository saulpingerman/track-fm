"""Dead-reckoning NULL GATE (architecture-review Day-1).

The gate every learned model must beat, per horizon, before its result
is credited: constant-velocity extrapolation from the last posit with a
per-bucket-calibrated isotropic Gaussian density on the SAME cone canvas
and scored with the SAME v2 metrics + split-conformal calibration as the
models. Constant-velocity nulls upset neural predictors in adjacent
domains (pedestrians); vessels are more inertial — if a model arm fails
this gate at some horizon, that horizon's compute is being wasted.

CPU-only. Splits val batches alternately into CAL (sigma fit + conformal
threshold) and EVAL (all reported numbers).

Reports per bucket: fixgrid p90 rank (0.6 km2 cells, +-0.3 deg), ceiling,
km2@90 (ceiling-aware), conformal-calibrated area@90 in km2.
"""
from __future__ import annotations

import json
import math
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/paul/projects/trackfm-v2/src")
from trackfm.config import NormalizationConfig
from trackfm.datasets.loaders import ShardedWindowDataset
from trackfm.eval.conformal import (conformal_threshold, hdr_mass_at_truth,
                                     hdr_region_cells)
from trackfm.eval.horizons import TIME_BUCKETS, time_bucket_indices
from trackfm.eval.search import gaussian_grid_log_density
from trackfm.eval.xgeometry import (FIXED_NATIVE_CELL_DEG, KM_PER_DEG, LAT0,
                                     ranks_on_fine_grid)
from trackfm.training.losses import (cone_elapsed_seconds, cone_ranges,
                                       dead_reckoning_displacement)

VAL_DIR = "/home/paul/data/trackfm/materialized/v1/val"
G = 64
SEQ = 128
CONE = dict(r0=0.02, v=0.000171, p=1.0)
N_BATCHES = 60
BS = 512
SIGMA_GRID = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]  # x bucket p50 disp


class _Cfg:                                     # minimal shim for helpers
    pass


def main():
    norm = NormalizationConfig()
    ds = ShardedWindowDataset(VAL_DIR, batch_size=BS, shuffle_shards=False)
    loader = DataLoader(ds, batch_size=None, num_workers=2)
    buckets = list(TIME_BUCKETS)

    # pass 1: gather per-bucket DR displacements, targets, elapsed
    data = {b: {"dr": [], "tgt": [], "el": [], "cal": []} for b in buckets}
    for i, batch in enumerate(loader):
        if i >= N_BATCHES:
            break
        step_idx, valid = time_bucket_indices(batch, norm, SEQ)
        el = cone_elapsed_seconds(batch, step_idx, SEQ, norm.dt_scale,
                                  causal=False)                            # (B,H)
        lat = batch[:, :, 0] * norm.lat_scale + norm.lat_center
        lon = batch[:, :, 1] * norm.lon_scale + norm.lon_center
        # constant-velocity DR from the last input step, per-sample elapsed
        # (dead_reckoning_displacement only takes shared (H,) horizons)
        dt_last = (batch[:, SEQ - 1, 5] * norm.dt_scale).clamp(min=1.0)
        vlat = (lat[:, SEQ - 1] - lat[:, SEQ - 2]) / dt_last               # deg/s
        vlon = (lon[:, SEQ - 1] - lon[:, SEQ - 2]) / dt_last
        dr = torch.stack([vlat[:, None] * el, vlon[:, None] * el], -1)     # (B,H,2)
        src = torch.stack([lat[:, SEQ - 1], lon[:, SEQ - 1]], -1)          # (B,2)
        gi = (SEQ + step_idx - 1)
        tgt = torch.stack([torch.gather(lat, 1, gi), torch.gather(lon, 1, gi)], -1) - src[:, None, :]
        for k, b in enumerate(buckets):
            m = valid[:, k]
            if m.any():
                data[b]["dr"].append(dr[m, k]); data[b]["tgt"].append(tgt[m, k])
                data[b]["el"].append(el[m, k])
                data[b]["cal"].append(torch.full((int(m.sum()),), i % 2 == 0))

    out = {}
    print(f"{'bucket':>6} {'n_eval':>7} {'sigma*':>7} {'ceil':>6} {'fixgrid_p90':>12} "
          f"{'km2@90':>8} {'conf_area@90':>13}")
    for b in buckets:
        dr = torch.cat(data[b]["dr"]); tgt = torch.cat(data[b]["tgt"])
        el = torch.cat(data[b]["el"]); cal = torch.cat(data[b]["cal"]).bool()
        R = CONE["r0"] + CONE["v"] * el.clamp(min=0) ** CONE["p"]          # (N,)
        # canvas coords
        drc, tgtc = dr / R[:, None], tgt / R[:, None]

        # fit sigma (canvas units) on CAL half by median CE proxy: pick the
        # sigma minimizing mean HDR-mass-at-truth (sharp but covering)
        p50 = drc[cal].sub(tgtc[cal]).norm(dim=-1).median().item()
        best = None
        for c in SIGMA_GRID:
            s = max(c * max(p50, 1e-3), 1e-3)
            ld = gaussian_grid_log_density(drc[cal][None], torch.full((1, cal.sum(), 2), s), 1.0, G)[0]
            node = ((tgtc[cal] + 1) / (2 / (G - 1))).round().long().clamp(0, G - 1)
            scores = hdr_mass_at_truth(ld, node)
            m = scores.mean().item()
            if best is None or m < best[1]:
                best = (s, m)
        sig = best[0]

        ev = ~cal
        ld_ev = gaussian_grid_log_density(drc[ev][None], torch.full((1, int(ev.sum()), 2), sig), 1.0, G)[0]
        # v2 metrics: fixgrid p90 + ceiling + km2@90
        fr, _, _ = ranks_on_fine_grid(ld_ev, tgtc[ev], R_deg=float(R[ev].mean()),
                                       cell_deg=FIXED_NATIVE_CELL_DEG, restrict_deg=0.3)
        fr_ok = fr[fr > 0].float()
        n_all = len(fr)
        ceiling = len(fr_ok) / max(n_all, 1)
        p90 = float(fr_ok.quantile(min(0.9 * n_all / max(len(fr_ok), 1), 1.0))) \
            if len(fr_ok) >= 0.9 * n_all else None
        # km2@90 via 1x1km fine grid, ceiling-aware
        fr1, _, cell_km2 = ranks_on_fine_grid(ld_ev, tgtc[ev], R_deg=float(R[ev].mean()),
                                               cell_deg=None, target_cell_km=1.0)
        f1_ok = fr1[fr1 > 0].float()
        km90 = float(f1_ok.quantile(0.9 * n_all / len(f1_ok)) * cell_km2) \
            if len(f1_ok) >= 0.9 * n_all else None
        # conformal-calibrated area@90 on the native cone canvas
        node_cal = ((tgtc[cal] + 1) / (2 / (G - 1))).round().long().clamp(0, G - 1)
        ld_cal = gaussian_grid_log_density(drc[cal][None], torch.full((1, int(cal.sum()), 2), sig), 1.0, G)[0]
        tau = conformal_threshold(hdr_mass_at_truth(ld_cal, node_cal), alpha=0.1)
        cells = hdr_region_cells(ld_ev, tau).float().mean().item()
        R_mean = float(R[ev].mean())
        native_km2 = (2 * R_mean / (G - 1) * KM_PER_DEG) ** 2 * math.cos(math.radians(LAT0))
        conf_area = cells * native_km2
        out[b] = dict(n_eval=int(ev.sum()), sigma_canvas=round(sig, 4),
                      ceiling=round(ceiling, 4),
                      fixgrid_p90=round(p90, 1) if p90 else None,
                      km2_at_90=round(km90, 1) if km90 else None,
                      conformal_area90_km2=round(conf_area, 1), tau=round(tau, 4))
        print(f"{b:>6} {out[b]['n_eval']:>7} {sig:>7.3f} {ceiling:>6.3f} "
              f"{str(out[b]['fixgrid_p90']):>12} {str(out[b]['km2_at_90']):>8} "
              f"{conf_area:>13.1f}")

    json.dump(out, open("/home/paul/data/trackfm/dr_null_gate.json", "w"), indent=2)
    print("\nwritten /home/paul/data/trackfm/dr_null_gate.json")


if __name__ == "__main__":
    main()
