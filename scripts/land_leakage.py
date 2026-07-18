"""Land-leakage diagnostic: does geo conditioning move density off land?

Compares a conditioned checkpoint against an unconditioned baseline on
val, per time bucket:

  * mass_on_land   — expected fraction of predicted density on land
                     cells of each pair's canvas (the number geography
                     conditioning exists to reduce);
  * truth_on_land% — fraction of TRUTH positions whose nearest canvas
                     cell the raster calls land (coastal
                     misregistration rate of the raster itself);
  * nll_sea/land   — mean NLL at truth split by that flag. A conditioned
                     model that carves land too hard shows a large
                     nll_land gap vs its own nll_sea: it is zeroing
                     cells that real vessels occupy (raster-resolution
                     coastline error) — the failure mode behind the v1
                     val-loss spikes.

Usage:
  python scripts/land_leakage.py BASE_CFG BASE_CKPT CTX_CFG CTX_CKPT \
      [max_batches] [batch_size]
Writes ~/data/trackfm/land_leakage.json and prints the table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/paul/projects/trackfm-v2/src")
from trackfm.config import PretrainConfig, load_config
from trackfm.context.crops import StaticContext
from trackfm.datasets.loaders import ShardedWindowDataset
from trackfm.eval.horizons import TIME_BUCKETS, time_bucket_indices
from trackfm.models.factory import build_model
from trackfm.training.losses import cone_elapsed_seconds, cone_ranges


@torch.no_grad()
def diagnose(cfg_path: str, ckpt: str, max_batches: int,
             batch_size: int | None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(f"configs/pretrain/{cfg_path}.yaml", PretrainConfig)
    m, t = cfg.model, cfg.train
    assert m.grid_mode == "cone", "diagnostic assumes cone canvases"
    state = torch.load(Path(ckpt).expanduser(), map_location="cpu",
                       weights_only=False)
    model = build_model(m, cfg.normalization, t.max_horizon,
                        len(TIME_BUCKETS)).to(device).eval()
    model.load_state_dict(state["model"])

    land_ctx = StaticContext(m.context_static_dir, with_traffic=False).to(device)

    bs = batch_size or t.batch_size
    ds = ShardedWindowDataset(cfg.data_dir / "val", batch_size=bs,
                              shuffle_shards=False)
    loader = DataLoader(ds, batch_size=None, num_workers=2,
                        pin_memory=device.type == "cuda")

    buckets = list(TIME_BUCKETS)
    acc = {b: {"mass": [], "nll_sea": [], "nll_land": [], "underway": [],
               "on_land": []} for b in buckets}
    G = m.grid_size
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
        el = cone_elapsed_seconds(batch, step_idx, m.max_seq_len,
                                  cfg.normalization.dt_scale, causal=False)
        R = cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)      # (B, H)
        tgt_canvas = tgt / R

        lat0 = batch[:, m.max_seq_len - 1, 0] * cfg.normalization.lat_scale \
            + cfg.normalization.lat_center
        lon0 = batch[:, m.max_seq_len - 1, 1] * cfg.normalization.lon_scale \
            + cfg.normalization.lon_center

        for k, b in enumerate(buckets):
            keep = valid[:, k] & (tgt_canvas[:, k].abs().amax(-1) <= 1.0)
            if keep.sum() == 0:
                continue
            crops = land_ctx.crops(lat0[keep], lon0[keep],
                                   R[keep, k], G)               # (n,C,G,G)
            land = crops[:, 0] > 0.5                            # (n,G,G)
            dens = ld[keep, k].exp()
            mass = (dens * land).sum(dim=(-2, -1))
            # nearest-cell NLL at truth, split by the raster's land flag
            cell = ((tgt_canvas[keep, k] + 1) / 2 * (G - 1)).round().long()
            cell = cell.clamp(0, G - 1)
            # density convention: dim -2 = lat, dim -1 = lon
            nll = -ld[keep, k][torch.arange(cell.shape[0]),
                               cell[:, 0], cell[:, 1]]
            on_land = land[torch.arange(cell.shape[0]), cell[:, 0], cell[:, 1]]
            # moored vessels in harbors sit on "land" pixels at raster
            # resolution (smoke run: 24-37% of truths!) — the underway
            # split separates real leakage from correct harbor mass
            sog_kn = batch[keep, m.max_seq_len - 1, 2] \
                * cfg.normalization.sog_scale
            acc[b]["mass"].append(mass.cpu())
            acc[b]["underway"].append((sog_kn >= 2.0).cpu())
            acc[b]["on_land"].append(on_land.cpu())
            acc[b]["nll_sea"].append(nll[~on_land].cpu())
            acc[b]["nll_land"].append(nll[on_land].cpu())

    out = {}
    for b in buckets:
        mass = torch.cat(acc[b]["mass"]) if acc[b]["mass"] else torch.zeros(0)
        ns = torch.cat(acc[b]["nll_sea"]) if acc[b]["nll_sea"] else torch.zeros(0)
        nl = torch.cat(acc[b]["nll_land"]) if acc[b]["nll_land"] else torch.zeros(0)
        uw = torch.cat(acc[b]["underway"]) if acc[b]["underway"] else torch.zeros(0, dtype=torch.bool)
        ol = torch.cat(acc[b]["on_land"]) if acc[b]["on_land"] else torch.zeros(0, dtype=torch.bool)
        n = mass.numel()
        out[b] = {
            "n": n,
            "underway_frac": round(uw.float().mean().item(), 4) if n else None,
            "mass_on_land_mean": round(mass.mean().item(), 5) if n else None,
            "mass_on_land_underway": round(mass[uw].mean().item(), 5) if uw.any() else None,
            "mass_on_land_moored": round(mass[~uw].mean().item(), 5) if (~uw).any() else None,
            "truth_on_land_frac": round(nl.numel() / max(ns.numel() + nl.numel(), 1), 5),
            "truth_on_land_underway": round(ol[uw].float().mean().item(), 5) if uw.any() else None,
            "nll_sea_mean": round(ns.mean().item(), 4) if ns.numel() else None,
            "nll_land_mean": round(nl.mean().item(), 4) if nl.numel() else None,
        }
    return out


def main():
    base_cfg, base_ckpt, ctx_cfg, ctx_ckpt = sys.argv[1:5]
    maxb = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    bs = int(sys.argv[6]) if len(sys.argv) > 6 else None
    res = {
        "base": {"cfg": base_cfg, "ckpt": base_ckpt,
                 **{"buckets": diagnose(base_cfg, base_ckpt, maxb, bs)}},
        "ctx": {"cfg": ctx_cfg, "ckpt": ctx_ckpt,
                **{"buckets": diagnose(ctx_cfg, ctx_ckpt, maxb, bs)}},
    }
    path = "/home/paul/data/trackfm/land_leakage.json"
    json.dump(res, open(path, "w"), indent=2)
    print(f"{'bucket':<6} {'role':<5} {'landmass(uw)':>13} {'landmass(moor)':>15} "
          f"{'truth_land%(uw)':>16} {'nll_sea':>9} {'nll_land':>9}")
    for b in res["base"]["buckets"]:
        for role in ("base", "ctx"):
            r = res[role]["buckets"][b]
            tlu = r["truth_on_land_underway"]
            print(f"{b:<6} {role:<5} {str(r['mass_on_land_underway']):>13} "
                  f"{str(r['mass_on_land_moored']):>15} "
                  f"{tlu*100 if tlu is not None else -1:>15.2f}% "
                  f"{str(r['nll_sea_mean']):>9} {str(r['nll_land_mean']):>9}")
    print(f"written {path}")


if __name__ == "__main__":
    main()
