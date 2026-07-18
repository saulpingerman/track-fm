"""muP cross-width verification (GPU; run ONLY after the chain drains).

TIER 1 (bug catcher, ~45 GPU-min): widths {64, 128, 256} at the SAME
base LR, depth 4, 3000 steps, same seed + shard order. Gates:
  - all losses finite everywhere
  - wider-never-worse within 2% on the fast-val EMA at step 3000
  - grad-RMS per muP group and activation RMS (layer-2 out, head logits)
    within [0.5, 2.0] across the 4x width span
  - clipped-fraction of clip_grad_norm_ divergence < 20pp across widths
    (grad clipping binds differently by width -> would break transfer
    while every unit test passes; a >20pp divergence blocks trusting
    any transfer result until clipping is redesigned under the flag)

TIER 2 (transfer evidence, ~3 GPU-h): widths x base_lr in
{1e-4, 3e-4, 1e-3, 3e-3}: the argmin over LR must align across widths
(or log-lr parabola optima within sqrt(10)), wider-never-worse at the
common optimum, no width-unique NaN at the aggressive end.

Usage:
  python scripts/mup_crosswidth_smoke.py tier1
  python scripts/mup_crosswidth_smoke.py tier2
Writes ~/data/trackfm/mup_crosswidth_{tier}.json; the companion report
script prints gates and PASS/FAIL verdicts.

PROTOCOL NOTES (restated so they cannot be forgotten):
  - The production base sweep runs at width 128 and DEPTH 16 (flagship
    depth) — muP transfers over width, NOT depth.
  - F/G/grid_mode/sigma must be frozen across the width series; the
    validator does not enforce these — this script's config constructor
    holds them fixed by construction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trackfm.config import ModelConfig, NormalizationConfig  # noqa: E402
from trackfm.datasets.loaders import ShardedWindowDataset  # noqa: E402
from trackfm.models.encoder import CausalAISModel  # noqa: E402
from trackfm.training.losses import (compute_soft_target_loss,  # noqa: E402
                                       cone_elapsed_seconds, cone_ranges)
from trackfm.training.mup import build_optimizer  # noqa: E402
from trackfm.training.pretrain import _lr_lambda  # noqa: E402

TRAIN_DIR = "/home/paul/data/trackfm/materialized/v1/train"
STEPS, WARMUP = 3000, 100
BS = 512
WIDTHS = (64, 128, 256)
TIER2_LRS = (1e-4, 3e-4, 1e-3, 3e-3)
EMA = 0.98


def _cfg(d):
    return ModelConfig(d_model=d, nhead=d // 16, num_layers=4,
                       dim_feedforward=4 * d, dropout=0.1, max_seq_len=128,
                       grid_size=64, grid_range=0.3, num_freqs=12,
                       grid_mode="cone",
                       mup={"enabled": True, "d_base": 128, "d_head": 16})


class _T:
    weight_decay = 1e-5

    def __init__(self, lr):
        self.learning_rate = lr


def run_one(d: int, lr: float, seed: int = 17) -> dict:
    device = torch.device("cuda")
    norm = NormalizationConfig()
    m = _cfg(d)
    torch.manual_seed(seed)
    model = CausalAISModel(m, norm, max_horizon=800,
                           num_horizon_samples=4).to(device)
    opt = build_optimizer(model, _T(lr), m)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, WARMUP, STEPS, "cosine"))
    ds = ShardedWindowDataset(TRAIN_DIR, batch_size=BS, seed=seed,
                              shuffle_shards=True)
    loader = DataLoader(ds, batch_size=None, num_workers=4, pin_memory=True)

    ema_loss, clipped, probes = None, 0, {}
    model.train()
    it = iter(loader)
    for step in range(STEPS):
        batch = next(it).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ld, tgt, hz, _ = model.forward_train(batch, causal=True)
            el = cone_elapsed_seconds(batch, hz, m.max_seq_len,
                                      norm.dt_scale, causal=True)
            tgt = tgt / cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)
            loss = compute_soft_target_loss(ld.float(), tgt.float(), 1.0,
                                            m.grid_size, 0.003 / m.grid_range)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped += int(total_norm.item() > 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        li = loss.item()
        ema_loss = li if ema_loss is None else EMA * ema_loss + (1 - EMA) * li
        if not torch.isfinite(loss):
            return {"d": d, "lr": lr, "nan_at": step, "ema": None}
        if step == STEPS - 1:
            with torch.no_grad():
                x = batch[:, :m.max_seq_len, :]
                enc = model.encode(x)
                probes["encoder_rms"] = enc.float().pow(2).mean().sqrt().item()
                probes["logit_range"] = (ld.max() - ld.min()).item()
                for gi, g in enumerate(opt.param_groups):
                    rms = torch.stack([
                        p.grad.float().pow(2).mean().sqrt()
                        for p in g["params"] if p.grad is not None]).mean() \
                        if any(p.grad is not None for p in g["params"]) else 0
                    probes[f"grad_rms_g{gi}"] = float(rms) if rms is not None else 0.0

    return {"d": d, "lr": lr, "ema": ema_loss, "nan_at": None,
            "clipped_frac": clipped / STEPS, "probes": probes}


def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else "tier1"
    out = {"tier": tier, "runs": []}
    if tier == "tier1":
        for d in WIDTHS:
            r = run_one(d, 3e-4)
            out["runs"].append(r)
            print(f"d={d}: ema={r['ema']:.4f} clipped={r.get('clipped_frac', 0):.2%} "
                  f"probes={r.get('probes')}", flush=True)
    else:
        for d in WIDTHS:
            for lr in TIER2_LRS:
                r = run_one(d, lr)
                out["runs"].append(r)
                print(f"d={d} lr={lr:.0e}: ema={r['ema']} nan_at={r['nan_at']}",
                      flush=True)
    path = f"/home/paul/data/trackfm/mup_crosswidth_{tier}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"written {path}")


if __name__ == "__main__":
    main()
