"""Frozen-encoder head-depth ladder (plan-revision item 4).

Freezes the medium-cone-mlp encoder (the strongest 18M cone checkpoint)
and trains FRESH Fourier-coefficient heads of depth 0/1/2/3 (hidden=384)
on frozen features, then scores each on the fixgrid protocol via the
pilot's validated eval path. Maps containment-vs-head-depth in ~1 GPU-h
per rung instead of a 6 h pretrain per point.

Screening bias (documented in DECISIONS): frozen features understate
deeper heads (no encoder co-adaptation), and the encoder was itself
trained against a depth-1 head. Flat ladder from depth 1 up => depth 1
suffices at this scale; a rising ladder is decided by ONE full-training
confirmation run, not by this screen.

Usage: python scripts/head_ladder.py [steps_per_rung] [batch_size]
Writes ~/data/trackfm/head_ladder.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trackfm.config import PretrainConfig, load_config  # noqa: E402
from trackfm.datasets.loaders import ShardedWindowDataset  # noqa: E402
from trackfm.models.factory import build_model  # noqa: E402
from trackfm.training.losses import (compute_soft_target_loss,  # noqa: E402
                                       cone_elapsed_seconds, cone_ranges)
from trackfm.training.pretrain import _lr_lambda  # noqa: E402
from ar_rollout_pilot import (BUCKETS, direct_ranks_for,  # noqa: E402
                              pick_truth, p90)

CKPT = "/home/paul/data/trackfm/checkpoints/scaling-medium-cone-mlp-50M/best.pt"
CFG = "configs/pretrain/scaling_medium_cone_mlp_50M.yaml"
TRAIN_DIR = "/home/paul/data/trackfm/materialized/v1/train"
VAL_DIR = "/home/paul/data/trackfm/materialized/v1/val"
DEPTHS = (0, 1, 2, 3)
HIDDEN = 384
SEED = 17
WARMUP = 100


def fresh_head(d_model: int, d_out: int, depth: int) -> nn.Module:
    """Depth-k coefficient head. depth=0 -> single linear (historical);
    depth=1 -> the CHAIN4 winner shape; deeper adds hidden->hidden."""
    if depth == 0:
        lin = nn.Linear(d_model, d_out)
        nn.init.normal_(lin.weight, std=0.01)
        return lin
    layers = [nn.Linear(d_model, HIDDEN), nn.GELU()]
    nn.init.normal_(layers[0].weight, std=0.01)
    for _ in range(depth - 1):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.GELU()]
    layers.append(nn.Linear(HIDDEN, d_out))
    return nn.Sequential(*layers)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    device = torch.device("cuda")
    cfg = load_config(CFG, PretrainConfig)
    m, norm = cfg.model, cfg.normalization
    torch.manual_seed(SEED)
    model = build_model(m, norm)
    model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                     weights_only=False)["model"])
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    # infer coefficient output dim from the existing head
    ref = model.fourier_head.coeff_predictor
    d_out = (ref[-1] if isinstance(ref, nn.Sequential) else ref).out_features

    results = {}
    for depth in DEPTHS:
        torch.manual_seed(SEED + depth)
        head = fresh_head(m.d_model, d_out, depth).to(device)
        model.fourier_head.coeff_predictor = head
        opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: _lr_lambda(s, WARMUP, steps, "cosine"))
        ds = ShardedWindowDataset(TRAIN_DIR, batch_size=bs, seed=SEED,
                                  shuffle_shards=True)
        it = iter(DataLoader(ds, batch_size=None, num_workers=2,
                             pin_memory=True))
        model.train()
        for step in range(steps):
            batch = next(it).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ld, tgt, hz, _ = model.forward_train(batch, causal=True)
                el = cone_elapsed_seconds(batch, hz, m.max_seq_len,
                                          norm.dt_scale, causal=True)
                tgt = tgt / cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)
                loss = compute_soft_target_loss(ld.float(), tgt.float(), 1.0,
                                                m.grid_size,
                                                0.003 / m.grid_range)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            if (step + 1) % 500 == 0:
                print(f"  depth {depth}: step {step + 1}/{steps} "
                      f"loss {loss.item():.4f}", flush=True)

        # eval: pilot-validated fixgrid protocol over shuffled val batches
        model.eval()
        agg = {b: [] for b in BUCKETS}
        vds = ShardedWindowDataset(VAL_DIR, batch_size=256, seed=SEED,
                                   shuffle_shards=True)
        vit = iter(DataLoader(vds, batch_size=None))
        with torch.no_grad():
            for _ in range(8):
                vb = next(vit).to(device)
                _, truth, _, _ = pick_truth(vb, m, norm)
                r, _ = direct_ranks_for(model, vb, truth, m, device,
                                        vb.shape[0])
                for b in BUCKETS:
                    agg[b].extend(r[b])
        results[depth] = {b: {"n": len(agg[b]), "p90": p90(agg[b])}
                          for b in BUCKETS}
        print(f"depth {depth}: " + " ".join(
            f"{b}={results[depth][b]['p90']}" for b in BUCKETS), flush=True)

    print(f"\n{'depth':<6}" + "".join(f"{b:>8}" for b in BUCKETS))
    for depth in DEPTHS:
        print(f"{depth:<6}" + "".join(
            f"{results[depth][b]['p90']!s:>8}" for b in BUCKETS))
    json.dump(results, open("/home/paul/data/trackfm/head_ladder.json", "w"),
              indent=2)
    print("written ~/data/trackfm/head_ladder.json")


if __name__ == "__main__":
    main()
