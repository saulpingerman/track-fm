"""Generate SP bit-for-bit baseline fixtures — RUN FROM THE PRE-muP TREE.

Acceptance gate for the muP retrofit: these fixtures pin the Standard
Parameterization training trajectory (init + 5 optimizer steps) of tiny
models on CPU/fp32 with fully seeded determinism. After the retrofit,
tests/training/test_sp_equivalence.py replays each fixture with muP
disabled and requires BITWISE-identical state hashes. Any RNG-consumption
reorder, added parameter, changed init, or optimizer-grouping change
fails with per-tensor localization.

The replayed loop mirrors run_pretraining's inner loop exactly:
  forward_train(batch, causal=True)   <- RANDOM horizons from seeded RNG
  cone/fixed loss branch (eager compute_soft_target_loss)
  loss / grad_accum(=1); backward; clip_grad_norm_(1.0); opt.step();
  zero_grad(set_to_none=True); sched.step()
Dropout stays 0.1 and model.train() so RNG-order changes are detected.

Variants: {fourier, direct} x {fixed, cone} + fourier/fixed/head_mlp=32.
Fixtures: tests/reference/sp_baseline_<variant>.json with per-tensor and
whole-state sha256 at step 0 and step 5, plus 5 exact loss reprs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trackfm.config import ModelConfig, NormalizationConfig  # noqa: E402
from trackfm.models.encoder import CausalAISModel  # noqa: E402
from trackfm.training.losses import (compute_soft_target_loss,  # noqa: E402
                                       cone_elapsed_seconds, cone_ranges)
from trackfm.training.pretrain import _lr_lambda  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "reference"
SEQ, MAX_H, N_H = 32, 64, 4
STEPS, WARMUP, MAX_STEPS = 5, 3, 20
LR, WD = 3e-4, 1e-5
BATCH = 8

VARIANTS = {
    "fourier_fixed": dict(head_type="fourier", grid_mode="fixed", head_mlp_hidden=0),
    "fourier_cone": dict(head_type="fourier", grid_mode="cone", head_mlp_hidden=0),
    "direct_fixed": dict(head_type="direct", grid_mode="fixed", head_mlp_hidden=0),
    "direct_cone": dict(head_type="direct", grid_mode="cone", head_mlp_hidden=0),
    "fourier_fixed_mlp32": dict(head_type="fourier", grid_mode="fixed",
                                head_mlp_hidden=32),
}


def tiny_cfg(**over) -> ModelConfig:
    return ModelConfig(d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                       dropout=0.1, max_seq_len=SEQ, grid_size=16,
                       grid_range=0.3, num_freqs=3, **over)


def synth_batch(norm: NormalizationConfig) -> torch.Tensor:
    """Deterministic synthetic windows: smooth tracks + positive dt."""
    g = torch.Generator().manual_seed(123)
    B, T = BATCH, SEQ + MAX_H
    lat0 = 56.0 + 0.5 * torch.rand(B, 1, generator=g)
    lon0 = 11.0 + 0.5 * torch.rand(B, 1, generator=g)
    step_deg = 0.001 * torch.randn(B, 2, generator=g)
    t_idx = torch.arange(T).float()[None, :]
    lat = lat0 + step_deg[:, 0:1] * t_idx
    lon = lon0 + step_deg[:, 1:2] * t_idx
    sog = 5.0 + 3.0 * torch.rand(B, 1, generator=g).expand(B, T)
    cog = (360.0 * torch.rand(B, 1, generator=g)).expand(B, T)
    dt = 60.0 + 30.0 * torch.rand(B, T, generator=g)
    feats = torch.stack([
        (lat - norm.lat_center) / norm.lat_scale,
        (lon - norm.lon_center) / norm.lon_scale,
        sog / norm.sog_scale,
        torch.sin(torch.deg2rad(cog)),
        torch.cos(torch.deg2rad(cog)),
        dt / norm.dt_scale,
    ], dim=-1)
    return feats.float()


def state_hashes(model: torch.nn.Module) -> dict:
    per_tensor = {}
    whole = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        b = v.detach().cpu().contiguous().numpy().tobytes()
        per_tensor[k] = hashlib.sha256(b).hexdigest()
        whole.update(k.encode())
        whole.update(b)
    return {"per_tensor": per_tensor, "whole": whole.hexdigest()}


def run_variant(name: str, over: dict) -> dict:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    norm = NormalizationConfig()
    m = tiny_cfg(**over)

    torch.manual_seed(0)
    model = CausalAISModel(m, norm, max_horizon=MAX_H, num_horizon_samples=N_H)
    model.train()

    init = state_hashes(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, WARMUP, MAX_STEPS, "cosine"))
    batch = synth_batch(norm)

    losses = []
    for step in range(STEPS):
        ld, tgt, hz, _ = model.forward_train(batch, causal=True)
        if m.grid_mode == "cone":
            el = cone_elapsed_seconds(batch, hz, m.max_seq_len,
                                      norm.dt_scale, causal=True)
            tgt = tgt / cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)
            loss = compute_soft_target_loss(ld.float(), tgt.float(), 1.0,
                                            m.grid_size, 0.003 / m.grid_range)
        else:
            loss = compute_soft_target_loss(ld.float(), tgt.float(),
                                            m.grid_range, m.grid_size, 0.003)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(repr(loss.item()))

    final = state_hashes(model)
    return {
        "variant": name, "config_overrides": over,
        "torch_version": torch.__version__,
        "seed": 0, "batch_seed": 123, "steps": STEPS,
        "lr": LR, "wd": WD, "warmup": WARMUP, "max_steps": MAX_STEPS,
        "losses": losses,
        "step0": init, "step5": final,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, over in VARIANTS.items():
        fx = run_variant(name, over)
        path = OUT_DIR / f"sp_baseline_{name}.json"
        path.write_text(json.dumps(fx, indent=1))
        print(f"{name}: step5 whole={fx['step5']['whole'][:16]}… "
              f"loss[0]={fx['losses'][0][:10]} loss[-1]={fx['losses'][-1][:10]}")
    print(f"\nfixtures in {OUT_DIR}")


if __name__ == "__main__":
    main()
