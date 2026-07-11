#!/usr/bin/env python
"""Find max trainable batch size + throughput per model scale on this GPU.

Binary-searches the largest batch that survives a full train step (forward
+ backward + optimizer) in bf16, then measures sustained samples/s and MFU
at ~80% of that batch (headroom for fragmentation over long runs).

Usage: uv run python scripts/max_batch_sweep.py [--scales small,medium,large,xlarge]
"""
import argparse
import time

import torch

from trackfm.config import NormalizationConfig
from trackfm.models.factory import SCALES, build_model, model_config_for_scale
from trackfm.training.flops import train_flops_per_sample
from trackfm.training.losses import compute_soft_target_loss

MAX_HORIZON = 800
SEQ = 128
PEAK_TFLOPS = 304.0


def try_step(model, opt, batch_size: int, m_cfg) -> bool:
    try:
        x = torch.randn(batch_size, SEQ + MAX_HORIZON, 6, device="cuda") * 0.1
        with torch.autocast("cuda", torch.bfloat16):
            ld, tgt, _, _ = model.forward_train(x, causal=True)
            loss = compute_soft_target_loss(ld.float(), tgt.float(),
                                            m_cfg.grid_range, m_cfg.grid_size, 0.003)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False


def measure(model, opt, batch_size: int, m_cfg, iters: int = 8) -> float:
    x = torch.randn(batch_size, SEQ + MAX_HORIZON, 6, device="cuda") * 0.1
    for _ in range(2):  # warmup
        with torch.autocast("cuda", torch.bfloat16):
            ld, tgt, _, _ = model.forward_train(x, causal=True)
            loss = compute_soft_target_loss(ld.float(), tgt.float(),
                                            m_cfg.grid_range, m_cfg.grid_size, 0.003)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", torch.bfloat16):
            ld, tgt, _, _ = model.forward_train(x, causal=True)
            loss = compute_soft_target_loss(ld.float(), tgt.float(),
                                            m_cfg.grid_range, m_cfg.grid_size, 0.003)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return batch_size * iters / (time.perf_counter() - t0)


def sweep(scale: str):
    m_cfg = model_config_for_scale(scale)
    model = build_model(m_cfg, NormalizationConfig(), MAX_HORIZON, 4).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    lo, hi = 8, 8
    while try_step(model, opt, hi, m_cfg):
        lo, hi = hi, hi * 2
        if hi > 16384:
            break
    while hi - lo > max(8, lo // 16):
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if try_step(model, opt, mid, m_cfg) else (lo, mid)

    use = int(lo * 0.8)
    sps = measure(model, opt, use, m_cfg)
    fps = train_flops_per_sample(m_cfg, 4)
    tflops = fps * sps / 1e12
    print(f"{scale:>7}: max batch {lo:>5} | recommend {use:>5} | "
          f"{sps:7.0f} samples/s | {tflops:5.1f} TFLOPS | MFU {tflops/PEAK_TFLOPS:5.1%}")

    del model, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="small,medium,large,xlarge")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    print(f"device: {torch.cuda.get_device_name(0)}, horizon={MAX_HORIZON}")
    for s in args.scales.split(","):
        assert s in SCALES, s
        sweep(s)
