#!/usr/bin/env python
"""Measure the practical bf16 tensor-core peak of this GPU.

Sweeps square matmul sizes with proper warmup and reports the best
sustained TFLOPS — the honest denominator for MFU on this specific card
(spec-sheet peaks assume unlimited power; the Max-Q edition is capped).

Usage: uv run python scripts/gpu_peak_bench.py
"""
import time

import torch


def bench(n: int, iters: int = 30, dtype=torch.bfloat16) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(10):  # warmup
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return 2 * n**3 * iters / dt / 1e12


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print(f"device: {torch.cuda.get_device_name(0)}")
    best = 0.0
    for n in (2048, 4096, 8192, 12288, 16384):
        t = bench(n)
        best = max(best, t)
        print(f"  {n:>6} x {n:<6} bf16: {t:7.1f} TFLOPS")
    print(f"\npractical bf16 peak: {best:.0f} TFLOPS")
    print("use this as train.peak_tflops for honest MFU numbers")
