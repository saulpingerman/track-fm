"""Time-indexed horizons: evaluate at wall-clock buckets, not step counts.

AIS cadence varies per vessel, so step-indexed horizons conflate physically
different questions (step 800 spans 1.3h-4.5h across the golden69 val set).
Operational metrics are asked in minutes: "where is it in 30 minutes."
"""
from __future__ import annotations

import torch

from trackfm.config import NormalizationConfig

# Operational containment horizons (seconds)
TIME_BUCKETS = {"15m": 900.0, "30m": 1800.0, "1h": 3600.0, "2h": 7200.0}


def time_bucket_indices(
    features: torch.Tensor,          # (B, L, 6) normalized window
    norm: NormalizationConfig,
    seq_len: int = 128,
    taus: dict[str, float] = TIME_BUCKETS,
    tolerance: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample future-step index closest to each time bucket.

    Returns (step_idx (B, H) 1-based, valid (B, H) bool) — valid is False
    where no future position lands within ±tolerance of the bucket (fast
    reporters can exhaust the window before 2h; those samples are excluded
    from that bucket, and availability is reported alongside the metric).
    """
    dt = features[:, :, 5] * norm.dt_scale
    cum = torch.cumsum(dt, dim=1)
    elapsed = cum[:, seq_len:] - cum[:, seq_len - 1:seq_len]     # (B, F)

    idxs, valids = [], []
    for tau in taus.values():
        diff = (elapsed - tau).abs()
        best = diff.argmin(dim=1)                                # 0-based future idx
        ok = torch.gather(diff, 1, best[:, None]).squeeze(1) <= tolerance * tau
        idxs.append(best + 1)                                    # 1-based step
        valids.append(ok)
    return torch.stack(idxs, dim=1), torch.stack(valids, dim=1)
