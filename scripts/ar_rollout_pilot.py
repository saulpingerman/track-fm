"""AR rollout pilot: sample-and-roll ensembles vs the direct density head.

Method 1 of docs/research/2026-07-autoregressive-generation.md. NO
training — rolls existing checkpoints forward on their own samples.

Verification-hardened (wf_fefb2c73 findings):
- Per-track TIME ALIGNMENT: the ensemble is evaluated at the rollout
  step closest to the truth posit's ACTUAL elapsed time (not the
  nominal bucket time) — the direct arm conditions on that same elapsed
  time, so both arms answer the same question. Rollout extends to
  1.2x the last bucket so tolerance-edge truths stay reachable.
- Truth tolerance matches the harness: |el - tau| <= 0.15*tau.
- Smoothing sensitivity: rollout ranks reported at sigma in {0,1,2}
  cells, plus a SYMMETRIC control (same kernel on the direct arm's
  exp-density) — a bandwidth-dominated comparison is flagged, not
  hidden.
- Sanity gates: step-1 sample log-prob vs direct-density entropy
  (a length-1 rollout IS the direct prediction), canvas-edge pileup
  fraction, over-speed fraction, in-restrict fraction per bucket.
- Direct-only harness-validation mode: `... direct <n_batches>` scores
  ONLY the direct arm over many shuffled batches — its p90s must land
  near rescore_v2's numbers (small-cone 2h ~ 120) before ensemble
  results are trusted.

Usage:
  python scripts/ar_rollout_pilot.py [n_tracks] [K] [chunk]
  python scripts/ar_rollout_pilot.py direct [n_batches]
Writes ~/data/trackfm/ar_rollout_pilot.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trackfm.config import PretrainConfig, load_config  # noqa: E402
from trackfm.datasets.loaders import ShardedWindowDataset  # noqa: E402
from trackfm.models.factory import build_model  # noqa: E402

CKPT = "/home/paul/data/trackfm/checkpoints/scaling-small-cone-50M/best.pt"
CFG = "configs/pretrain/scaling_small_cone_50M.yaml"
VAL_DIR = "/home/paul/data/trackfm/materialized/v1/val"
BUCKETS = {"15m": 900.0, "30m": 1800.0, "1h": 3600.0, "2h": 7200.0}
CADENCE_S = 60.0                 # overridable via argv[4]; data median dt
                                 # is ~10 s — 60 s windows are themselves
                                 # off-manifold (cadence sensitivity is
                                 # part of the pilot's question)
TOL = 0.15                       # harness tolerance (horizons.py)
RESTRICT = 0.3
NFX = 64
SEED = 17
SIGMAS = (0, 1, 2)               # smoothing sensitivity (cells)


def denorm_latlon(feat, norm):
    return (feat[..., 0] * norm.lat_scale + norm.lat_center,
            feat[..., 1] * norm.lon_scale + norm.lon_center)


@torch.no_grad()
def next_step_density(model, window, norm, device):
    pad = torch.zeros(window.shape[0], 1, 6, device=device)
    pad[..., 5] = CADENCE_S / norm.dt_scale
    feats = torch.cat([window, pad], dim=1)
    ld, _, _, _ = model.forward_train(
        feats, horizon_indices=torch.tensor([1], device=device), causal=False)
    return ld[:, 0]


@torch.no_grad()
def sample_step(model, window, m, norm, device, gen, chunk, stats):
    G = m.grid_size
    R = m.cone_r0 + m.cone_v * CADENCE_S ** m.cone_p
    # FINE-GRID step sampling: at short elapsed times the cone floor
    # (r0=0.02°) makes a native 64-node cell BIGGER than the true
    # per-step displacement — lattice sampling noise then exceeds the
    # motion signal (observed: 5.6%% over-speed at 10 s cadence) and the
    # rollout diffuses regardless of model quality. The Fourier density
    # is continuous and band-limited (F=12 over 64 nodes = 2.7x
    # oversampled), so bicubic upsampling to GF is faithful.
    GF = 256
    new_rows = []
    for s in range(0, window.shape[0], chunk):
        w = window[s:s + chunk]
        ld = next_step_density(model, w, norm, device)
        fine_ld = F.interpolate(ld.unsqueeze(1).float(), size=(GF, GF),
                                mode="bicubic", align_corners=True).squeeze(1)
        probs = fine_ld.reshape(w.shape[0], -1).softmax(dim=-1)
        idx = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        if stats.get("step") == 1:                 # step-1 sanity gate data
            lp = probs.gather(1, idx.unsqueeze(1)).squeeze(1).log() \
                + math.log(GF * GF / (m.grid_size * m.grid_size))
            stats["s1_sample_logp"].append(lp)
            p_c = ld.reshape(w.shape[0], -1).float().softmax(dim=-1)
            stats["s1_entropy"].append(-(p_c * p_c.clamp_min(1e-12).log())
                                       .sum(-1))
        nodes = torch.linspace(-1.0, 1.0, GF, device=device)
        half = 1.0 / (GF - 1)
        jy = (torch.rand(len(idx), generator=gen, device=device) * 2 - 1) * half
        jx = (torch.rand(len(idx), generator=gen, device=device) * 2 - 1) * half
        oy, ox = nodes[idx // GF] + jy, nodes[idx % GF] + jx
        stats["edge"] += int(((oy.abs() > 0.98) | (ox.abs() > 0.98)).sum())
        lat_prev, lon_prev = denorm_latlon(w[:, -1], norm)
        dlat, dlon = oy * R, ox * R
        lat_new, lon_new = lat_prev + dlat, lon_prev + dlon
        cosl = torch.cos(torch.deg2rad(lat_prev))
        knots = 60.0 * torch.sqrt(dlat ** 2 + (dlon * cosl) ** 2) \
            * (3600.0 / CADENCE_S)
        stats["over_speed"] += int((knots > 37.0).sum())
        heading = torch.atan2(dlon * cosl, dlat)
        new_rows.append(torch.stack([
            (lat_new - norm.lat_center) / norm.lat_scale,
            (lon_new - norm.lon_center) / norm.lon_scale,
            knots / norm.sog_scale,
            torch.sin(heading), torch.cos(heading),
            torch.full_like(lat_new, CADENCE_S / norm.dt_scale),
        ], dim=-1))
    rows = torch.cat(new_rows)
    return torch.cat([window[:, 1:], rows.unsqueeze(1)], dim=1)


def fixgrid_index(dlat, dlon):
    inside = (dlat.abs() < RESTRICT) & (dlon.abs() < RESTRICT)
    yi = ((dlat + RESTRICT) / (2 * RESTRICT) * NFX).floor().long().clamp(0, NFX - 1)
    xi = ((dlon + RESTRICT) / (2 * RESTRICT) * NFX).floor().long().clamp(0, NFX - 1)
    return torch.where(inside, yi * NFX + xi, torch.full_like(yi, -1))


def mid_rank(values, truth_idx):
    """Expected search rank under random tie order (mid-rank)."""
    p = values[truth_idx]
    return int((values > p).sum()) + (int((values == p).sum()) + 1) / 2.0


_K5 = [0.06136, 0.24477, 0.38774, 0.24477, 0.06136]


def smooth(flat, sigma):
    """Separable Gaussian on a flattened NFX x NFX grid; sigma in cells."""
    if sigma == 0:
        return flat
    if sigma == 1:
        k = torch.tensor(_K5, device=flat.device)
    else:
        xs = torch.arange(-2 * sigma, 2 * sigma + 1, device=flat.device,
                          dtype=torch.float32)
        k = torch.exp(-0.5 * (xs / sigma) ** 2)
        k = k / k.sum()
    h = flat.view(1, 1, NFX, NFX)
    h = F.conv2d(h, k.view(1, 1, -1, 1), padding=(len(k) // 2, 0))
    h = F.conv2d(h, k.view(1, 1, 1, -1), padding=(0, len(k) // 2))
    return h.view(-1)


def direct_fine(model_ld, el, m, device):
    """Model canvas -> fixgrid cell-center values (harness convention)."""
    R = m.cone_r0 + m.cone_v * el ** m.cone_p
    scan = RESTRICT / R
    c = ((torch.arange(NFX, device=device, dtype=torch.float32) + 0.5)
         / NFX * 2 - 1) * scan
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)[None]
    return F.grid_sample(model_ld[None, None].float(), grid, mode="bilinear",
                         padding_mode="border", align_corners=True).view(-1)


def pick_truth(batch, m, norm):
    inp, fut = batch[:, :m.max_seq_len], batch[:, m.max_seq_len:]
    origin_lat, origin_lon = denorm_latlon(inp[:, -1], norm)
    fut_lat, fut_lon = denorm_latlon(fut, norm)
    fut_elapsed = torch.cumsum(fut[..., 5] * norm.dt_scale, dim=1)
    truth = {}
    for name, secs in BUCKETS.items():
        j = (fut_elapsed - secs).abs().argmin(dim=1)
        el = fut_elapsed.gather(1, j.unsqueeze(1)).squeeze(1)
        ok = (el - secs).abs() <= TOL * secs
        dlat = fut_lat.gather(1, j.unsqueeze(1)).squeeze(1) - origin_lat
        dlon = fut_lon.gather(1, j.unsqueeze(1)).squeeze(1) - origin_lon
        cell = fixgrid_index(dlat, dlon)
        truth[name] = {"j": j, "ok": ok & (cell >= 0), "cell": cell,
                       "elapsed": el}
    return inp, truth, origin_lat, origin_lon


@torch.no_grad()
def direct_ranks_for(model, batch, truth, m, device, n_tracks,
                     with_smooth_control=False):
    uniq_h = sorted({int(truth[b]["j"][t]) + 1 for b in BUCKETS
                     for t in range(n_tracks)})
    ld_chunks = []
    for s in range(0, len(uniq_h), 32):
        ld, _, _, _ = model.forward_train(
            batch, horizon_indices=torch.tensor(uniq_h[s:s + 32],
                                                device=device), causal=False)
        ld_chunks.append(ld)
    ld_all = torch.cat(ld_chunks, dim=1)
    hpos = {h: i for i, h in enumerate(uniq_h)}
    out = {b: [] for b in BUCKETS}
    ctrl = {b: [] for b in BUCKETS}
    for b in BUCKETS:
        for t in range(n_tracks):
            if not truth[b]["ok"][t]:
                continue
            h = int(truth[b]["j"][t]) + 1
            fine = direct_fine(ld_all[t, hpos[h]],
                               float(truth[b]["elapsed"][t]), m, device)
            out[b].append(mid_rank(fine, truth[b]["cell"][t]))
            if with_smooth_control:
                # kernel must act on PROBABILITIES, not log-density
                ctrl[b].append(mid_rank(smooth(fine.softmax(dim=-1), 1),
                                        truth[b]["cell"][t]))
    return out, ctrl


def p90(v):
    return round(float(torch.tensor(v, dtype=torch.float32)
                       .quantile(0.9)), 1) if v else None


def main_direct_only(n_batches):
    device = torch.device("cuda")
    cfg = load_config(CFG, PretrainConfig)
    m, norm = cfg.model, cfg.normalization
    model = build_model(m, norm)
    model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                     weights_only=False)["model"])
    model.to(device).eval()
    ds = ShardedWindowDataset(VAL_DIR, batch_size=256, seed=SEED,
                              shuffle_shards=True)
    it = iter(DataLoader(ds, batch_size=None))
    agg = {b: [] for b in BUCKETS}
    for i in range(n_batches):
        batch = next(it).to(device)
        _, truth, _, _ = pick_truth(batch, m, norm)
        r, _ = direct_ranks_for(model, batch, truth, m, device,
                                batch.shape[0])
        for b in BUCKETS:
            agg[b].extend(r[b])
        print(f"batch {i + 1}/{n_batches}", flush=True)
    print(f"\nDIRECT-ONLY harness validation ({n_batches} shuffled batches)")
    for b in BUCKETS:
        print(f"{b:<5} n={len(agg[b]):>5}  p90={p90(agg[b])}"
              f"  (rescore_v2 small-cone: 7/18/59/120)")


def main():
    global CADENCE_S
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(SEED)
    n_tracks = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    if len(sys.argv) > 4:
        CADENCE_S = float(sys.argv[4])
    cfg = load_config(CFG, PretrainConfig)
    m, norm = cfg.model, cfg.normalization
    model = build_model(m, norm)
    model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                     weights_only=False)["model"])
    model.to(device).eval()

    ds = ShardedWindowDataset(VAL_DIR, batch_size=n_tracks, seed=SEED,
                              shuffle_shards=True)
    batch = next(iter(DataLoader(ds, batch_size=None))).to(device)[:n_tracks]
    inp, truth, origin_lat, origin_lon = pick_truth(batch, m, norm)

    direct, direct_ctrl = direct_ranks_for(model, batch, truth, m, device,
                                           n_tracks, with_smooth_control=True)

    # rollout: 1.2x last bucket so tolerance-edge truths stay reachable
    n_steps = math.ceil(max(BUCKETS.values()) * (1 + TOL) / CADENCE_S)
    window = inp.repeat_interleave(K, dim=0)
    stats = {"edge": 0, "over_speed": 0, "s1_sample_logp": [],
             "s1_entropy": [], "step": 0}
    ens_lat = torch.empty(n_steps + 1, n_tracks * K, device=device)
    ens_lon = torch.empty(n_steps + 1, n_tracks * K, device=device)
    for step in range(1, n_steps + 1):
        stats["step"] = step
        window = sample_step(model, window, m, norm, device, gen, chunk, stats)
        la, lo = denorm_latlon(window[:, -1], norm)
        ens_lat[step], ens_lon[step] = la, lo
        if step % 20 == 0:
            print(f"  rollout step {step}/{n_steps}", flush=True)

    roll = {s: {b: [] for b in BUCKETS} for s in SIGMAS}
    in_restrict = {b: [] for b in BUCKETS}
    for b in BUCKETS:
        for t in range(n_tracks):
            if not truth[b]["ok"][t]:
                continue
            # per-track TIME ALIGNMENT: ensemble step nearest the truth's
            # actual elapsed time (must-fix: nominal-time scoring charged
            # the tolerance window to the rollout arm only)
            st = min(max(int(round(float(truth[b]["elapsed"][t])
                                   / CADENCE_S)), 1), n_steps)
            sl = slice(t * K, (t + 1) * K)
            dlat = ens_lat[st][sl] - origin_lat[t]
            dlon = ens_lon[st][sl] - origin_lon[t]
            cells = fixgrid_index(dlat, dlon)
            valid = cells[cells >= 0]
            in_restrict[b].append(len(valid) / K)
            hist = torch.zeros(NFX * NFX, device=device)
            if len(valid):
                hist.scatter_add_(0, valid,
                                  torch.ones_like(valid, dtype=hist.dtype))
            for s in SIGMAS:
                roll[s][b].append(mid_rank(smooth(hist, s),
                                           truth[b]["cell"][t]))

    total = n_tracks * K * n_steps
    s1_lp = torch.cat(stats["s1_sample_logp"]).mean()
    s1_ent = torch.cat(stats["s1_entropy"]).mean()
    out = {"n_tracks": n_tracks, "K": K, "cadence_s": CADENCE_S,
           "gates": {"s1_sample_logp": float(s1_lp),
                     "s1_neg_entropy": float(-s1_ent),
                     "edge_frac": stats["edge"] / total,
                     "over_speed_frac": stats["over_speed"] / total,
                     "in_restrict": {b: (sum(v) / len(v) if v else None)
                                     for b, v in in_restrict.items()}},
           "buckets": {}}
    print(f"\nGATES  step-1 E[log p(sample)]={s1_lp:.3f} vs "
          f"-H(direct)={-s1_ent:.3f} (should be close) | "
          f"edge={stats['edge'] / total:.4f} "
          f"overspeed={stats['over_speed'] / total:.4f}")
    print(f"\n{'bucket':<6} {'n':>4} {'direct':>8} {'direct+s1':>10} "
          + " ".join(f"roll s={s:>1}" for s in SIGMAS))
    for b in BUCKETS:
        n = len(direct[b])
        row = {"n": n, "direct_p90": p90(direct[b]),
               "direct_smoothed_p90": p90(direct_ctrl[b]),
               "rollout_p90": {str(s): p90(roll[s][b]) for s in SIGMAS},
               "in_restrict": out["gates"]["in_restrict"][b]}
        out["buckets"][b] = row
        print(f"{b:<6} {n:>4} {row['direct_p90']!s:>8} "
              f"{row['direct_smoothed_p90']!s:>10} "
              + " ".join(f"{row['rollout_p90'][str(s)]!s:>8}" for s in SIGMAS))
    json.dump(out, open("/home/paul/data/trackfm/ar_rollout_pilot.json", "w"),
              indent=2)
    print("written ~/data/trackfm/ar_rollout_pilot.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "direct":
        main_direct_only(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    else:
        main()
