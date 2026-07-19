"""Cross-geometry containment scoring: compare checkpoints on ONE physical
basis, regardless of grid geometry (fixed vs cone) or head (fourier vs
direct).

Raw CE and raw cell-counts are NOT comparable across geometries — a cone
cell at 2h is physically ~17x a fixed cell, and the fixed grid clamps
targets its window can't reach. This scorer reports what is comparable:

  * containment in PHYSICAL km^2 to capture 90% of vessels, per time
    bucket (15m/30m/1h/2h) — the pinned search-effort metric in real units;
  * the ceiling (on-canvas fraction) per bucket, reported separately —
    a geometry that clips 25% of 2h targets simply CANNOT reach 90%, and
    that shows up here as "unreachable", not as a silently-good number.

Long horizons are the point; h1-style short-horizon CE is deliberately
not the headline.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from trackfm.config import PretrainConfig
from trackfm.datasets.loaders import ShardedWindowDataset
from trackfm.eval.horizons import TIME_BUCKETS, time_bucket_indices
from trackfm.eval.search import capture_curve, search_ranks, summarize_ranks
from trackfm.models.factory import build_model
from trackfm.training.losses import cone_elapsed_seconds, cone_ranges

LAT0 = 56.25
KM_PER_DEG = 111.32


def _cell_area_km2(cell_deg: float) -> float:
    """Physical area of one square grid cell of side cell_deg, at LAT0."""
    return (cell_deg * KM_PER_DEG) * (cell_deg * KM_PER_DEG * math.cos(math.radians(LAT0)))


FIXED_NATIVE_CELL_DEG = 0.009375   # = 2 * 0.3 / 64 — fixed-grid native cell


def ranks_on_fine_grid(log_density: torch.Tensor, targets_canvas: torch.Tensor,
                        R_deg: float, cell_deg: float | None = None,
                        target_cell_km: float | None = 1.0,
                        restrict_deg: float | None = None
                        ) -> tuple[torch.Tensor, int, float]:
    """Rank the target on a physical fine grid over the ±R_deg canvas.

    Bilinearly resamples the model's native log_density (whose canvas spans
    [-R_deg, +R_deg]² in physical degrees) onto fine cells at LAT0, then
    returns each target's rank on that grid. Comparable across geometries
    because the fine cell size is uniform, regardless of the model's
    native cell size.

    cell_deg: if given, use this square-in-DEGREES cell size (0.009375° =
              fixed grid's native cell; ~1.04 x 0.58 km ~= 0.60 km²).
    target_cell_km: else, size cells to ~this many km per side (default 1;
                    fine cells are ~1 km lat × cos(lat) km lon).
    restrict_deg: optional. If given (< R_deg), the fine grid spans only
                  ±restrict_deg (not the model's full ±R_deg canvas), AND
                  targets outside this restricted box are censored — same
                  vessel population a fixed-grid model of ±restrict_deg
                  would evaluate. Used to compare cone runs apples-to-
                  apples with fixed-grid runs (same cells, same vessels).

    Returns (ranks (B,) 1-based, or -1 if off-canvas; n_fine_cells;
    cell_area_km2 — for converting rank -> km² of area searched).
    """
    B, G, _ = log_density.shape
    device = log_density.device
    cos_lat = math.cos(math.radians(LAT0))
    extent_deg = restrict_deg if restrict_deg is not None else R_deg
    R_lat_km = extent_deg * KM_PER_DEG
    R_lon_km = extent_deg * KM_PER_DEG * cos_lat
    if cell_deg is not None:
        N_lat = max(2, int(round(2 * extent_deg / cell_deg)))
        N_lon = N_lat                                              # square in DEGREES
        cell_km2 = (cell_deg * KM_PER_DEG) * (cell_deg * KM_PER_DEG * cos_lat)
    else:
        N_lat = max(2, int(round(2 * R_lat_km / target_cell_km)))
        N_lon = max(2, int(round(2 * R_lon_km / target_cell_km)))
        cell_km2 = (2 * R_lat_km / N_lat) * (2 * R_lon_km / N_lon)
    scan = extent_deg / R_deg                                       # <= 1.0

    # fine cell centers in canvas coords ∈ (-scan, +scan); when scan<1
    # this only samples the restricted region of the model's canvas.
    y_c = ((torch.arange(N_lat, device=device, dtype=torch.float32) + 0.5)
           / N_lat * 2 - 1) * scan
    x_c = ((torch.arange(N_lon, device=device, dtype=torch.float32) + 0.5)
           / N_lon * 2 - 1) * scan
    yy, xx = torch.meshgrid(y_c, x_c, indexing="ij")
    # grid_sample expects (…, 2) = (x=col, y=row) in normalized [-1, 1]
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    fine = torch.nn.functional.grid_sample(
        log_density.unsqueeze(1).float(), grid.float(),
        # align_corners=True matches the density lattice convention
        # (values at linspace(-R,R,G) nodes — first/last node ON the
        # canvas edge). False compressed the field by (G-1)/G and
        # misregistered up to a native cell near edges (metrics v2, F3).
        mode="bilinear", padding_mode="border", align_corners=True,
    ).squeeze(1)                                          # (B, N_lat, N_lon)

    # inside the RESTRICTED box (targets in [-scan, +scan] canvas units)
    inside = ((targets_canvas > -scan) & (targets_canvas < scan)).all(dim=-1)
    lat_i = ((targets_canvas[:, 0] + scan) / (2 * scan) * N_lat).floor().long().clamp(0, N_lat - 1)
    lon_i = ((targets_canvas[:, 1] + scan) / (2 * scan) * N_lon).floor().long().clamp(0, N_lon - 1)
    flat = fine.reshape(B, -1)
    truth_flat = lat_i * N_lon + lon_i
    p_truth = torch.gather(flat, 1, truth_flat.unsqueeze(-1)).squeeze(-1)
    ranks = (flat > p_truth.unsqueeze(-1)).sum(dim=-1) + 1
    ranks = torch.where(inside, ranks, torch.full_like(ranks, -1))
    return ranks, int(N_lat * N_lon), float(cell_km2)


@torch.no_grad()
def score_geometry(checkpoint: Path, cfg: PretrainConfig, split: str = "test",
                   max_batches: int = 120,
                   fixgrid_restrict_deg: float = 0.3) -> dict:
    """Per-bucket containment in km^2, ceiling, ranks — geometry-agnostic."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, t = cfg.model, cfg.train
    cone = m.grid_mode == "cone"
    state = torch.load(Path(checkpoint).expanduser(), map_location="cpu",
                       weights_only=False)
    model = build_model(m, cfg.normalization, t.max_horizon,
                        len(TIME_BUCKETS)).to(device).eval()
    model.load_state_dict(state["model"])

    ds = ShardedWindowDataset(cfg.data_dir / split, batch_size=t.batch_size,
                              shuffle_shards=False)
    loader = DataLoader(ds, batch_size=None, num_workers=4,
                        pin_memory=device.type == "cuda")

    buckets = list(TIME_BUCKETS)                       # ["15m","30m","1h","2h"]
    rank_chunks = {b: [] for b in buckets}
    fine_rank_chunks = {b: [] for b in buckets}          # 1×1 km
    fixcell_rank_chunks = {b: [] for b in buckets}       # fixed-native cell
    fixgrid_rank_chunks = {b: [] for b in buckets}       # + ±0.3° population
    fine_ncells = {b: 0 for b in buckets}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        step_idx, valid = time_bucket_indices(batch, cfg.normalization, m.max_seq_len)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            ld, tgt = model.forward_at_indices(batch, step_idx)
        ld, tgt = ld.float(), tgt.float()
        g_rng = 1.0 if cone else m.grid_range
        if cone:
            el = cone_elapsed_seconds(batch, step_idx, m.max_seq_len,
                                      cfg.normalization.dt_scale, causal=False)
            tgt = tgt / cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)
        ranks = search_ranks(ld, tgt, g_rng)           # (B, H) canvas/physical cells
        # samples not available at a bucket are dropped from THAT bucket
        ranks = torch.where(valid, ranks, torch.full_like(ranks, -2))
        for k, b in enumerate(buckets):
            r = ranks[:, k]
            keep = r != -2
            rank_chunks[b].append(r[keep].cpu())
            # fine ranks per bucket on TWO physical grids:
            # (a) 1×1 km — operational unit;
            # (b) fixed's native cell size (0.009375° = ~0.60 km²) — for
            # direct comparison to the fixed-grid p90rank numbers.
            tau = TIME_BUCKETS[b]
            R_deg = (m.cone_r0 + m.cone_v * tau ** m.cone_p) if cone else m.grid_range
            tgt_canvas = tgt[:, k] if cone else tgt[:, k] / m.grid_range
            fr1, n_fine, _ = ranks_on_fine_grid(ld[:, k], tgt_canvas, R_deg,
                                                 target_cell_km=1.0)
            fine_ncells[b] = n_fine
            fine_rank_chunks[b].append(fr1[keep].cpu())
            frF, _, _ = ranks_on_fine_grid(ld[:, k], tgt_canvas, R_deg,
                                            cell_deg=FIXED_NATIVE_CELL_DEG)
            fixcell_rank_chunks[b].append(frF[keep].cpu())
            # (metrics v2, audit F5) fixgrid = same cell AND same vessel
            # population as a ±0.3° fixed grid — the only column that is
            # honestly apples-to-apples with fixed-grid p90 numbers.
            frG, _, _ = ranks_on_fine_grid(ld[:, k], tgt_canvas, R_deg,
                                            cell_deg=FIXED_NATIVE_CELL_DEG,
                                            restrict_deg=fixgrid_restrict_deg)
            fixgrid_rank_chunks[b].append(frG[keep].cpu())

    n_cells = m.grid_size * m.grid_size
    out = {"checkpoint": str(checkpoint), "geometry": m.grid_mode,
           "head": m.head_type, "split": split, "buckets": {}}
    for k, b in enumerate(buckets):
        ranks = torch.cat(rank_chunks[b]) if rank_chunks[b] else torch.zeros(0)
        curve = capture_curve(ranks, n_cells)
        summ = summarize_ranks(ranks)
        tau = TIME_BUCKETS[b]
        if cone:
            R = m.cone_r0 + m.cone_v * tau ** m.cone_p
            cell_deg = (2.0 * 1.0 / m.grid_size) * R
        else:
            cell_deg = 2.0 * m.grid_range / m.grid_size
        cell_km2 = _cell_area_km2(cell_deg)
        k90 = curve["k@90"]

        # fine 1x1 km grid rankings
        fine_ranks = torch.cat(fine_rank_chunks[b]) if fine_rank_chunks[b] else torch.zeros(0)
        fine_curve = capture_curve(fine_ranks, max(fine_ncells[b], 1))
        fine_summ = summarize_ranks(fine_ranks)
        k90_fine = fine_curve["k@90"]

        # fixed-cell-size grid (0.009375° = ~0.60 km²) — direct rank
        # comparison to the fixed-grid p90 numbers we've been tracking
        fx_ranks = torch.cat(fixcell_rank_chunks[b]) if fixcell_rank_chunks[b] else torch.zeros(0)
        fx_summ = summarize_ranks(fx_ranks)
        R_bucket = (m.cone_r0 + m.cone_v * tau ** m.cone_p) if cone else m.grid_range
        n_fx_side = max(2, int(round(2 * R_bucket / 0.009375)))
        fx_curve = capture_curve(fx_ranks, n_fx_side * n_fx_side)
        fx_cell_km2 = (0.009375 * KM_PER_DEG) * (0.009375 * KM_PER_DEG * math.cos(math.radians(LAT0)))
        k90_fx = fx_curve["k@90"]

        out["buckets"][b] = {
            "n": int(ranks.numel()),
            "ceiling": round(curve["ceiling"], 4),
            "native_cell_km2": round(cell_km2, 4),
            "native_k90_cells": k90,
            "native_km2_to_capture90": round(k90 * cell_km2, 1) if k90 else None,
            # (A) fixed-native-cell fine grid — SAME cell size as the fixed
            #     grid you've been reading in MLflow, so counts compare directly.
            "fixcell_k90_cells": k90_fx,
            "fixcell_km2_to_capture90": round(k90_fx * fx_cell_km2, 1) if k90_fx else None,
            "fixcell_p90_rank": fx_summ.get("p90_rank"),
            "fixcell_median_rank": fx_summ.get("median_rank"),
            # (C) fixgrid — same 0.6 km² cell AND ±0.3° population
            #     (metrics v2): the honest fixed-vs-cone column.
            "fixgrid_p90_rank": summarize_ranks(
                torch.cat(fixgrid_rank_chunks[b]) if fixgrid_rank_chunks[b]
                else torch.zeros(0)).get("p90_rank"),
            # (B) 1×1 km fine grid — operational unit; km² == cell count.
            "km2_at_capture_90": k90_fine,
            "fine_median_rank_km2": fine_summ.get("median_rank"),
            "fine_p90_rank_km2": fine_summ.get("p90_rank"),
            "fine_capture@10km2": fine_summ.get("capture@10"),
            "unreachable_reason": None if k90_fine else (
                "ceiling<0.9 (targets escape grid)" if fine_curve["ceiling"] < 0.9
                else "ceiling>=0.9 but curve never hits 90%"),
        }
    return out
