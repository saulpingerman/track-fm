# Migration notes: legacy experiments → monorepo

Old repo: `track-fm` @ `archive/2026-pre-overhaul` (tag) / `archive-main` (branch).
Old pipeline repo: `ais-analysis` (archived on GitHub; absorbed as `trackfm.data`).

## Model port (`trackfm.models`)

Canonical source: `experiments/11_long_horizon_69_days/run_experiment.py`
(L505–L829). The old repo had THREE encoder definitions; divergences found
when consolidating:

| Divergence | Decision |
|---|---|
| Exp 14 adds `.clone()` on `grid_x`/`grid_y` buffers (meshgrid views break `state_dict` round-trips) | **Taken** — bug fix, no numerical effect |
| Exp 14 target-computation tweak in `forward_train` | **Not taken** — exp 11 is canonical; revisit if golden comparison drifts |
| Exp 12/13 `TrackFMEncoder` re-implementation | Superseded by `CausalAISModel.encode()` |

Equivalence guard: `tests/models/test_legacy_equivalence.py` loads the legacy
classes (`tests/reference/legacy_model.py`, extracted verbatim) and asserts
bit-identical forward outputs and loss. The paper-era 478MB checkpoint no
longer exists, so code equivalence is the only regression guard.

Pinned parameter counts (must not silently change):
small 1,004,898 · medium 5,259,234 · large 18,273,378 · xlarge 116,145,122.

## Data pipeline (`trackfm.data`)

Moved verbatim from `ais-analysis` `src/ais_pipeline/`, plus one behavioral
fix made at migration time (also applied upstream before the full cleaning
campaign ran):

- **Per-day zip-member streaming** (`io/reader.py::iter_zip_csvs`,
  `pipeline.py::iter_processed_members`): monthly DMA zips bundle ~31 daily
  CSVs; the old whole-zip path peaked at ~115GB RSS on `aisdk-2023-01`
  (304M records) and OOM-killed a 125GB machine. Members are now processed
  and written one day at a time. Track continuity is unaffected (members are
  date-ordered, same as the daily-file regime).

## Materialization (`trackfm.datasets`)

Port of `materialize_samples.py` (ais-analysis history @ a9d516f, deleted in
1575f14) with changes:

- S3 → local filesystem; input is day-partitioned clean parquet directly
  (the intermediate `shard_by_track` step is dropped — per-track buffering
  over date-ordered days replaces it).
- **Temporal split** (train/val/test by day ranges, 80/10/10) instead of the
  legacy track-shard split. Tracks spanning a split boundary are truncated at
  the boundary day (buffers reset between splits), preventing leakage.
- Legacy constants kept: window 928 (128+800), stride 32,
  features `[lat, lon, sog, cog, dt_seconds]`, zstd-3 parquet,
  FixedSizeList<float32>[4640].
- No SOG/length filtering by default (matches legacy materialization).
  Exp-11-style filters (min_sog 3.0, min length 600) available via config.
  NOTE: if `min_sog_knots` > 0, `dt_seconds` is NOT recomputed after
  dropping slow positions.

## Training (`trackfm.training`)

Loss semantics identical to exp 11 (soft-target KL, random horizon sampling,
causal subwindow training; sigma 0.003, dr_sigma 0.05). Infrastructure new:

- bf16 autocast (no GradScaler) instead of fp16 manual AMP
- MLflow (local server, systemd unit) instead of print-to-log
- Time-based validation (default 30 min) instead of every-20-batches;
  early-stop patience raised accordingly
- Cosine schedule with real warmup (old warmup_steps=20 was tuned for
  69 days of data)

## Known deviations to watch in the golden comparison

1. Cleaning config differs from exp 11's pretrain filters (pipeline
   gap-segments at 4h vs exp 11's 30-min re-segmentation; pipeline keeps
   SOG ≥ 0.5-style validation while exp 11 filtered min_sog 3.0). The
   materialized windows therefore include slower/gappier segments than the
   paper's 69-day dataset. If the golden 69-day run diverges badly, enable
   the config filters and re-materialize.
2. dt normalization assumes ~30s cadence data (dt/300); unchanged from
   legacy.
