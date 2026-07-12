# Research notes

Literature reviews and forward-looking research, separate from active
engineering docs. One file per review, named `YYYY-MM-topic.md`, each
opening with a **Status** line (active / queued / someday) and a date.

| Review | Status | Summary |
|---|---|---|
| [2026-07-downstream-tasks.md](2026-07-downstream-tasks.md) | **active** — feeding the current fine-tuning campaign | Verified catalog of single-track downstream tasks + datasets (GFW fishing/gear, AIS-shutdown recipe, EnvShip-Bench, peer models on DMA data) |
| [2026-07-worldwide-positional-data.md](2026-07-worldwide-positional-data.md) | **someday** — future world-model direction | Global positional datasets across domains (maritime, aviation, vehicles, human mobility, animal tracking) for a multi-domain trajectory foundation model |
| [2026-07-method-comparisons.md](2026-07-method-comparisons.md) | **active** — defines the paper's comparison section | Published-method comparison plan: EnvShip-Bench protocol, TrAISformer head-to-head (density vs four-hot bins), GeoTrackNet anomaly baseline, NLL/calibration as the fair probabilistic meeting ground |
| [2026-07-hard-example-training.md](2026-07-hard-example-training.md) | **active** — feeds post-flagship experiments | Verified strategies for easy-dominated data: prune-easy caveats (Sorscher), learnability vs raw difficulty (RHO-LOSS/RHO-1), group protection (REDUCR), middle-selection (Marion); DR-residual difficulty appears novel |
| [2026-07-context-conditioning.md](2026-07-context-conditioning.md) | **next-major-project** — post-campaign direction | Weather/exogenous-field conditioning: staged design (local features -> egocentric field-crop head fusion -> full cross-attention); materialization v3 must add per-window timestamps |
