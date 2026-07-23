# Results master — canonical numbers for the v2 paper

Single source of truth for every number that enters trackfm.tex. Each
table states its harness + provenance. If a number here disagrees with
an MLflow curve, THIS file (uniform-harness rescores) wins — never mix
metric versions (audit rule).

## Data

26 months Danish AIS (DMA), Jan 2023 – Feb 26 2025, 788 days:
7.14B positions, 3.11M tracks, 45,469 vessels. Materialized v1:
168.7M windows (131.9M train-period); window = 128-posit input +
800-posit horizon (~2.6 h). 50M-sample budget runs = 38% of
train-period windows, uniform-sampled. Provenance: clean MANIFEST +
docs/MATERIALIZED_DATA.md.

## Forecasting — containment (PRIMARY metric)

fixgrid p90 rank = 90th-pct rank of true cell, ±0.3° frame, 0.6 km²
physical cells, val split, uniform harness. Multiply rank by 0.6 for
km². Provenance: ~/data/trackfm/rescore_v2.json (scripts/rescore_v2.py).

| model | params | geometry | head | 15m | 30m | 1h | 2h |
|---|---|---|---|---|---|---|---|
| **large-cone-mlp (CHAMPION)** | 18M | cone | mlp | **4** | **8** | **27** | 58 |
| large-fixed-mlp | 18M | fixed ±0.3 | mlp | 4 | 8 | 23 | **39**† |
| large-fixed-R125-mlp | 18M | fixed ±1.25 | mlp | 11 | 15 | 32 | 52 |
| large-cone-mlp-F24G128 | 18M | cone | mlp F24/G128 | 4 | 8 | 26 | 55 |
| cone-spectrum | 18M | cone | spectrum | 4 | 10 | 32 | 67 |
| xlarge-fixed | 116M | fixed ±0.3 | linear | 6 | 14 | 34 | 52† |
| xlarge-cone | 116M | cone | linear | 7 | 15 | 42 | 85 |
| large-cone | 18M | cone | linear | 7 | 14 | 40 | 87 |
| large-fixed | 18M | fixed ±0.3 | linear | 6 | 16 | 40 | 64† |
| large-fixed-R125 | 18M | fixed ±1.25 | linear | 14 | 21 | 47 | 82 |
| golden-large (69d replica) | 18M | fixed | linear | 6 | 17 | 45 | 86 |
| exp14-100M (old era, 1yr) | 100M | fixed | linear | 11 | 35 | 90 | 129 |
| exp14-18M (old era) | 18M | fixed | linear | 11 | 34 | 87 | 133 |
| exp11-116M (old era, h400) | 116M | fixed | linear | 22 | 52 | 697 | 2532‡ |

† ±0.3° fixed models only cover 81% of vessels at 2h (population
  censoring — see coverage tables). Cone/R125 cover 100%.
‡ exp11 trained h400; 1h/2h are extrapolation — report with caveat only.

Scaling (fixed geometry, linear head, 50M): nano 20/83/286/335,
small 9/26/68/109, medium 7/19/47/75, large 6/16/40/64, xlarge 6/14/34/52.
Cone small tier: small-cone 7/18/59/120, small-cone-mlp 5/18/62/121
(projector uplift concentrates at LARGE scale + short horizons at small).

## Coverage / frame decomposition (claim 2)

Populations inside frame at 2h: ±0.3° = 81%, ±0.6° = 96%, ±1.2° = 100%.
Wide-frame strict-2h, FULL population (±1.2° frame, 0.6 km² cells,
n=3,317): R125-mlp 78 < cone-mlp 84 << fixed±0.3-mlp 852 (coverage wall).
Strict ±60s, ±0.3°-censored frame (n=2,685): fixed-mlp 22, R125-mlp 30,
cone-mlp 36 → decomposition: coverage burden 22→30, multiplexing 30→36.
Sparse reporters (excluded 14%) inflate p90 ~40% for every model.
Provenance: widegrid_compare.json + DECISIONS 2026-07-20 (both entries).

## Transfer suite

### Port-destination (811 classes, 300k/75k/75k, test f1_macro / acc)
LP: fixed-mlp .254, cone-mlp .248, xlarge .245, cone-spectrum .245
(others: see MLflow trackfm/ft-port-dest sweep A; random .095).
LP-FT: fixed-mlp .346/.804, cone-mlp .335/.805, xlarge .330/.802 (+35% rel, ordering preserved).
Full FT: fixed-mlp .364/.802 ≈ cone-mlp .363/.803 (GEOMETRY GAP CLOSES).
Baselines (full 2.3M test): majority .194 acc/.001 f1; nearest-port
.199/.125; origin-copy .425/.439; dest-given-origin lookup .754 acc/.411 f1,
top-k .754/.878/.907. Encoders beat lookup at every k
(cone-mlp .776/.899/.920).

### ETA (regression, LP, test MAE / MedAE minutes)
xlarge 505/191 < cone-mlp 517/206 < fixed-mlp 524/211 < spectrum 544/213.
Ordering REVERSES vs classification: capacity wins on regression.

### Vessel type (15 classes, 677k windows, vessel-disjoint, test f1/acc/top3)
fixed-mlp .3825/.421/.707 ≈ **cone-spectrum .3824/.419/.700** (tie #1),
xlarge .378/.413, cone-mlp .366/.413, small-cone .347/.387,
golden-large .346, exp14-100M .341, exp11 .341, large-cone .335,
exp14-18M .334, large-fixed .322, ctx-geotraffic .310, exp10 .269.
Baselines: random-encoder .200/.235/.516, kinematic-stats .151/.194,
majority .009/.073. Provenance: ft_vessel_probe_v2.json,
MLflow trackfm/ft-vessel-class (*-v2-lp).

### Cross-task headlines
- Projector transfers: mlp-over-linear +18% (cone) / +25% (fixed) LP f1.
- Scale saturates on classification by 18M; wins on regression (ETA).
- Spectrum encoder: vessel #1-tie / port-dest mid / ETA last —
  head↔backbone dissociation is task-dependent.
- ctx-geotraffic: forecasting −9% stack but vessel transfer −11% f1 vs
  unconditioned sibling → optional-channel doctrine.

## Conditioning (claim 8)

geotraffic-v3: 102 vs 112 control 2h-stack (−9%). Crutch test: field-off
119.5 < control 112 → context dropout mandatory. cap=8 destroys 85% of
lane contrast; eval-time cap16 improves 30m/1h → future conditioned runs
default wider cap. Two-timescale context_lr_mult=0.1 validated.

## Spectrum head (claim 7)

Native-canvas k90: spectrum 6/8/8/8 vs cone-mlp 5/7/7/7 (both
horizon-FLAT; constant +1 cell blur). Exact analytic dilation
(slot j @ R == slot 2j @ 2R, test-pinned). Param confound: phi machinery
217k vs mlp head 628k. val_loss NOT cross-comparable (h2 vs h4 pair mix).

## Pending (update when they land)

- [ ] LR A/B: large-cone-mlp @1e-3 vs 58 (due ~3:30 PM EDT 7/23)
- [ ] LR A/B: small-cone-mlp @1e-3 vs 121 (due ~7 PM EDT 7/23)
- [ ] xlarge-cone-mlp 50M scout (due ~noon 7/24) — missing scaling cell
- [ ] Flagship (user launches; config flagship_xlarge_cone_mlp_26mo.yaml)
- [ ] Anomaly detection v2 rerun (not yet scheduled)
