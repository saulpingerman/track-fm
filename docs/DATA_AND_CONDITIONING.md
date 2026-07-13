# TrackFM data & conditioning plan

What goes into the model, how each input enters, and what state it's in.
Written 2026-07-13. Architecture choices trace to the verified fusion
review (research/2026-07-context-fusion-architectures.md); experiment
gates (C1/C2, D1–D5) to research/2026-07-context-conditioning.md.

## The contract

One model, any input subset. The kinematic core is the only required
input; every other channel is optional at inference and the model must
degrade gracefully to bare kinematics. Mechanically: whole-channel
dropout during training (~10–20%/channel, rates sampled from a range,
plus explicit sampling of the all-dropped case), learned null embeddings
(never zero-fill — zero is a real value), presence bits, and
forecast-interval channels ALWAYS inside the masking curriculum
(an always-present informative future channel poisons pretraining —
Forecast-MAE shortcut lesson [CONFIRMED]).

Evaluation contract: the availability matrix (bare kinematics / each
channel alone / all channels) is a standing regression test — the
bare-kinematics row must never fall below the no-extras baseline.

## Tier 0 — kinematic core (required, exists)

| field | source | notes |
|---|---|---|
| lat, lon, SOG, COG→(sin,cos), Δt | cleaned DMA AIS, 7.95B rows, 26 months | the current 6-feature input, unchanged |

Enters: linear input projection → transformer trunk (as today). All
scaling-study results are on this tier alone.

## Tier 1 — per-point AIS extras (own data, no external dependency)

| field | source / state | how it enters |
|---|---|---|
| heading | in clean data; in v3 windows (−1 = missing; ~22% missing) | per-point channel; **heading−COG = free per-point drift measurement** (current/wind set acting on the hull) |
| ROT, nav status | aux change-log (extraction ~½ done) | per-point (ROT deferred) / segment-level channel; ROT is an early maneuver signal — targets the hard decile |
| Δheading vs COG | derived | the drift channel proper; validated by gate C2 against real currents |

Enters: appended per-point features through a **gated per-channel
projection** (TFT-style variable selection / sigmoid gating), NOT naive
concat — the one AIS fusion ablation that exists puts ~2/3 of the
environmental gain in the fusion mechanism [CONFIRMED], and naively
concatenated static attributes actively hurt until gated (DualSTMA
[CONFIRMED]).

## Tier 2 — vessel-static / voyage-intent (own data)

| field | source / state | how it enters |
|---|---|---|
| ship type, dims (W×L), draught | aux change-log | one "vessel token": embeddings (type) + scaled scalars, gated into the trunk |
| declared destination | aux change-log + `normalize_destination()` (registry match + confidence; junk kept, it's signal) | matched-port embedding + match confidence; unmatched strings via hashed/top-K embedding |
| declared ETA | aux change-log + `parse_eta()` (year-roll + staleness flags) | time-to-ETA scalar + staleness flag |

This tier is Paul's "classification may not exist at inference" case —
exactly what the null-embedding + dropout contract is for. Experiment
D1 (first GPU experiment: cheapest, fully own data).

## Tier 3 — static / semi-static context grids (feature-based location encoding)

| layer | resolution / state | how it enters |
|---|---|---|
| bathymetry (EMODnet DTM) | ~115 m, **on disk** | egocentric crop channel (Stage 2); depth-vs-draught is a hard physical constraint |
| distance-to-lane, lane bearing | ~500 m rasters, **on disk** (OSM seamarks; lanes are centerlines — mask is a 1 km buffer, documented) | Stage 1′ local sample AND Stage 2 crop channels |
| lane / anchorage / restricted / wind-farm masks, distance-to-windfarm | ~500 m, **on disk** (Skagen anchorage = known OSM gap) | Stage 2 crop channels |
| harbours + port table | 456 ports / 514 anchorages (dwell-validated, eps=0.75) | distance-to-nearest-port feature; port-task labels |
| traffic-density climatology | planned (self-derived from our own AIS) | crop channel; also closes the OSM lane-area gap — real lanes emerge from density |
| sea ice concentration + thickness (FMI) | 1 km daily, Oct–May, east of ~9°E, pulling | crop channel with strong seasonality; strongest measured non-weather effect in the literature (62–67% of Baltic speed variance) |

Rationale: encode places by WHAT they are, not where they are — behavior
learned around a feature type pools across all its instances and
transfers to feature-similar places with little local data (Paul's
generalization hypothesis; deepSSF + fairway-fusion precedents). The
egocentric formulation already prevents memorizing absolute position.
Killer test: **D5 spatial-holdout transfer** (train without Bornholm
waters; feature-conditioned vs kinematics-only on the held-out region).

## Tier 4 — fast fields (hourly; forecastable at inference)

| field | product | resolution | state |
|---|---|---|---|
| 10 m wind u/v, MSL pressure | ERA5 (reanalysis — training) | 0.25°, hourly | pulling (~85 MB) |
| surface currents u/v | CMEMS Baltic (2 km, hourly, ≥9°E) + NWS multiyear (7 km, hourly, ≤13°E) | pair required; overlap 9–13°E kept in both | pulling (~7 GB) |
| significant wave height | CMEMS Baltic (1 h) + NWS reanalysis (1.5 km, 3 h) | same pair | pulling |
| **archived real forecasts** 10u/10v/msl | ECMWF open-data (HRES), WITH issuance times | 0.4°→0.25°, daily runs | **on disk, 767 runs** (ensemble also available in the bucket) |

How they enter, by stage:

- **Stage 1′ (first weather experiment, D2):** sample each field at every
  input ping — bilinear in space, linear in time to the exact ping
  timestamp (alignment fidelity alone is worth ~11% [CONFIRMED]) — via
  `trackfm.context.fields.FieldStore` keyed on the v3 window t0 +
  cumsum(Δt). Gated projection, same as Tier 1. Future conditioning:
  sample along the DR extrapolation at the target time.
- **Stage 2:** egocentric field crop **ahead of and rotated to** the
  vessel, sized to the reachable set at the horizon (DeepTP/Pang
  [CONFIRMED]), small CNN (≥32-d embedding, no pooling), conditioning
  the **density decoder**, not just the trunk — up to ImplicitO-style
  continuous (x,y,t) query readout where each density query learns where
  in the field to look [CONFIRMED, largest single verified gain].
- **Stage 3 (only if Stage 2 shows field structure matters):** per-
  variable field tokens + latent-query compression (ClimaX pattern
  [CONFIRMED]), cross-attention from track tokens; forecast-interval
  fields enter as unmasked horizon-position tokens, ingested one-shot
  for the whole horizon (no recursive re-querying).

**Reanalysis→forecast discipline (D4, non-negotiable for operational
claims):** train the conditioning on ERA5/CMEMS (long archive), then
adapt the conditioning pathway on paired forecast/analysis data and
evaluate under archived real ECMWF forecasts — LT3P showed a ~2×
residual gap when this is skipped [CONFIRMED]. The FieldStore enforces
issuance-time ordering as a hard error (`LeakageError`), so operational
evaluation physically cannot see a forecast run that hadn't been
published at the window's origin.

## Zero-GPU gates before any conditioned training

- **C1 headroom:** regress DR-residual difficulty on local field values.
  If fields explain ~none of the hard tail, stop — expected gains are
  diluted anyway (~92% of an AIS benchmark was open-water; report by
  strata, never averages).
- **C2 drift audit:** heading−COG vs CMEMS currents + ERA5 winds —
  validates the drift channel, the field join, and the physical premise
  in one shot. Runs as soon as the pulls land (needs no v3 rebuild).

## Experiment order (GPU, post-flagship-gate)

D1 vessel-intrinsic (Tier 1+2) → D2 Stage-1′ local weather (Tier 4) →
D3 Stage-2 egocentric crops (Tiers 3+4) → D4 forecast adaptation →
D5 spatial holdout (Tier 3). Each A/B at Medium on identical splits,
judged on difficulty- and weather-stratified containment/CE (never
averages alone), logged in DECISIONS.md with outcome interpretations.

## Plumbing status

| piece | state |
|---|---|
| v3 windows (t0 + mmsi + heading per window) | code shipped + tested; **full rebuild queued post-campaign** (blocks D2+; D1 partially, C1/C2 not at all) |
| FieldStore (bilinear lat/lon/t + leakage enforcement) | shipped, tested |
| aux change-log (draught/destination/ETA/nav/dims) | extracting, ~½ done |
| destination/ETA interpretation | shipped, calibrated on real data |
| port labels v2 | ports.parquet done; window build running |
| masked-channel training code (dropout + null embeddings + gating) | **not yet written** — next code milestone, built with D1 |
