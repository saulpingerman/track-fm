# Context conditioning: weather & exogenous spatiotemporal fields

**Status: next-major-project** — Paul's direction after the scale-up
campaign. Goal: condition the model on weather over the FULL region and
FULL time span of the track, through the forecast horizon (forecast
weather at inference is operationally legitimate). Verified literature
pass running; findings will be merged here.

## The core architectural insight

TrackFM's output IS an egocentric spatial grid; weather IS a spatial
field. Align them. The conditioning ladder:

### Stage 1 — local weather features (cheap, proves value)
Sample W(lat_i, lon_i, t_i) at each input position -> append to the 6
input features (wind u/v, sig wave height, current u/v => ~12). Future
conditioning: sample along the DR extrapolation at target time. One-day
experiment at Medium once data is joined. Limitation: no field structure
(can't see the storm to the west).

### Stage 2 — egocentric weather-crop fusion into the density head (novel)
Crop the FORECAST field over the same +-0.9-degree window the head
predicts on, AT the target time tau; encode (small CNN); fuse into the
horizon conditioning — density at each candidate destination informed by
weather at that destination at arrival time. Optionally an additive
spatial prior on the Fourier logits. Answers "full grid through the
projection" in the frame that matters, destination-free.

### Stage 3 — full spatio-temporal cross-attention (research project)
Patchify the weather cube (region x track-span-through-horizon; ERA5
hourly 0.25 deg over our bbox ~= 648 cells/var/hour -> a few hundred to
2k tokens) as K/V; track tokens cross-attend (Flamingo/Perceiver style);
horizon conditioning also attends to future-time slices. Only if Stage 2
shows field context matters beyond local + destination.

## Practical facts
- ERA5 for our bbox x 26 months x ~6 vars ~= hundreds of MB — fits in
  RAM; the weather join can live in the DATALOADER (no shard rebuild)...
- ...EXCEPT windows currently store NO absolute timestamp (dt only).
  **Materialization v3 must add t0 per window** (8 bytes; queued as a
  hard requirement for the next rebuild regardless of this project).
- Train on reanalysis (ERA5); for OPERATIONAL claims evaluate with
  archived real forecasts (GFS/ECMWF) so reported skill includes
  forecast error. Report both, labeled. (Reanalysis->forecast domain gap
  is a known hazard — lit pass verifying mitigation strategies.)

## Ties to the hard-example thread
Weather should EXPLAIN part of the hard-trajectory tail (heavy-weather
rerouting looks like unlearnable kinematic surprise without context).
Add weather-severity deciles to the stratified eval; measure how much
"hardness" weather context converts to learnable. The two projects share
their evaluation machinery.

## Evaluation plan
- Stratify all containment/CE metrics by wind/wave severity deciles.
- A/B at Medium: no-weather vs Stage 1 vs Stage 2 on identical splits.
- Operational eval: archived-forecast conditioning vs reanalysis
  conditioning (the honesty gap).


## VERIFIED data sources (2026-07-12 review; 107 agents, primary sources)

**Reanalysis (training):**
- **ERA5** (CDS API, free): hourly, 1940-present. 10m wind u/v at 0.25
  deg (18x36 cells over our bbox); WAVES on a coarser 0.5 deg grid
  (9x18 cells, near-coast cells land-contaminated in Kattegat/Belts) —
  use ERA5 winds but CMEMS regional waves. **ERA5 has NO ocean
  currents.**
- **CMEMS currents/waves — a regional PAIR is required** (neither covers
  the bbox alone): BALTICSEA_ANALYSISFORECAST (~2 km, hourly, east of
  9.04 E: Kattegat/Belts/Skagerrak) + NWSHELF (~1.5 km, 7-13 E North Sea
  strip, masks Kattegat south of 57.25 N). Global multi-obs product
  (0.25 deg) too coarse for the Danish straits (Oresund ~4 km wide).

**Archived operational forecasts (honest inference-time eval):**
- ECMWF open-data AWS bucket: CC-BY-4.0, daily runs SINCE 2023-01-18 —
  covers our whole study period. IFS + AIFS.
- NCAR GDEX d084001: GFS 0.25 deg, 2015-present, 3-hourly leads to 240h,
  CC-BY-4.0.
- NEGATIVE result: WeatherBench 2's HRES Zarr ends 2023-01-10 — NOT
  usable for our period.
- CMEMS Baltic issues 10-day ocean forecasts twice daily — operational
  ocean conditioning through our 2h horizon is trivially covered.

**Honest gap:** no claims survived verification on prior
weather-conditioned VESSEL-trajectory work, conditioning architectures,
reanalysis-vs-forecast domain-gap quantification, or hydrodynamic
speed-loss literature — those sections remain design-reasoning-only for
now (the movement-context review may partially fill prior work; treat
architecture choice as our contribution to validate empirically).

## VERIFIED context-layer catalog beyond weather (2026-07-12; 99 agents)

Ranked by (measured effect) x (availability) x (grid-alignability):

1. **SEA ICE** — strongest measured non-weather effect anywhere in the
   lit: explains 62-67% of Baltic ship-speed variance (Loptien & Axell
   2014); ice-product choice moves icebreaker-need prediction F1 from
   <90% to 95% (Ocean Eng 2026). FREE daily 1km grids (Copernicus
   SEAICE_BAL_..._011_004) + FMI quarter-nm charts. Caveats: seasonal
   (Oct-May), covers east of ~9E only; effect sizes are for SPEED/mode,
   not yet density-grid routing — transfer is our experiment.
2. **BATHYMETRY** — EMODnet DTM 2024, ~115m over our exact bbox, open.
   Depth-vs-draught is a hard physical constraint; transformer
   conditioning precedent exists (Zhang 2023, 6-DoF), routing effect
   unquantified for density models (opportunity). Static field,
   perfectly grid-alignable; join is trivial (one raster, no time axis).
3. **EMODnet HUMAN ACTIVITIES bundle** — one CC-BY portal, ~18 layers:
   offshore wind farms (poly/points, region fully covered, annual
   updates — critical for 2023-25 North Sea/Baltic build-out), oil/gas,
   cables, pipelines, ports, protected areas, dredging.
4. **TRAFFIC DENSITY as an aggregate field** — EMODnet ready-made 1x1km
   vessel-density rasters, or self-derived from our own AIS (climatology
   or trailing-window). Bridges toward multi-vessel work with zero
   architecture change.
5. **TSS / shipping-lane geometry** — OSM/OpenSeaMap seamark vectors,
   open; rasterize to lane-membership/distance-to-lane fields.
6. **Evidence the program works**: inland-waterway transformer gained
   ~19-22% ADE/FDE from fusing river discharge + fairway geometry —
   direct proof exogenous navigation context helps trajectory
   transformers.

REFUTED: a claim that tides helped while weather HURT AIS prediction
(Minssen et al.) did not survive — treat tide-vs-weather rankings
skeptically; tides remain plausible for the shallow straits, unproven.

GAPS (searches rate-limited, re-run later): vessel-intrinsic
(draught/destination/ETA — we hold these in raw AIS regardless) and
traffic-interaction ML literature; economic layers (fuel/freight)
unverified.

## EXECUTION PLAN (2026-07-13) — measured steps, gated

Architecture choices follow the verified fusion review
(2026-07-context-fusion-architectures.md): gated fusion not concat, null
embeddings + channel dropout from day one, decoder conditioning, and a
reanalysis->forecast adaptation stage.

### Track A — data acquisition (CPU/network only; overlaps GPU campaign)
1. **Own-AIS extras** (no external dependency, highest value/cost):
   `heading` + absolute timestamps + ship_type ALREADY SURVIVE in
   ~/data/ais/clean (checked 2026-07-13) — heading-vs-COG is a free
   per-point drift measurement. Extraction pass over raw zips recovers
   what cleaning dropped: draught / destination / ETA / dims / nav-status
   as a compact change-log table per (mmsi, day). Per-point ROT deferred.
2. ERA5 10m winds, hourly, bbox subset (CDS API — needs free account).
3. CMEMS regional pair (Baltic ~2km + NWShelf ~1.5km currents/waves) +
   Baltic sea-ice 1km (Copernicus Marine — needs free account).
4. ECMWF open-data archived operational forecasts (VERIFIED live
   2026-07-13, no auth, s3://ecmwf-forecasts, daily runs from
   2023-01-18, 0.4deg beta early 2023 then 0.25): 10m wind/msl for bbox,
   HRES + **ensemble (enfo)** — feeds the forecast-adaptation stage and
   the ensemble-conditioning gap the review critic flagged.
5. EMODnet bathymetry DTM via WCS (VERIFIED live, no auth), static.
Storage: bbox subsets are GB-scale vs 1.4TB free — not a constraint.

### Track B — plumbing (code now; heavy rebuild AFTER gated chain)
- Materialization v3: per-window t0 (epoch s) + heading channel +
  (mmsi, track_id) refs for aux joins. Rebuild only after the gated
  scaling chain stops needing v1 (disk + IO contention).
- Field store: hourly bbox grids -> zarr/memmap; dataloader join keyed
  by t0; leakage tests (operational eval may only see fields from
  forecast runs issued <= window start).

### Track C — zero-GPU decision gates (BEFORE any conditioned training)
- C1 **Headroom gate**: regress DR residuals (our difficulty score) on
  local field values over sampled windows. If fields explain ~none of
  the hard tail, conditioning headroom is small — stop and rethink
  before spending GPU. (Review expectation: gains are diluted, ~92%
  open-water; judge on strata, not averages.)
- C2 **Drift audit**: heading-vs-COG differences vs CMEMS current + ERA5
  wind vectors — validates both the drift channel and join correctness.
- C3 Add weather-severity deciles to the stratified eval.

### Track D — GPU experiments (post-flagship-gate; each logged in DECISIONS.md)
- D1 vessel-intrinsic channel at Medium: gated projection + learned null
  embeddings + channel dropout (type/dims/draught/destination/drift) vs
  plain Medium. Cheapest, fully own data — runs first.
- D2 Stage-1' local weather (gated fusion, interpolated to exact
  ping/time) A/B at Medium, judged on weather+difficulty strata.
- D3 Stage-2 forward-rotated egocentric field crop conditioning the
  DENSITY DECODER (ImplicitO-style continuous query readout as stretch).
- D4 Forecast adaptation: evaluate reanalysis-trained conditioning under
  archived real forecasts (paired ECMWF vs ERA5); adapt conditioning
  pathway on paired data before any operational claim.
- D5 **Spatial-holdout transfer** (Paul's feature-generalization
  hypothesis, 2026-07-13): places should be encoded by WHAT they are
  (distance-to-lane, lane bearing, anchorage/restricted/wind-farm masks,
  depth), not where they are — behavior learned around a feature type
  pools across all instances and transfers to feature-similar spots with
  little local data. TrackFM's egocentric formulation already prevents
  location memorization; static layers enter as Stage-2 crop channels.
  Test: hold out a region entirely (e.g. Bornholm waters) during
  training; compare kinematics-only vs static-feature-conditioned on the
  held-out region. Feature model degrading much less = the world-model
  generalization claim in one figure, and the value proposition for
  sparse-AIS regions. Sources: OSM seamark:* (OpenSeaMap layer) + EMODnet
  Human Activities; extraction running to
  ~/data/context/seamarks/static_layers.npz. Supporting precedent:
  deepSSF (raw covariate rasters generalized where curated local
  features overfit), inland-waterway fairway fusion (+19-22%).

Paul's actions: CDS + Copernicus Marine registrations (free, minutes);
AISFormer full text via PSU library (requested 2026-07-13).

## Implied layer stack for the context grid
Static (join once): bathymetry, TSS/lanes, infrastructure masks, ports.
Slow (daily): sea ice, traffic-density climatology.
Fast (hourly, forecastable): weather/currents/waves [verified above].
Per-vessel tokens: type, dims, draught, declared destination/ETA, nav
status (from our own raw AIS; extraction pass needed — fields currently
dropped at cleaning).
