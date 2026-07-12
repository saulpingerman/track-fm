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

(Verified data-source specifics, prior-art map, and architecture
evidence to be merged when the deep-research pass completes.)
