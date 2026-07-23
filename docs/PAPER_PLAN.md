# TrackFM v2 paper plan — "best pre-trained model for vessel forecasting"

Goal: publish the v2 campaign. The old draft (paper/trackfm.tex) is the
AWS-era story (69 days, 116M champion, loss-vs-DR ratios) and is superseded
in data (26 months), metrics (containment area, not loss ratios), champion
(18M-scale cone-mlp, not 116M), and downstream evidence (811-way port-dest,
ETA, 15-class vessel; not exp-12/13). Rewrite, keep the skeleton + related
work + leak-test/causal-subwindow material.

## Headline

An 18M-parameter horizon-scaled ("cone") model with a projector density
head, pretrained on 26 months of Danish AIS, forecasts every vessel at
every horizon with p90 90%-containment areas of 4/8/27/58 x 0.6 km2
(15m/30m/1h/2h) — matching static-grid accuracy where static grids apply
while covering the 19% of vessels they censor at 2h — and its
representations transfer to port-destination, ETA, and vessel-type tasks,
beating strong lookup baselines.

## Claims -> evidence (all numbers in paper/results_master.md)

1. METRIC. Operational forecasting quality = containment area at fixed
   capture (p90 rank x cell area), not loss ratios vs dead reckoning.
   Uniform harness (rescore_v2, one metric version, val). Frame +
   population stated on every table (house rule).
2. GEOMETRY. Horizon-scaled canvas (R(t) = r0 + v t, linear reachable-set
   bound) covers 100% of vessels at every horizon. Static ±0.3° censors
   19% at 2h; static-wide (±1.25°) covers but pays resolution
   (multiplexing ~1/3, coverage burden ~2/3 of the gap decomposition).
   Wide-frame strict-2h on ALL vessels: static-wide 78 vs cone 84 vs
   narrow-box 852 (the coverage wall). Cone wins 15m-1h outright.
3. PROJECTOR. One hidden layer in the density head is the largest single
   quality lever: cone 87->58, fixed 64->39 at 2h; replicates in transfer
   (+18-25% LP f1). SSL-projector connection (d0-on-mlp-features anomaly,
   head ladder knee at depth 1).
4. SCALE. Transfer classification saturates by 18M (116M ties); ETA
   regression rewards capacity (xlarge best). Forecasting containment
   also ~saturated large->xlarge on home metric.
5. FULL-FT NULL. Full fine-tuning erases geometry differences (.364 vs
   .363) -> geometry is chosen on forecasting merits, transfer is not a
   tiebreaker.
6. TRANSFER SUITE. Port-dest 811-way: LP beats majority/nearest-port;
   top-k beats origin->dest lookup at every k; LP-FT +35%, full FT .364.
   ETA: MAE 505-524 min LP. Vessel-15 (677k windows, vessel-disjoint):
   f1 .38 vs random-encoder .20 vs kinematics .15.
7. GRID-FREE HEAD (spectrum). Physical-frequency head with exact analytic
   scale-equivariance: containment -16% vs slot-mlp (uniform +1 native
   cell = smoothness-vs-oscillation cost, param-confounded) but TIES #1
   on vessel transfer -> head choice and backbone quality dissociate.
   Position as the long-horizon extension path (no grid retrain), with
   the param-matched revision as future/appendix work (or run it if
   schedule allows).
8. CONDITIONING (appendix or short section). Geo+traffic context -9%
   containment stack, two-timescale lr, context dropout crutch test; but
   transfer f1 -11% vs unconditioned sibling -> optional-channel
   doctrine: condition at fine-tune time, keep the foundation clean.
9. LR/muP (pending CHAIN15). Tier study: optimum 1e-3 stable across
   widths; A/B on champion + small running now. If 1e-3 wins, headline
   numbers improve and the recipe section documents the transfer.

## What the paper still lacks (queue candidates, in priority order)

- FLAGSHIP: the release model = full-data pretrain at the winning recipe
  (geometry=cone, head=mlp projector, LR from CHAIN15, conditioning off).
  USER LAUNCHES; everything above de-risks it.
- Anomaly detection (v1 paper had it; v2 has not rerun it — DTU data,
  exp-12 protocol, LP + FT under the new harness) — 1 evening unit.
- Baseline table for forecasting: DR/last-position under the CONTAINMENT
  metric (dr_null_gate exists; present as the null gate row).
- Region transfer (train DK -> eval elsewhere): future work unless cheap
  data appears.
- Seeds/error bars on the champion (2-3 reruns) if reviewers demand.

## Build order (CPU while CHAIN15 runs)

1. paper/results_master.md — canonical numbers, single source of truth,
   every table cites its JSON/MLflow provenance.   [THIS COMMIT]
2. Restructure trackfm.tex: new abstract/intro/contributions; Methods =
   geometry + heads + training recipe; Experiments = claims 1-7;
   Discussion = 8 + foundationness; keep related work, refresh 2024-26.
3. Regenerate figures at paper quality (coverage_story, old_vs_new,
   ft_sweep summary, head ladder, + new: transfer-vs-containment scatter
   showing the spectrum dissociation).
4. Appendices: metric definitions (v2 semantics), strict/wide-frame
   protocols, conditioning study, spectrum head math (phi, gamma,
   analytic dilation, R-binning).
