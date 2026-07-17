# Conditioning literature scan (2026-07-17)

Actionable distillation for the Tier-3 conditioning build. Confidence
tags: [FETCHED] = read this session from source; [MEMORY] = from prior
knowledge, verify before citing in the paper.

## 1. The density-head + map-crop design has a winning precedent in AV forecasting

- **HOME** (Gilles et al., 2021, arxiv 2105.10968) [FETCHED abstract/
  search]: future-position **probability heatmap on a grid** decoded by
  a CNN from rasterized map + agent context. Was **#1 on Argoverse**
  motion forecasting; heatmap output "enables multimodal ensembling and
  improves 1st place MissRate6 by more than 15%".
- **GOHOME** (2109.01827, ICRA 2022) [FETCHED]: replaces full-square
  raster CNN with lane-graph encoding producing **local curvilinear
  rasters per lane** combined into the heatmap — cheaper, 2nd on
  Argoverse, SOTA nuScenes/Interaction.
- Read-across: our Fourier/direct density head IS the marine heatmap
  head. AV showed (a) heatmap outputs beat regression sets when the
  environment constrains motion; (b) map context enters most
  effectively CLOSE to the output grid (HOME: CNN raster; GOHOME:
  per-lane rasters) — i.e., exactly the canvas-registered crop + bias
  design, not encoder-side token mixing. Our traffic flow_u/flow_v
  channels are the raster analogue of GOHOME's lane directionality.

## 2. EnvShip-Bench (2606.15240, June 2026) — benchmark ON OUR DATA [FETCHED]

- DMA (Danish) + NOAA AIS, unified protocols; ~40k-sample curated
  subset (cargo/tanker/passenger/ferry); ADE/FDE + minADE/minFDE.
- Context extensions: rasterized land/water/navigable layers,
  nearest-shore/barrier distance; **neighbors within 5000 m with
  relative history, type, velocity, CPA, TCPA** — independently
  validates the traffic-gate feature set (our C1-traffic used the same
  geometry; 5 km radius matches our n5 sweet spot).
- KEY NEGATIVE RESULT: their context injections (Social-LSTM env
  branch, land-penalty "feasibility-aware losses") give only modest
  gains (63.4->60.6 m ADE) and "context-augmented variants do not yet
  surpass the strongest trajectory-only baselines."
- Read-across: naive context injection (side branch + penalty loss on
  point trajectories) is the design that DOESN'T work. Nobody in the
  marine literature has done canvas-registered density conditioning —
  the AV-proven design. This is the paper's novelty slot. Also a
  ready-made external benchmark: decode our density argmax/expectation
  to points and report ADE/FDE on their subset.

## 3. Inland GMM+Transformer context work (2406.02344) [FETCHED]

- Preprocesses waterway context into **statistics-based feature
  vectors** (GMM density curves of lateral positioning + speeds),
  fed alongside spatio-temporal features; beats a prior inland
  transformer. Lesson: hand-preprocessed context features > raw
  context dumped on the model at small scale — consistent with our
  gates-first approach and with giving the CNN normalized O(1)
  channels rather than raw counts.

## 4. Implications adopted

1. Bias-into-logits at the output canvas (HOME-style) is the primary
   mechanism; encoder-side context tokens stay deferred.
2. Traffic crops (density + flow direction) are the marine lane-graph:
   prioritized with static geography (matches C1-traffic gate result).
3. Keep EnvShip-Bench as external validation target; their metrics
   (ADE/FDE) need a point-decode utility on our density head.
4. Their negative result predicts our S1/S2 ablations must show the
   mechanism works where naive injection failed — the zero-init bias
   + proper-scoring-rule loss is the bet.

Sources: arxiv.org/abs/2105.10968, arxiv.org/abs/2109.01827,
arxiv.org/abs/2606.15240 (EnvShip-Bench), arxiv.org/abs/2406.02344.
