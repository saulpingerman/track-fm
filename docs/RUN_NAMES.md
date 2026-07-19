# Run-name → encoder-size legend (read this before pairing ANY runs)

The cone series was misnamed by ladder position instead of size class
(a mistake, kept for provenance — MLflow/checkpoints/logs reference
these names). The collision: **cone's "medium" is fixed's "large"**.

| run name prefix | encoder params | d_model / layers | geometry |
|---|---|---|---|
| scaling-nano | 25k | — | fixed ±0.3° |
| scaling-micro | 70k | — | fixed ±0.3° |
| scaling-mini | 244k | — | fixed ±0.3° |
| scaling-tiny | 485k | — | fixed ±0.3° |
| scaling-small | 1.0M | 128 / 4 | fixed ±0.3° |
| scaling-medium | 5.3M | 256 / 6 | fixed ±0.3° |
| scaling-large | **18.3M** | 384 / 8 | fixed ±0.3° |
| scaling-xlarge | 116M | 768 / 16 | fixed ±0.3° |
| scaling-small-cone* | 1.0M | 128 / 4 | cone |
| scaling-medium-cone* | **18.3M** (= large!) | 384 / 8 | cone |
| scaling-xlarge-cone* | 116M | 768 / 16 | cone |
| scaling-medium-fixed-R125* | **18.3M** (= large!) | 384 / 8 | fixed ±1.25° |

Encoder-matched comparison pairs (the only valid geometry pairings):
- 1.0M:  small ↔ small-cone
- 18.3M: **large ↔ medium-cone ↔ medium-fixed-R125**
- 116M:  xlarge ↔ xlarge-cone

Rules going forward:
1. NEW configs use explicit param-class names consistent with the FIXED
   series (e.g., a future 18.3M cone variant is `large_cone_*`), or raw
   param counts in the name.
2. Paper tables and verdict writeups use raw encoder param counts, never
   series names.
3. Never pair runs without checking `n_params` in MLflow.
