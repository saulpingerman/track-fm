# Run naming: the 2026-07-19 consistency rename

The cone/R125 series were originally named by ladder position, colliding
with the fixed series' size classes (an 18.3M encoder was called
"medium"). On 2026-07-19 the naming was made consistent EVERYWHERE —
MLflow run tags, checkpoint dirs, configs, scripts, docs, campaign.log —
with no reruns. If you find an old name in an external note, map it:

| OLD name | NEW name | encoder |
|---|---|---|
| scaling-medium-cone-50M | scaling-large-cone-50M | 18.3M |
| scaling-medium-cone-mlp-50M | scaling-large-cone-mlp-50M | 18.3M |
| scaling-medium-fixed-R125-50M | scaling-large-fixed-R125-50M | 18.3M |
| scaling-medium-fixed-R125-mlp-50M | scaling-large-fixed-R125-mlp-50M | 18.3M |

(MLflow run_uuids unchanged; the git_sha/hash-map provenance files under
~/data/trackfm/ are unaffected.)

## Current, consistent size classes (encoder params)

| size class | params | d_model / layers |
|---|---|---|
| nano | 25k | — |
| micro | 70k | — |
| mini | 244k | — |
| tiny | 485k | — |
| small | 1.0M | 128 / 4 |
| medium | 5.3M | 256 / 6 |
| large | 18.3M | 384 / 8 |
| xlarge | 116M | 768 / 16 |

A geometry suffix never changes the size class: `large-cone`,
`large-fixed-R125`, `xlarge-cone` all carry their class's encoder.
Encoder-matched geometry pairings:
- 1.0M:  small ↔ small-cone
- 18.3M: large ↔ large-cone ↔ large-fixed-R125
- 116M:  xlarge ↔ xlarge-cone

Rules:
1. New configs follow the size classes above (or raw param counts).
2. Paper tables use raw encoder param counts, never bare class names.
3. Never pair runs without checking `n_params` in MLflow.
