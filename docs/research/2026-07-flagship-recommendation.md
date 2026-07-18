# Flagship pretrain recommendation (DRAFT — slots marked TBD fill as the queue drains)

Decision to be made: the configuration of the full-26-month flagship
pretrain (geometry, capacity, sigma, head, conditioning, LR), and the
evidence for each choice. This document is assembled stage-by-stage as
the ablation queue drains; TBD slots name the run that fills them.

## 1. Data: the full-26mo argument (final numbers)

232.0M windows / 8.36M effective non-overlapping samples / 7.76G unique
posits (27.8x overlap, exact accounting validated to 0.004% —
DECISIONS 2026-07-18). The 50M-sample ablation budget equals ~30
effective epochs of the v3sub200 train slice (1.64M effective samples);
the full set carries 5.1x that unique information. Every capacity
conclusion below is conditioned on this: the ablations are
unique-data-limited, the flagship is much less so.

## 2. Geometry: cone vs fixed (+R125 coverage variant)

Evidence so far (all metrics v2, budget-matched 50M, annealed):
- cone fixgrid-2h: 118 (4.5M) / 86 (18.3M) / 82-85 (117M) — saturates
  at ~18M under the ablation budget.
- fixed fixgrid-2h: 113 / 64 / 52-54 (117M) — still scaling at 117M.
- BUT ceilings differ: cone reaches ~1.00 coverage at 2h; fixed +/-0.3
  reaches 0.81 — a fifth of 2h targets are simply off-canvas for fixed.
  Fixed's better ranks are on the easier, contained subpopulation.
- The saturation-vs-scaling asymmetry is consistent with cone's easier
  normalized task extracting less from 30-effective-epoch repetition
  (Sec. 1), NOT with an architectural ceiling.

TBD(chain5 medium-fixed_R125): does widening fixed's canvas to
R=1.25deg (full 2h coverage) preserve its rank advantage once it must
carry the whole population? If yes, wide-fixed is a real flagship
candidate; if its ranks blow up, the cone's per-horizon normalization
is doing real work and cone is the flagship geometry.

TBD(flagship-scale evidence): none of this measures the full-data
regime; the recommendation will state the geometry pick as
conditional on Sec. 1's reactivation argument.

## 3. Capacity and LR

- Under the ablation budget, params beyond ~18M buy nothing for cone
  and keep paying for fixed (Sec. 2). With 5.1x effective data, the
  117M tier is justified for the flagship on standard data-scaling
  grounds (7.8G unique posit-tokens comfortably support ~100M params;
  the case for >=300M rests on repetition efficiency and is NOT made).
- muP retrofit is in place (DECISIONS 2026-07-18): flagship peak LR
  transfers from the d_base sweep by construction.
  TBD(muP smoke tiers): cross-width smoke validation.
  TBD(base LR sweep): swept peak LR at d_base before the flagship.

## 4. Soft-target sigma

Aliasing analysis (DECISIONS 2026-07-17): at sigma=0.32 cells, 27% of
val_loss is irreducible sub-cell jitter; sigma=1.0 cell removes ~85%.
- RESULT(sig10, schedule-identical): sigma=1.0 cell LOSES on
  containment at every horizon (fixgrid p90 8/21/66/138 vs baseline
  7/19/59/118, +11-17%). The jitter is a val_loss floor, not a
  learning handicap; the wide target trains a broader density whose
  blur inflates search area (DECISIONS 2026-07-18).
- TBD(sig05): monotonicity check (0.5 cells, running).
- TBD(conformal pass at drain): broader may be better calibrated —
  calibrated area@90 gives the final verdict.
- STAGED(sig015, config ready, not armed): 0.16-cell target — tests
  the other side of 0.32. Expected flat-to-worse (the F=12 band limit
  ~2.7 cells already exceeds target sharpness), but if containment
  IMPROVES the flagship sigma should shrink further.
Outcome unless conformal reverses: flagship keeps sigma=0.003
(0.32 cells).

## 5. Head

- Fourier F=12 vs F=18/F=24: no containment gain past F=12 at small
  scale (F18/F24 runs); band limit is not the binding constraint.
- 1-hidden-layer MLP coefficient projector (d=hidden) beat the linear
  projector significantly at small scale; TBD(chain4
  medium-cone-mlp): does the MLP win persist at 18M ("vs medium-cone
  86" is the bar)?
- Direct per-cell head: trains ~2x faster, slightly worse, no
  continuous density (rules out sub-cell conformal refinement); kept
  as ablation, not flagship.
- STAGED(mdn, config ready, not armed): grid-rendered mixture-density
  head (K=8 axis-aligned Gaussians, same loss/metrics/bias mechanism —
  isolates the density FAMILY). Param-light (40 projector outputs vs
  Fourier's 1250). Post-drain bake-off candidate alongside trope and
  sig015.

## 6. Conditioning

Constraints (user, standing): the model must work on bare posits;
location-time-derived fields (geography, traffic-from-posits) are fair
game; AIS-luxury fields only ever with dropout/graceful degradation.
- Gate evidence: traffic >> weather >> nothing (C1-traffic +0.42
  sog-partialed r on 39% underway; C1-tube storm-crossing real but
  0.2% of windows; point-weather ~0).
- v1 ctx-geo run was destabilized by the unbounded bias field's DC
  drift (root-caused and fixed: centered + tanh-capped at +/-8 logits;
  DECISIONS 2026-07-18). v1 numbers are non-evidence.
- Land-leakage baseline (DECISIONS 2026-07-18): underway land mass is
  only 3-9% and tracks genuine port arrivals; moored vessels correctly
  hold ~50% mass on harbor pixels. Geography's win, if any, is in-disk
  shaping. Expectations should be modest.
- TBD(ctx-geo v2 / ctx-geotraffic v2 vs bs1024-control): the actual
  containment deltas + land-leakage comparison.
Recommendation shape: include geo(+traffic) conditioning in the
flagship ONLY if v2 shows a clean, stable containment win; traffic
prior must be rebuilt train-period-only for the flagship's train
window (leakage rule).

## 7. Evaluation protocol for the flagship (settled)

- Metrics v2 only (single-harness rescore; never mix eras). DR null
  gates as floor rows. Split-conformal calibrated area@90 columns.
- Val CIs: overlapping windows inflate naive CIs ~5.3x (Sec. 1) —
  report effective-n CIs.
- Selection on val CE; ranking metrics monitor. Test split held out
  until the paper freeze.
- Single-pass multi-horizon anchors and continuous-time RoPE
  (implemented, config-gated) are post-flagship ablations unless
  TBD(trope run) shows a step-change.

## 8. Current recommended flagship config (draft)

| axis | pick | status |
|---|---|---|
| data | full 26mo, temporal split, train-period traffic prior | firm |
| geometry | cone (R(t)=0.02+1.71e-4 t) | pending chain5 R125 check |
| capacity | 117M (d=768) | firm at current evidence |
| LR | muP-transferred from d_base sweep | pending smokes + sweep |
| sigma | 0.003 (0.32 cells) | sig10 lost on containment; conformal check pending |
| head | Fourier F=12 + MLP projector | pending chain4 |
| conditioning | geo_traffic w/ capped bias | pending ctx-v2 |
| pos encoding | index PE | trope is post-flagship ablation |

Flagship launch is the user's call; nothing here auto-launches.
