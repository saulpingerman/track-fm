# Training on easy-dominated data: verified strategies (July 2026)

**Status: active** — feeds the hard-trajectory experiments queued after the
flagship run. All claims 3-vote adversarially verified against primary
sources (9 findings, 1 refuted). Motivation: straight transits dominate
the 200M-window corpus; maneuvers/port approaches are rare but carry the
operational value (project framing; measurement via the new
difficulty-stratified eval).

## Verified findings

1. **Prune-easy is right for our regime, with sharp caveats**
   (Sorscher et al., NeurIPS 2022, arxiv 2206.14486): with ABUNDANT data,
   keep hard examples; can break power-law scaling toward exponential.
   BUT: only a few difficulty metrics work at scale, theory assumes a
   noise-free metric, and coverage collapses under aggressive pruning
   (Zheng, ICLR 2023, arxiv 2210.15809).
2. **Score-based pruning provably fails at high compression** (Ayed &
   Hayou, TMLR, arxiv 2302.06960): below ~30% kept, random pruning beats
   most score-based methods ("No Free Lunch" theorems). Use MODERATE keep
   fractions; never aggressive easy-pruning by raw score.
3. **Middle-of-distribution selection wins in LLM pretraining** (Marion
   et al., arxiv 2309.04564): drop BOTH the trivially-easy and the
   extreme-hard/noisy tails; ~30% perplexity-selected data beat full-data
   baselines.
4. **Small reference models can select for big ones, but criteria don't
   transfer across datasets** (Ankner et al., arxiv 2405.20541) — must be
   tuned on OUR data; no borrowing thresholds from language.
5. **Raw high-difficulty selection chases unlearnable noise** — our exact
   GPS-glitch worry, confirmed as the central failure mode. The fix is
   the reducible/excess-loss learnability criterion (RHO-LOSS, Mindermann
   et al., arxiv 2206.07137): difficulty relative to a reference/holdout
   model separates hard-but-learnable from hard-because-noisy.
6. **Per-example selection fails rare subpopulations under imbalance**
   (REDUCR, NeurIPS 2024): RHO-style selection alone collapses on
   worst-group accuracy; GROUP-aware reweighting (e.g., maneuvering /
   port-approach windows as an explicit group) must be layered on top.
7. **Sub-example selective loss works** (RHO-1 SLM, arxiv 2404.07965):
   loss only on high-excess-loss tokens in continued pretraining → large
   domain gains. Direct TrackFM analog: per position-horizon PAIR
   selective loss (we have 512 pairs/sample).
8. **Reference-free variant exists** (ICA, arxiv 2510.14459, medium
   confidence): in-context approximation of excess loss used as soft
   gradient reweighting — worth watching, less proven.
9. **Curriculum ORDERING at scale: weak evidence either way** — no strong
   support that easy-first or hard-first beats no ordering in large-data
   pretraining. (One anti-curriculum-harms claim was REFUTED as
   overgeneralized — treat ordering claims skeptically in both
   directions.) Selection/weighting, not ordering, is where the evidence
   lives.

## No prior art found on physics-baseline difficulty for trajectories

The review found no existing work weighting trajectory samples by
kinematic-model residual — the DR-residual difficulty signal appears to
be novel territory (open question #1 in the report). Contribution
opportunity if the experiments pan out.

## The design the evidence supports for TrackFM

NOT: aggressive static easy-pruning by raw Kalman/DR residual (fails
findings 2, 5).

INSTEAD, layered:
1. **Learnability, not rawness**: excess loss vs a reference model
   (golden-large or the flagship itself) — the DR residual proposes,
   the reference-model excess DISPOSES (filters glitches).
2. **Moderate selection as SOFT weighting or per-pair selective loss**
   (RHO-1 analog), not hard pruning; keep fraction well above the ~30%
   failure regime; consider Marion-style both-tails trimming.
3. **Group protection**: maneuvering/port-approach windows as an explicit
   group with floor weights (REDUCR lesson) — per-example criteria alone
   won't protect them.
4. Tune everything on OUR data (finding 4 — criteria don't transfer).

## Experiment plan (post-flagship, Medium scale)
A. Baseline: plain training (exists: scaling-medium-50M).
B. + soft DR-residual weighting with cap (the naive version — expected to
   partially work, partially chase noise; worth having as the ablation).
C. + excess-loss (vs golden-large) per-pair selective weighting (RHO-1
   analog) with group floor for high-residual-learnable windows.
Judge on: difficulty-stratified eval (deciles), esp. hard-decile
model-vs-tuned-DR ratio and hard-decile capture@10, with easy-decile
regression as the guardrail.


## Measured artifact contamination (2026-07-12, golden69 val)

One-step DR residuals (physics bound at ~10s median dt): median 1m,
p99 92m, **>1km (impossible) = 0.012%** of windows. Cleaning already
removed 202k single-point teleports; the survivors are multi-point
excursions and sub-threshold offsets. Implications: negligible for
training gradients (1/8000 with soft targets); MATERIAL for the extreme
difficulty tail (they concentrate there) — confirming the need for the
learnability criterion + extreme-tail trimming in any hard-example
scheme, and a physics-bound censor flag on the top stratified decile.

## Open questions logged by the review
- Does an excess-loss transform of the DR residual actually separate
  maneuvers from glitches on real AIS? (First thing experiment C tests.)
- Per-window vs per-pair vs soft weighting for density prediction?
- Where is the safe keep-fraction on our 200M-window Pareto frontier?
- Two-stage hard-subset continued pretraining vs mixed-in reweighting?
