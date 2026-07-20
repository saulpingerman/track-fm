# World-scale expansion: foundationness of conditioning channels

**Status:** someday — user-flagged 2026-07-20 as "consider testing way
later"; nothing blocks on this. Recorded while the thinking is fresh.

## Channel taxonomy (the working hypothesis, TO BE TESTED not assumed)

- PHYSICS channels (land, depth, sdist_coast): globally available at
  uniform quality -> hypothesized foundation-ENHANCING (convert
  memorized geography into portable rules; in sparse regions they are
  the only geographic signal available).
- BEHAVIORAL channels (traffic prior/flow): derived from AIS itself,
  region+era specific, absent exactly where data is sparse ->
  hypothesized anti-foundational as a PRETRAINING dependency.

User position: whether traffic channels are actually anti-foundational
is an EMPIRICAL question, not settled by the taxonomy. Test it when the
time comes; do not design around the assumption.

## User's design notes (2026-07-20)

1. Traffic channels can alternatively be supplied AT FINE-TUNING time
   (region-local adapter data) rather than as a pretraining input —
   keeps the pretrained backbone free of the dependency while letting
   deployments that have traffic history use it.
2. If spatial grids are consumed during fine-tuning, the ENCODER may
   need more expressive power reserved for that fusion than the
   pretraining objective alone would select for — a capacity-planning
   consideration for the flagship-after-next, not the flagship.
3. Timeline: none of this gates current work.

## Test designs (pre-registered, cheap, for when it matters)

- ANTI-FOUNDATIONALITY PROBE: linear-probe frozen encoders (conditioned
  vs unconditioned, and traffic-conditioned vs geo-only) on downstream
  tasks + a spatial holdout; degradation of probe quality or holdout
  transfer = the channel is a crutch.
- D5 SPATIAL HOLDOUT (promoted to first experiment of any expansion
  roadmap): train with a sub-region held out; conditioned vs
  unconditioned on the holdout directly measures geography-as-features
  vs geography-as-memorization. Small-cone scale, ~2 runs.
- CONTEXT DROPOUT: train with context absent p of the time; preserves
  standalone competence, makes context evidence-not-oxygen. Standard
  guardrail if fusion deepens (stage-2+).
- COORDINATE MEMORIZATION: the larger world-scale issue — absolute
  lat/lon inputs buy an untransferable coordinate atlas. Ego-relative
  encoding + feature-borne geography is the likely world-scale input
  schema; revisit at expansion design time.
