# Compute-optimal scaling surface for maritime AIS

**Status:** queued — needs the full-26mo runs as fit support. 2026-07-18.

## Motivation

The 8-point fixed-geometry series (25k -> 116M params) holds data fixed at
50M samples, so its compute frontier bends: under the Chinchilla parametric
form L(N, D) = E + A/N^a + B/D^b, fixed D freezes the B/D^b term into a
loss floor, and the frontier saturates as N grows. Measured: pre-bend slope
~ -0.21 (nano -> large), local slope ~ -0.05 (large -> xlarge). The bend IS
the 18M capacity knee projected onto the compute axis — a data-budget wall,
not a model failure.

Waymo's motion-forecasting scaling study (arXiv:2506.08228) fit exactly
this kind of surface for driving and derived N_opt ∝ C^0.63, D_opt ∝
C^0.44, with optimal models ~50x smaller than language models at matched
compute. No such exponents exist for any maritime/AIS domain.

## Plan (post-flagship)

1. Fit L(N, D) = E + A/N^a + B/D^b to the existing grid plus the
   full-data runs as they land:
   - fixed-D axis: the 8-run 50M series (done)
   - variable-D support: small-isoflop-62M, small-26mo-b384 (done),
     flagship xlarge@full-data, plus any medium/large full-data runs the
     campaign produces anyway
   - a few cheap deliberate (N, D) cells if the fit is ill-conditioned
     (e.g. medium@12.5M, large@25M — hours each, not days)
2. Derive maritime N_opt(C) / D_opt(C) exponents a la Waymo — first such
   numbers for AIS; directly comparable to driving's 0.63/0.44 and
   language's ~0.5/0.5.
3. Paper figure: fixed-D curves with the fitted surface's floor overlaid;
   xlarge falling off the pre-bend power law is the visual argument for
   training the flagship on 26 months.

## Cautions

- Do NOT publish a fitted b (data exponent) from the current support: the
  bend is visible in only one segment (large -> xlarge at one D). Needs
  the full-data runs at minimum.
- Loss surfaces are geometry-specific (soft-target CE not comparable
  across fixed/cone); fit per-geometry, fixed first.
- "Samples" is the D unit (928-pos windows, stride 32) — document the
  token-equivalence caveat if comparing exponents across domains: AIS
  positions are far lower-entropy than language tokens or driving
  scenes, which is exactly why cross-domain constants don't transfer.
