# Autoregressive generation for TrackFM

**Status:** active — rollout pilot (method 1) approved and queued as an
eval-only drain candidate; belief-state feedback (method 2) USER-APPROVED
2026-07-19, slotted post-flagship-launch (DECISIONS drain plan item 8a),
final framing gated on the pilot's result. 2026-07-19.

Anti-collapse addendum (from design discussion): the NLL objective is
mode-COVERING (collapse is not a loss minimum — dropped modes pay
unbounded log-loss on their continuations). Residual collapse doors +
levers: (1) belief-token bottleneck -> reconstruction probe, widen to
multi-token/cross-attention if needed; (2) chain-length distribution
shift -> curriculum to inference-length chains, supervise EVERY masked
step; (3) mean-conditioning shortcut -> fork-heavy curriculum segments.
Detection: K-rollout ensemble as the collapse-free Monte Carlo reference
(entropy + calibration vs chain depth). Distillation-ceiling note: the
ensemble is a DIAGNOSTIC (no gradient) or an annealed anti-collapse
regularizer — never the sole target; the primary loss stays NLL on true
posits, which aims past the rollout teacher's exposure bias (the student
can exceed the teacher because the data signal exceeds the teacher).

## Framing

TrackFM is already autoregressive in the LIKELIHOOD sense: causal
training factorizes p(track) = prod p(x_i | x_<i), which is what the
zero-shot anomaly NLL exploits. What it lacks is autoregressive
GENERATION — closing the loop on its own predictions. The obstacle:
output is a density, the input slot wants a posit. Consequence today:
multi-horizon densities are marginals with no joint consistency; you
cannot sample a coherent track.

## Method 1 — sample-and-roll ensembles (pilot; NO training)

At each step: sample a posit from the predicted next-step density
(multinomial over cells + within-cell jitter), derive SOG/COG by finite
difference, append with fixed cadence dt=60s, slide the window,
repeat. K parallel rollouts (K folded into batch). Density re-emerges
at the ensemble level: distribution at 2h = the K endpoints — a MIXTURE
OF COHERENT MODES, vs the direct head's single smeared marginal
(which must average over unresolved intermediate decisions).
Precedent: TrAISformer (four-hot sampling), MotionLM (motion-token
rollouts + clustering), DeepAR (sample feedback).

Generation needs no architecture change: pad the input window with
dummy future rows whose dt column is the rollout cadence, call
forward_train(causal=False, horizon_indices=[1]) -> (B,1,G,G) density
on the R(60s) cone canvas centered at the last posit.

Pilot design (scripts/ar_rollout_pilot.py):
- small-cone best.pt, val batches, K=256 rollouts/track, 120 steps
  (2h at 60s cadence); a few GPU-minutes, co-runnable.
- Score at 15m/30m/1h/2h: empirical ensemble density on the SAME
  fixgrid (0.6 km2 cells, restrict_deg=0.3, censoring identical to
  ranks_on_fine_grid) vs the direct head's density on the same
  tracks. Count-based ranks need smoothing at K=256 (add-eps or KDE
  bandwidth ~ 1 cell); report sensitivity to the smoothing choice.
- Sanity gates: sampled posits must lie in-canvas; derived SOG within
  physical bounds; step-1 ensemble must match the direct 1-step
  density (rollout of length 1 IS the direct prediction).

Outcome tree:
- Rollout p90 BEATS direct at 1-2h => closed-loop inference has real
  headroom (modes are real); belief-state work is justified; flagship
  inference story changes; paper section.
- Rollout matches direct => joint consistency is free but marginals
  don't improve; keep rollouts for scenario generation/anomaly only.
- Rollout collapses (drift off-manifold) => exposure bias dominates;
  direct head vindicated; belief-state (trained feedback) becomes the
  ONLY viable route to generation and inherits priority evidence.

RESULT (2026-07-19): COLLAPSE branch, on clean ground. K=256, n=256,
verification-hardened protocol (17 findings fixed), direct arm
validated vs harness at 5k tracks (7/20/57/129.5 vs 7/18/59/120).
Rollout p90 at sigma=1: 62 at 15m (direct 6.8), tie-ceiling (~2100+)
at 30m/1h/2h. Robust to cadence (10 s and 60 s) and to sampler
fidelity (bicubic 256^2 fine sampling; over-speed 5.6%% -> 0.03%%,
step-1 gate exact). Failure signature: ensembles stay 98-100%% inside
±0.3° but land compact-and-WRONG — bias, not variance explosion.

Mechanism (named): FEATURE-DERIVATION NOISE AMPLIFICATION, a specific
exposure-bias channel. Training-time SOG/COG are sensor-clean; rollout
SOG/COG are finite-differenced from sampled displacements, so each
step's kinematic features carry noise ~ the model's own per-step
predictive spread (comparable to the true motion at 60 s for a 10 kn
vessel). The model trusts kinematics it has only ever seen clean;
noisy kinematics in, biased prediction out, compounding. Possible
in-method mitigations (untested, low priority): EMA-smoothed derived
kinematics; sampling displacement jointly over multi-step segments.
Belief-state feedback avoids the channel BY CONSTRUCTION — the belief
token declares "no sensor data here" instead of fabricating it. This
is the strongest single piece of evidence behind item 8a's priority.

Known caveats: fixed-cadence rollout is slightly off-distribution for
irregular AIS (time-rope would neutralize this — synergy noted);
SOG/COG derived from sampled displacements are noisier than sensor
values; both biases are measured by the step-1 sanity gate and the
15m bucket (short-horizon agreement before compounding).

## Method 2 — belief-state feedback (gated on pilot)

Feed the density ITSELF back: the Fourier coefficient vector (the
complete band-limited parameterization — no Gaussian collapse) is
projected into a belief token for the unobserved slot. A learned
non-Gaussian Bayes filter: one forward chain, per-step honest
densities, amortized marginalization (what K rollouts do by Monte
Carlo, the chain does analytically in the weights).

Training needs NO ground-truth densities — beliefs are model-produced
on the fly; supervision stays the true next posit:
mask k future slots -> predict density at i+1 -> embed own coeffs ->
continue -> CE against true x_{i+2}. Minimizing NLL of true
continuations given belief inputs learns the correct predictive
marginal in expectation.

Design commitments (recorded for the eventual implementation):
- Zero-init belief pathway (house discipline): belief token = learned
  mask embedding + zero-init coeff projection; step-0 model identical
  to a beliefless one.
- Warm-start from pretrained checkpoint; curriculum on gap length
  k: 1 -> horizon; detach beliefs initially (differentiable beliefs =
  stage 2 of stage 2).
- Belief token carries dt + type flag; SOG/COG channels absent.
- Literature anchors: scheduled sampling (Bengio 2015), professor
  forcing, Dreamer latent rollouts, GraphCast/GenCast rollout
  fine-tuning.

What the belief chain still cannot do (vs rollouts): cross-step JOINT
structure — it propagates marginals. The two methods are complements,
not substitutes.
