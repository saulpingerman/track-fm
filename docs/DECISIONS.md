# Experiment design decisions & their interpretation

Running log of design forks: what was chosen, why, and — for experiments —
what each possible outcome would mean. Newest first.

## 2026-07-14 — Cone-grid study at small scale (pre-registered)

Paul asked for a direct cone-vs-fixed comparison at the small slot (the
comparison workhorse). Setup: scaling-small-cone-50M — identical to
scaling-small-50M except grid_mode=cone: origin-centred canvas with
R(t) = 0.05 + 1.5e-4 deg/s * ELAPSED SECONDS (light-cone reachable-set
bound: maneuvers are contained by construction; only sustained speed
outliers ~>32 kn escape). AMENDED same day per Paul: growth is per TIME
DELTA, never per step — AIS report intervals are irregular, so a
per-step cone would hand a 2s-cadence vessel a 10x smaller window than
a 20s-cadence vessel at the same step (the identical bug class as
step-indexed horizon metrics). Elapsed time computed per pair from the
cumulative dt channel, mirroring the encoder's own conditioning math.
Canvas-normalized loss (targets/R, sigma and dr_sigma scaled by
1/grid_range); degenerate-cone==fixed equivalence AND per-time-not-
per-step growth are unit-tested.
Queued behind the neutral-window xevals (cone_study_queue.sh).

Judged via the cross-geometry harness (shared-window CE excluding
clamped samples + containment in km^2), NEVER raw CE across geometries.
Physical resolution: cone is SHARPER than fixed below h~170 (R<0.3),
coarser above (at h800: lobe ~9 km vs fixed 2.8 km) but with ~99%
ceiling everywhere vs fixed's ~75% at 2h.

Outcome interpretation:
- Cone ~= fixed on shared-window CE at all horizons -> cone dominates
  (same interior quality + no censoring + constant cost): becomes the
  default geometry for long-horizon work and a strong flagship
  alternative; fixed grids survive only for paper continuity.
- Cone worse at long horizons on shared window (resolution trade bites
  harder than scale-stationarity helps) -> fixed+bigger stays the
  flagship answer; cone reserved for >4h horizons where fixed is
  impossible anyway.
- Cone BETTER at short horizons (its resolution advantage below h~170,
  plus scale-stationary targets) -> consider hybrid: cone for training
  signal, evaluate anywhere via the continuous density.

## 2026-07-14 — Cross-geometry eval harness (flagship vs family), incl. the clamp bias

Family (±0.3/64) and flagship (±0.9/192) share CELL PITCH (0.009375°),
so rank/containment metrics compare in physical units. Raw CE does NOT
compare: bigger-grid normalization, and — Paul's catch — the CLAMP BIAS:
small-grid training clamps ~25% of long-horizon targets to the boundary,
teaching family models an "edge exit bucket". Scoring a shared window
with clamped targets rewards that artifact on exactly the samples the
flagship predicts better (its mass is correctly OUTSIDE the window).

Harness spec:
1. Flagship density evaluated on the ±0.3° window via the continuous
   Fourier form, renormalized over that window.
2. Shared-window CE compares ONLY samples with true targets strictly
   inside ±0.3° (no clamped samples in any head-to-head number).
3. Off-window fraction reported separately; flagship performance there
   reported WITHOUT comparator (capability only it has).
4. Containment: ranks among the shared capturable subset (+ both
   ceilings), cells == km² since pitch matches.
Comparison is thereby conservative toward the flagship — a shared-subset
win cannot be attributed to grid size. Family-internal results (7-pt
curve, floor fit) unaffected: identical clamping distribution + baselines
clamped identically. Cone grid remains the artifact-free fix long-term.

## 2026-07-14 — isoFLOP control OUTCOME: capacity, not compute

small pushed to medium's exact total training FLOPs (62M samples,
+24%): val CE 1.6557 -> 1.6344. Closes only 19% of the small->medium
gap; medium stays 0.089 ahead at matched compute. p90rank_2h 113 ->
106 (vs medium's 76). The same-samples scaling comparisons stand as
capacity effects. Implied data exponent at small ~0.060 (== the params
exponent at the small end). CONSEQUENCE FOR THE GATE: the xlarge@50M
prediction (1.417) UNDERESTIMATES a real flagship, which trains on
~190M windows (3.8x this family's data budget) — and the data axis
still pays even for a capacity-saturated small model, so it pays more
for larger ones. Treat 1.417 as the flagship's pessimistic bound; the
data axis adds an unquantified but positive margin (not measurable
without a large-scale data-budget run).

## 2026-07-13 — Gate evidence in: 7-point curve bends; 2x2 flips (with one confound left)

**Scaling curve (identical protocol, 26mo, 50M samples):** nano 2.078 /
micro 1.945 / mini 1.801 / tiny 1.730 / small 1.656 / medium 1.545 /
large 1.493. Pairwise exponents: ~0.06 flat across nano->small (clean
power law), then 0.042, 0.027 — the curve BENDS above ~1M params. A
floored fit L = 1.25 + 5.79*N^-0.191 matches all 7 points (max resid
0.009) and predicts xlarge ~1.417 (−5.1% vs large) and a hypothetical
1B model only 1.361: **capacity alone is approaching an irreducible
floor ~1.25 on kinematics-only inputs.** The remaining headroom argument
shifts to the INFORMATION axis (context conditioning — C2 already shows
the physical coupling is real).

**2x2 data-vs-capacity — CONTAMINATION CAUGHT (Paul), one cell retracted:**
The headline cross-eval cell "small@69d on 26mo test = 1.184, beats
1.282" is INVALID: golden69 TRAIN = 2025-01-01..02-14 sits inside v1
TEST = 2024-12-09..2025-02-26 — the model was scored partly on days it
trained on. Retracted; never cite it. Cells that survive:
- Feb 20-26 window (clean for both: after golden train, inside v1
  never-trained test): small@69d 1.242 vs small@26mo 1.300 (69d +4.5%)
  — but 69d has a RECENCY+SEASON advantage (6-day vs 5-month gap to
  train data) on top of the batch confound below.
- CONFOUND 1 (optimizer): small-69d = batch 384 / 130,208 steps;
  small-26mo = 1638 / 30,525 (4.3x fewer updates). Resolver armed:
  scaling-small-26mo-b384 after the isoFLOP control (b384_queue.sh).
- NEUTRAL WINDOW armed (valwindow_xeval_queue.sh): v1 val days
  2024-09-22..12-08 — never trained on by EITHER model. Caveats: 26mo
  model checkpoint-selected on it (mild optimism); 69d model
  generalizes BACKWARD in time; autumn window less season-matched to
  golden. All three checkpoints (69d, 26mo-b1638, 26mo-b384) evaluated
  there once b384 finishes.
Verdict on the data axis: OPEN until the neutral-window + batch-matched
results land. Lesson institutionalized: any cross-dataset eval must
begin with a MANIFEST split-overlap check — cross-eval date ranges are
now a mandatory line in every such report.

## 2026-07-13 — Port clustering eps: 3.0 -> 0.75 km (chaining at full density)

At full corpus density (1.46M dwells vs the 165k it was tuned on), DBSCAN
eps=3.0 CHAINS: one 201k-dwell cluster spanned the whole Oresund and
swallowed Copenhagen; Fredericia merged into Middelfart; Ronne into a
west-Bornholm blob. The binned-DBSCAN memory fix was verified faithful —
this is pure density-dependent chaining. Selection criterion (Paul's
cross-referencing methodology turned into a validation set): eps must
recover the 22 independently-verified major commercial ports. Sweep:
0.75 km -> 22/22 recovered (970 clusters, 456 ports); 1.0 -> 21/22;
1.5 -> 20/22; 3.0 -> catastrophic. Chose 0.75. Residual fragmentation is
terminal-level granularity (benign for origin/destination labels);
chaining is wrong-label (harmful). Note: eps is now a DENSITY-DEPENDENT
parameter — revalidate against the known-port set if the dwell corpus
changes materially. v2 table: 970 clusters = 456 ports / 514 anchorages;
Copenhagen = "Lynettehavnen" (nearest registry basin), 82,317 dwells.

## 2026-07-13 — FLOPs accounting for the scaling ladder (Paul's challenge)

Paul asked whether medium's win over small is just MORE FLOPS. Analytic
accounting (training/flops.py, paper geometry): the Fourier head + loss
dominate — 96% of small's per-sample FLOPs, 82% of medium's — so 5.2x
params cost only **1.24x training FLOPs** (854 vs 1062 PF @50M samples).
+24% compute cannot explain -6.7% val CE (data exponent would credit
~1%). Consequences: (1) nano..small ladder points are within 8% total
FLOPs — the ladder bottom is accidentally a clean ISO-FLOP capacity
sweep; (2) compute-matched control armed: small @62M samples (= medium's
total FLOPs), runs at the flagship gate (isoflop_queue.sh). Outcome
interpretation: small-62M ~= small-50M -> capacity, not compute, is the
currency at this rung (expected); small-62M closes most of the gap ->
FLOPs confound is real, all same-samples comparisons must be redone
compute-matched. Report scaling BOTH ways (vs params at fixed samples,
vs total FLOPs) in the paper.

## 2026-07-12 — Flagship GATED on scaling evidence (Paul's call)

Paul predicts small@26mo ~= medium@26mo — i.e. capacity is NOT the binding
constraint at 50M samples, and a week of XLarge compute would buy nothing.
The campaign's auto-launch of the flagship was therefore REMOVED
(campaign.sh killed after medium started; replaced by campaign_gated.sh).
Gate evidence, gathered before any flagship decision:
1. **7-point scaling curve** at identical protocol (50M samples, 26mo,
   paper geometry): nano 25k / micro 70k / mini 244k / tiny 485k /
   small 1.0M / medium 5.3M / large 18.3M. (True 10k impossible — the
   Fourier head + input machinery floor is ~25k params.)
2. **The 2x2** {small,medium} x {69d,26mo} — small-69d moved AHEAD of the
   flagship (it is decision evidence, not a side experiment).

Decision rule: launch XLarge manually ONLY if val CE is still falling
at the 18M point of the curve (power-law fit, not eyeball). If the curve
bends at/below ~1M: capacity demand must be re-created by adding
INFORMATION (per-point AIS features, context grids) before any big run —
scale-up compute shifts to feature/context experiments at Medium.
Related doctrine (flagship exclusivity) unchanged: whenever a flagship
DOES run, it runs solo.

## 2026-07-12 — Data-vs-capacity at the small end (Paul's challenge)

Claim "data volume is doing enormous work" from small@26mo ~= golden-
medium@69d was CONFOUNDED (varied model and data together). Decisive
cell launched: **small@69d, identical 50M-sample budget** (4.5 repeated
epochs of golden69 vs fresh 26-month windows; same model, protocol,
sample count — only data DIVERSITY differs). Combined with the running
medium@26mo vs golden-medium@69d pair this gives a 2x2 mini scaling grid
{small,medium} x {69d,26mo}.

Outcome interpretation:
- small@69d ~= small@26mo -> Paul right: Small saturates; capacity binds
  at the small end; the 26-month corpus pays only at scale (watch the
  medium/large/xlarge points for where diversity starts mattering).
  Data-scaling claims must then be made per-scale, never pooled.
- small@26mo clearly better (>~0.05 val CE or >0.15x tuned-DR ratio) ->
  data diversity pays even at 1M params; repeated-epoch training on
  small corpora underperforms fresh data at matched sample counts.
- Intermediate -> report the interaction; the 2x2 grid becomes a paper
  figure either way.

## 2026-07-12 — Saturation stop v2: median-block halves (noise-robust)

Iterated twice under Paul's challenges: (1) a rate-floor stop kills
healthy power-law runs (relative gains decay below any floor); (2) a
window-local slope projection has SNR < 1 under realistic +-1% val noise
(measured 26 consecutive false positives). Final criterion: compare
MEDIANS of the two halves of the full validation history — half-over-half
gain < 0.4% for 4 consecutive validations, minimum 24h history. Median
blocks grow with the run so noise (1/sqrt n) cannot fake saturation;
healthy power laws show ~1.5% half-over-half. Seeded sims: 10/10 noisy
power laws survive full budget, 10/10 flat runs stop within ~24h of
eligibility. Doctrine: loss selects, medians thrift, ranking monitors.

## 2026-07-12 — Campaign plan: hybrid geometries (Paul's pick)

Measured at ±0.9/192/F18: head+loss dominates at EVERY scale (~300-400
samples/s, 41% MFU) -> single-geometry campaign = ~12 GPU-days. Chosen
hybrid: **scaling study at the paper's exact geometry** (0.3/64/F12, 50M
samples per scale — doubles as the direct "paper architecture x 26 months"
comparison to Table 1) and **only the XLarge flagship at 0.9/192/F18**
(~8 GPU-days total). Horizon-scaled cone remains the efficiency play if a
future run needs both coverage and speed (would be ~5 days, ~99% coverage).

## 2026-07-12 — Head ablation OUTCOME: near-tie, mixed regime -> keep Fourier

Ran overnight at golden geometry (0.3°/64/F12), medium encoder, test split:

| metric | Fourier | Direct |
|---|---|---|
| mean vs tuned-DR | 2.46x | 2.53x |
| CE @ h800 | 2.436 | **2.256** (-7%) |
| CE @ h1 | **0.193** | 0.210 |
| median rank @ h400/800 | **1 / 1** | 2 / 2 |
| k@90 (all horizons) | ~equal | ~equal |

Exactly the predicted mixed case: direct wins CE at long horizons (the F12
band limit IS discarding structure where densities spread), Fourier wins
ranking there and CE at short. Differences are 2-7% — small.

**Decision: Fourier for the scale-up run**, because (1) the scale-up
already raises num_freqs to 18, loosening the exact band limit that
explains direct's long-horizon CE edge; (2) Fourier saves 26M params at
scale-up geometry (117.3M vs 143.5M); (3) only Fourier gives the
continuous density that the H3/containment evaluation machinery is built
on. Direct head stays as one paper line ("a per-cell softmax head matches
within 2-7% at matched compute but forfeits the continuous density") and
horizon-dependent num_freqs is logged as future work.

## 2026-07-11 — Fourier head vs direct-grid head (ablation)

**Setup.** `head_type: direct` replaces `FourierHead2D` (predict spectral
coefficients, combine with fixed cos/sin bases, softmax over the sampled
grid) with `Linear(d_model -> G^2)` per-cell logits. Same encoder, same
data (golden69), same loss; only the density parametrization differs.
Config: `configs/pretrain/ablation_direct_head.yaml` vs `golden_medium.yaml`.

**The concrete stakes at XLarge dimensions (192-cell grid):**
- Parameters: Fourier 117.3M vs direct 143.5M — the direct head spends
  +26M params just to emit the grid, and that cost scales with G^2, while
  the Fourier coefficient count is grid-independent (~2.1M at F=18).
- Compute: direct is actually CHEAPER per step at F=18 (768 x G^2 vs
  2,738 x G^2 in the logits matmul).
- Capabilities: only the Fourier head defines a CONTINUOUS density —
  arbitrary-resolution sampling, exact evaluation at H3 hexagon centers,
  physically-honest containment metrics at any footprint. The direct head
  exists only at its own cells.

**Outcome interpretation:**
- **Fourier matches or beats direct** -> the spectral smoothness prior
  earns its place: same or better forecast quality with 26M fewer
  parameters AND the continuous-density property. The paper's architecture
  section has its empirical justification; proceed to the 9-day XLarge run
  with the Fourier head, and cite this ablation when reviewers ask
  "why not just softmax over cells?"
- **Direct wins meaningfully (beyond noise)** -> the smoothness prior is
  costing accuracy — the density's band limit is throwing away real
  structure. Decisions cascade: (a) consider raising num_freqs (buys back
  sharpness at compute cost) before abandoning the head; (b) if direct
  still wins, the continuous-density property has a measurable accuracy
  price that must be weighed explicitly against the H3/containment
  workflow it enables; (c) the XLarge run's head choice must be revisited
  BEFORE committing the 9 days.
- **Direct wins only at short horizons / loses at long** (or vice versa)
  -> mixed regime: the prior helps where data is sparse per cell (long
  horizons spread mass over many cells) and hurts where the truth is
  near-delta (short horizons). Would motivate horizon-dependent num_freqs
  — worth a paper paragraph either way.

Judge on: val soft-target CE (proper score), time-bucketed containment
(15m/30m/1h/2h p90rank + capture@10), and NLL-at-truth. If the heads
disagree across metrics (e.g. direct wins CE, Fourier wins containment),
report both — that itself is a finding about what the smoothness prior
trades.

## 2026-07-11 — Density grid geometry for the scale-up run

**Chosen: ±0.9° / 192 cells / num_freqs=18** (from ±0.6/128/F12).

**The governing insight:** grid cells do NOT set the model's resolution —
the Fourier band limit does (finest lobe ~ L/(2F)). Adding cells past the
basis Nyquist rate samples the same smooth function finer. Any range
increase must scale num_freqs proportionally to hold sharpness, and THAT
(not cell count) is where compute lives. Measured cost/quality table:

| config | coverage@2h | eff. resolution | step cost |
|---|---|---|---|
| ±0.3/64/F12 (paper) | 74.5% | 1.4 km | 0.9x |
| ±0.6/128/F12 | 91.5% | 2.8 km | 1.0x |
| **±0.9/192/F18 (chosen)** | **98.8%** | **2.8 km** | **1.8x** |
| ±1.2/256/F24 | 99.7% | 2.8 km | 3.7x (rejected: last 0.9% doubles cost) |
| ±1.2/256/F12 | 99.7% | 5.5 km | 1.6x (rejected: hidden 2x blur) |

Also measured: p99 displacement is LINEAR in horizon (0.0012°/step) — the
reachable region is a cone. A horizon-scaled grid range (range ∝ elapsed
time) would give ~99% coverage at ALL horizons at 1.0x cost with constant
relative resolution; set aside in favor of the fixed grid for formulation
continuity with the paper, but it remains the principled alternative if
long-horizon censoring or compute ever bites again. DR-centered residual
grids were measured too (95.2% at ±0.6): helps but doesn't solve — the
escape tail is maneuvering vessels, not just fast ones.

## 2026-07-11 — Metrics doctrine (accumulated through review)

1. Containment/search metrics are indexed by WALL-CLOCK horizon
   (15m/30m/1h/2h, ±15% tolerance, per-bucket availability reported) —
   never by step count (step 800 spans 1.3-4.5h across vessels).
2. Early stopping selects on val soft-target CE (proper scoring rule).
   Ranking metrics (p90rank, capture@k) MONITOR, never select.
3. Report per-horizon k@90 only against the capturable ceiling; the
   off-grid fraction is its own number, never silently folded in.
4. Baseline ladder: LP -> DR(fixed sigma, paper-comparable) -> DR(tuned
   per-horizon) -> Kalman(tuned) -> TrAISformer -> TrackFM. Fixed-sigma
   ratios flatter the model (smoke: 1.71x vs 1.08x tuned); strong
   baselines are the honest headline.
