# Experiment design decisions & their interpretation

Running log of design forks: what was chosen, why, and — for experiments —
what each possible outcome would mean. Newest first.

## 2026-07-19 — decay_bias_norm flag (wd-sweep prerequisite)

Historical optimizer decays EVERY param (biases/LayerNorm included) —
invisible at wd=1e-5 (decay timescale ~3e8 steps >> any run: provably a
no-op), but divergent from universal LLM practice once the muP base
sweep raises wd toward 0.01-0.3. New `decay_bias_norm` (TrainConfig +
FinetuneTrainConfig, default True = historical, SP off-path stays
verbatim single-group). False routes ndim<=1 params to wd=0 at full lr
(4th muP group / 2-group SP split). Flip it for the wd sweep only —
never mid-series. Provenance note: pretraining wd=1e-5 was never swept
(hardcoded exp-11 default); the old Ax BO searched wd only for
downstream FT and found ~0 optimal for the pretrained condition.

## 2026-07-18 — Fine-tune wiring: LP-FT scheduling + muP grouped optimizer

Implements the decisions from docs/research/2026-07-finetuning-review.md
in the shared finetune loop (design -> implement -> adversarial verify ->
mutation-check, same protocol as the muP retrofit).

1. `FinetuneTrainConfig.strategy: full | lp | lp-ft`. lp-ft = freeze ->
   linear-probe the head to EARLY-STOP (convergence, not a fixed warm-up:
   the protection mechanism is the converged head norm, not gradient
   timing) -> reload the BEST probe head -> unfreeze -> full FT at
   `ft_learning_rate` (default learning_rate/10). Early stopping is
   phase-LOCAL (verify finding: benchmarking FT's patience against the
   probe's converged best would kill a monotonically improving FT phase);
   checkpointing is GLOBAL, so if full FT never beats the probe on val,
   the probe model ships. lp_*/ft_* fields govern ONLY the lp-ft phases;
   strategy=lp === freeze_encoder=True (historical behavior preserved).
2. Optimizer routed through `mup.build_finetune_optimizer`: muP off
   reproduces the VERBATIM historical call (AdamW over requires_grad-
   filtered params — note the pretrain off-path does NOT filter; the
   distinction is deliberate and tested). muP on: the 3-group split now
   classifies downstream wrappers — 'encoder.' prefix stripped so
   backbone params reuse the pretraining rules; fresh MLPHead roles:
   net.0 (d->d/2) hidden eta/m, net.3 (d/2->128, fan_in scales) output
   eta/m, net.6 (128->out, width-free) full eta. LR-only treatment for
   the fresh head is sufficient under Adam (update-scale argument);
   default kaiming init already gives width-invariant activations at
   init. Frozen backbone (probe phase) yields head-only groups for free
   since build_param_groups skips non-trainable params.
3. Mean pooling stays the pinned default for pretrained backbones.
4. Verification: 48-agent workflow (3 lenses + mutation testing in a
   worktree, 2 refuters per finding) -> 9 confirmed findings, all fixed:
   falsy-or LR sentinel (0.0 silently replaced), phase-local early stop,
   lp_* scoping docs, and 6 test-coverage holes now closed by
   mutation-killing tests (scripted-score loop test pins the
   overall-val-winner contract, cross-phase mlflow step monotonicity,
   distinct lp/ft budgets, both LR overrides; strategy=lp is exercised
   through run_finetune end-to-end).

Next: the four <2 GPU-h ablations after the queue drains (zero-shot
anomaly NLL w/ per-cell normalization, LP-FT vs full FT on ports w/ OOD
split, same on ETA, FT-LR transfer spot-check at 117M).

## 2026-07-18 — muP retrofit: hyperparameter transfer by construction

Retrofitted Maximal Update Parameterization (Yang & Hu 2022, Tensor
Programs V) behind `ModelConfig.mup` (default OFF = SP path, bit-for-bit
preserved and gate-tested). One LR sweep at d_base=128 now transfers to
every width — this closes the "are our hyperparameters just inherited
guesses" question for the flagship.

Design decisions (5-agent spec, 4-reviewer adversarial verify):
1. FIXED d_head=16 across the width series (nhead = d/16: 8/24/48 at
   128/384/768). With d_head constant, PyTorch's 1/sqrt(d_head)
   attention scale is a width-CONSTANT absorbed by the base sweep
   (many-heads limit is muP-valid) — ZERO attention code changes, no
   forward fork. Config validator makes the invariant unviolable.
2. Recipe (a): output-layer width correction in init variance
   (readout std = 0.01 * d_base/d, Var ~ 1/fan_in^2) + optimizer LR.
   NO forward multipliers anywhere; state_dict keys identical on/off.
3. 3-group AdamW (trackfm/training/mup.py): input-like at eta, hidden
   at eta/m, readout at eta/m (LINEAR in width — the Adam column; 1/sqrt
   appears in no muP column). wd*m on down-LR'd groups keeps AdamW's
   lr-coupled decay width-invariant. Classifier RAISES on unclassified
   params: future modules break CI, not the flagship.
4. Found + fixed a real SP width bug: fixed readout std=0.01 made
   initial logit std grow ~sqrt(d) — xlarge's softmax started ~2.4x
   hotter than small's. Width-invariant under muP; SP untouched.

Verification (adversarial workflow, verdict fix-then-ship, all must-fix
items landed): SP bit-preservation 10/10 fixtures through the production
optimizer path; eta/m independently re-derived and numerically confirmed
(eta/sqrt(m) drifts slope +0.23-0.30, eta/m stays <=0.08); mutation
testing found + closed a wd-wiring test hole and 3 real bugs in the
cross-width script (dead grad probes, NaN-crash losing partial results,
width-unpaired horizon draws polluting the 2% gate).

Tests: 141 green. SP gate = golden fixtures (5 head/geometry variants,
per-tensor sha256, 5 seeded steps) + Layer-2 no-key-vs-disabled A/B.
Coordinate check: activations width-invariant (|slope|<=0.35 over 4x
width) under muP with an SP power check + step-0 readout-temperature
probe.

Next (GPU, after the queue drains — never interleaved):
scripts/mup_crosswidth_smoke.py tier1 (~45 min bug-catcher: gates on
finiteness, wider-never-worse 2%, probe RMS ratios in [0.5,2], clip-frac
divergence <20pp) then tier2 (~3h: LR argmin alignment across widths).
Then the production base sweep at d=128 DEPTH 16 (muP transfers over
width, not depth), and the flagship inherits the swept LR unchanged.
Known-accepted deviations: PyTorch's width-shrinking default bias init
(decays toward the mu-limit; documented, not fixed — fixing would break
muP@base == SP@base); LayerNorm gains at full eta (standard practice).

## 2026-07-17 — XLARGE-CONE@50M RESULT: cone capacity saturates at ~18M under the 50M-sample budget

The budget-matched rerun completed cleanly (78,125 steps, cosine fully
annealed, validation OOM fixed by the no_grad patch). Trainer-metric
comparison (all cone runs pre-v2 metrics — internally consistent;
uniform v2 rescoring runs when the chain drains):

| model | params | fixgrid p90 15m/30m/1h/2h | ceil@2h | best val_loss |
|---|---|---|---|---|
| small-cone | 4.5M | 7/19/59/118 | 1.00 | 1.262 |
| medium-cone | 18.3M | 7/15/41/86 | 1.00 | 1.089 |
| xlarge-cone | 117M | 7/14/39/82 | 1.00 | 1.077 |
| large-fixed | 18.3M | 5/14/38/64 | 0.81 | n/c |

Findings:
1. Cone capacity scaling DECELERATES hard at this data budget: -27%
   (2h) for 4.5->18.3M, then only -5% for 18.3->117M. 6.4x params
   bought 5%. The 50M-sample budget is the binding constraint at 117M
   (data-limited regime), consistent with motion-model scaling laws
   (N_opt ~ C^0.63). The kill-was-wrong audit stands (the run was
   healthy and improving), but the fair result shows the extra capacity
   pays little AT 50M SAMPLES.
2. On the shared +-0.3 deg population, large-fixed still beats
   xlarge-cone at 2h (64 vs 82) with 6.4x fewer params — fixed remains
   more parameter-efficient inside its box; cone remains the only
   full-coverage geometry (ceiling 1.00 vs 0.81).
3. Flagship implication: full-26mo (~4x data) is exactly the move that
   should reactivate capacity scaling at 117M. The 50M-budget curve
   cannot justify >18M params; the full-data run can.
4. Queue decision: medium-cone-mlp@50M appended (NOT xlarge-cone-mlp) —
   the MLP-head effect is cleanly measurable at 18M where data does not
   bind, directly comparable to medium-cone's 86; at 117M it would be
   confounded by data starvation. One run, ~6h, pre-authorized budget.

## 2026-07-17 — Session infrastructure consolidation (metrics v2 era)

Everything below landed in one overnight autonomous session and defines
the current evaluation regime:

1. **Metrics v2** (audit F1-F7 fixes): analytic baseline log-densities,
   node-lattice truth indexing, align_corners consistency, ceiling-aware
   km2@90, fixgrid population matching, test split retired from
   selection. RULE: pre-v2 and v2 numbers never share a table;
   scripts/rescore_v2.py re-scores every checkpoint on val in one
   command (run it whenever the chain drains).
2. **Split-conformal HDR calibration** (eval/conformal.py): calibrated
   area@90 with coverage guarantees; a standing eval column from now on
   (architecture review's highest-leverage upgrade).
3. **DR null gate** (scripts/dr_null_gate.py): fixgrid bars
   26/166/1545/UNREACHABLE at 15m/30m/1h/2h. Trained small-cone
   (~8/18/65/107 under v2 smoke) clears them 3-24x at every horizon —
   the learned model earns its compute even at 15m, refuting the
   constant-velocity-saturation risk.
4. **Aliasing finding + queued sigma ablation** (see entry below);
   sig10/sig05 + bs1024 control queued behind the conditioning runs.
5. **Audit outcomes** (docs/audit/2026-07-audit.md): 7 confirmed
   findings all fixed; F24 quantified minor (documented); F27/F30/F26/
   F20/F21/F10 hardened; gate effect sizes downgraded to upper bounds
   pending block-bootstrap replication.
6. **Conditioning readiness**: v3sub200 (58.3M windows, t0+mmsi+heading
   meta) verified; matched traffic prior built at TRAIN_END=2023-06-09;
   static prior for v1 unchanged. S1/S2 run on v1 and are unaffected.
7. **Architecture review** (docs/research/2026-07-architecture-review.md,
   109-agent verified study): keep transformer+cone+CE; adopt
   continuous-time RoPE ablation, single-pass multi-horizon anchors
   (gated), conformal calibration, head bake-off; skip Mamba/ODE/
   diffusion-primary/1B-scaling. Ordered 8-day ablation plan pinned.

## 2026-07-17 — Soft-target ALIASING quantified: 27% of val_loss is sub-cell jitter

scripts/aliasing_analysis.py: with canvas sigma 0.01 = 0.32 cells at
G=64, the discretized Gaussian target's shape depends on where the true
position falls WITHIN its cell. The model cannot see sub-cell position,
so this is irreducible loss noise:

| sigma (cells) | aliasing penalty (nats) | per-sample CE noise | supervision blur |
|---|---|---|---|
| 0.32 (current) | 0.341 | +-0.492 | 0.53 cells |
| 0.50 | 0.271 | +-0.202 | 0.81 |
| 1.00 | 0.080 | +-0.049 | 1.47 |

Aliasing = 27% of small-cone's trained val_loss (1.26); the noise term
is 39%. KEY: at sigma=1 cell the supervision blur (1.47 cells) is STILL
sharper than the F=12 Fourier band limit (2.7-cell finest lobe) — the
head cannot express what the sharper target demands anyway. Widening
sigma removes ~85% of aliasing noise at zero expressible-supervision
cost.

Consequences: (a) cross-run val_loss comparisons at different sigma are
meaningless (different target entropy) — judge the ablation on
containment metrics ONLY; (b) part of every training signal to date has
been sub-cell jitter; (c) queued ablation: sig10 (1.0 cell) and sig05
(0.5 cell) at small-cone scale, sigma_chain.sh armed behind the
conditioning chain. Falsifiable expectation: sig10 improves
val_fixgrid_p90rank at all horizons or the aliasing account is wrong.

## 2026-07-17 — C1-TRAFFIC: other vessels are the strongest conditioning signal measured

Same gate design as the weather gates (2h DR residuals, 6 val days,
sog-partialed), signal = dynamic traffic computed from our own AIS at
each origin: density (n within 2/5/10km), nearest-vessel distance,
min CPA/TCPA over approaching neighbors, crossing-course indicator,
anchored count. Population fix that MATTERS: pooled stats are dominated
by moored windows (median residual 0.04km, raw density correlations go
NEGATIVE because dense areas = ports = parked ships). All headline stats
on UNDERWAY (sog>=2kn) origins: 55k windows, median residual 13.5km.

Gate scoreboard (all sog-controlled):

| gate                 | best partial r | dR^2 over sog | hard-dec r | indicator (coverage) |
|----------------------|---------------:|--------------:|-----------:|----------------------|
| point-C1 weather     |          ~0.04 |        +0.006 |         ~0 | —                    |
| tube-C1 weather      |          +0.18 |        +0.045 |      +0.08 | 3.8x on 0.2%         |
| traffic (underway)   |      **+0.42** |    **+0.097** |  **+0.13** | 1.35x on 39%         |

Traffic is a BROAD moderate signal (39% of underway windows have a
<1km-CPA approach, each 1.35x likelier hard at matched speed) vs
weather's rare strong one. Decile profile is monotone: density, CPA
rate, anchored count all rise through difficulty deciles 0-8 (dip at 9
— extreme tail may be destination changes, see Type-5 below).

Caveat kept honest: traffic partially proxies port/anchorage proximity
(arrival maneuvers happen where ships cluster). The gate can't separate
"maneuvered for traffic" from "maneuvering near port"; both are
conditioning signal, one dynamic (traffic crops) one static (port
geometry) — the architecture carries both.

ALSO: Type-5 check — raw DMA CSVs carry Destination/ETA/Draught/
NavStatus/ROT/dims ON EVERY ROW; the clean pipeline dropped them.
Destination+draught are route-determining at long horizons. Recoverable
via slim per-vessel change-log side table without a full re-clean.

Priority order for Tier-3 conditioning (updated; revised same day per
the transferability requirement — see below):
1. Static geography crops (land/bathy/dist-to-coast/port geometry) +
   traffic-prior raster — unchanged.
2. DYNAMIC TRAFFIC crops + nearest-neighbor scalars — promoted by this
   gate.
3. Weather movie-crops (K-slice) — real but rare-event.
4. Type-5 intent (destination/ETA/draught) — DEPRIORITIZED, see below.

## 2026-07-17 — Transferability constraint: no dependence on AIS-luxury posit fields

Requirement: the model (and successors trained on other data) must
work on positional data that does NOT carry rich AIS metadata. Don't hand
the model a feature so powerful (destination especially) that it stops
working — or stops being honest research — on bare posits.

Field taxonomy this induces:
- KINEMATIC CORE (lat/lon/sog/cog/dt): universal, any positional dataset.
  The model's backbone input. Never conditioned away.
- LOCATION/TIME-DERIVED FIELDS (geography, bathymetry, weather, traffic
  rasters computed from the dataset itself): always obtainable for any
  data — "fields are fine, those are always obtainable". Full speed
  ahead (priorities 1-3 above). Note traffic crops qualify: they are
  derived from the other posits in the dataset, not from AIS metadata.
- AIS-LUXURY POSIT FIELDS (destination, ETA, draught, nav status, ship
  type, dims): dataset-specific. DEPRIORITIZED as conditioning inputs.
  If ever added, ONLY with feature-dropout training (each luxury group
  masked to a learned null embedding with prob p during training) so a
  single checkpoint serves any subset at inference, and the marginal
  value of each group is measurable by toggling at eval. Never as
  always-present inputs.

Implication: the ship_type embedding idea from the Tier-3 design memo
falls under the same dropout rule if used at all.

## 2026-07-17 — C1-TUBE: storm-crossing is real but RARE; speed is the master variable

The critique of C1: a storm crossing the vessel's path mid-window is
invisible to endpoint snapshots — the field must be sampled where/when the
vessel would encounter it. Rerun with DR space-time-tube sampling
(field(DR_pos(f·tau), t0+f·tau), f∈{0,.25,.5,.75,1}), 33k val windows @2h,
sog-partialed correlations + speed-matched lifts (raw 'worsen' features
mechanically scale with tube length = sog·tau — half the raw signal was
that confound).

Results (scripts/c1_tube.py, c1_tube.json):
- sog alone: R² 0.30 overall / 0.59 hard-decile. Speed is the master
  variable of DR difficulty.
- Origin-only fields on top of sog: +0.006 R². Point-C1's null stands.
- TUBE fields on top of sog: +0.045 R² overall, +0.005 hard-decile.
  Real overall signal (mostly wave/current gradients along path), ~nothing
  extra in the bulk hard tail.
- STORM-CROSSING (wave tube-max exceeds origin by >1m): 3.8x hard-decile
  lift at matched speed — the storm scenario is REAL — but only 74/32,823
  windows (0.2%). A rare-event feature, not a bulk-difficulty feature.

Verdict: dynamic fields enter the architecture as cheap K-slice movie
crops on the canvas (they pay for themselves on the 0.2% storm windows
and the overall wave/current gradient signal), but they stay BEHIND
static geography + traffic priors in priority. The bulk of the hard tail
is still decisions, and the biggest untested decision-driver is OTHER
VESSELS (dynamic traffic fields from AIS itself) — neither C1 tested it.

## 2026-07-17 — AUDIT: the xlarge-cone kill was WRONG (schedule-position artifact)

Full audit of the 07-15..07-16 session (xlarge-cone launch, kill decision,
ablation chain), prompted by the observation that the ablations didn't explain WHY xlarge
"failed," which contradicted both the pre-registered research prediction
and the small-scale result.

**Finding: xlarge-cone never failed. It was killed at 14.5M samples (29%
of the 50M budget every comparison run got) with lr still at 2.96e-4 — 99%
of peak — because max_epochs=1 stretched its cosine over a full 189M-sample
epoch (bs=128 placeholder was never budget-matched). Every run it was
compared against had trained on 50M samples with cosine fully annealed
to 0.**

At MATCHED samples (~14.5M), on val_fixgrid_p90rank_2h (0.6 km² cells,
±0.3° population):

| samples | small-cone | medium-cone | xlarge-cone |
|---|---|---|---|
| ~14.5M | 153 | 113 | **96** — best cone ever, un-annealed, still falling |
| 50M + anneal | 118 | 86 | (killed) |

The "saturation" read was mid-cosine noise; the "15m regression 6→10" was
1 km²-cell quantization + high-LR fluctuation (fixgrid 15m was stable 7-8).
The resampling-bias hypothesis was tested directly (bilinear vs area-avg
on the xlarge ckpt): no bias, metric is sound. 95/95 tests pass. The
research doc's prediction (cone matches/beats at long horizons, gap grows
with t) was tracking correctly when the run was killed.

**Ablation chain results (all 50M, annealed, G=64/F=12 unless noted) —
still valid, they answer the confounder questions:**
- F18/F24 ≈ F12 (2h: 114/114 vs 118): num_freqs exonerated.
- medium-cone (18.3M): 86@2h — cone DOES scale with encoder.
- small-cone-mlp (head MLP 128): val_loss 1.044 (beats 18.3M medium's
  1.089!), 15m fixgrid 7→5; long horizons unchanged. Head mixing is real
  but short-horizon-only at this scale.
- direct-mlp: still behind fourier-mlp. Direct stays retired.
- G128: OOM'd (bs=1638 kept at G=128 — config error), G confound untested.

**Lessons (append to metrics doctrine):**
1. NEVER compare runs at different points of their LR schedule; anneal
   state dominates val metrics.
2. Sample budget must be part of the run name / comparison key.
3. A "flat" val curve mid-cosine at constant LR is not saturation.

**Decision: relaunch xlarge-cone budget-matched (50M samples, cosine
annealed over exactly that), at G=64/F=12 to stay on the series' axes —
the killed run's G=192/F=18 was an unablated double-change; at G=64 the
head is ~9x cheaper and bs can rise to ~512-1024, cutting wall time from
~12 days to well under a day. Optionally the same at fixed geometry for
the 117M fixed point.**

2x2 {geometry: fixed, cone} x {head: fourier, direct} at small/26mo/50M/F12
scored on the cross-geometry harness (1x1 km fine grid, k@90-of-all with
ceilings, test split, 120 batches). Full table (km² to capture 90% of ALL
vessels; UNREACH = ceiling<0.9):

  cell           15m   30m    1h     2h    ceil@2h
  cone+fourier   3.6   11.5   36    136    1.00
  cone+direct    4.2   12.7   41    146    1.00
  fixed+fourier  4.2   14.5  162    UNREACH  0.80
  fixed+direct   4.8   16.3  174    UNREACH  0.80

Ranking at every horizon: cone+fourier > cone+direct > fixed+fourier
> fixed+direct. Fourier beats direct on both geometries by ~7% at long
horizons. Cone dominates fixed everywhere, with the gap exploding at
long horizons (1h: cone 36 vs fixed 162 = 4.5x less area; 2h: fixed
structurally cannot reach 90%).

Combined with C1 (fields explain ~0% of hard-tail difficulty at 2h),
the flagship-vs-conditioning tradeoff is now:
- Cone is a proven, model-side geometry improvement independent of any
  conditioning work. It is not gated on new data or new features.
- Weather-context conditioning is UNCERTAIN as a headroom source (C1);
  static Tier-3 geography remains the more promising conditioning axis
  but requires materialization v3 + new training runs.
- The scaling curve extrapolation says flagship (xlarge, 116M) at fixed
  geometry buys ~5% val CE beyond large. At cone geometry, the expected
  gain compounds: xlarge's capacity + cone's coverage/resolution regime.

RECOMMENDATION for flagship: switch geometry to CONE for the xlarge
flagship run, keeping Fourier head (parameter efficiency at G=192 +
continuous density needed for the harness/downstream). Recalibrate
num_freqs (18 -> 24?) to sharpen long-horizon lobes on the bigger cone
canvas. This combines the two verified wins (bigger encoder + better
geometry) rather than only one.

Alternatives on the table (PI decision):
A. Flagship AS CONFIGURED (fixed +-0.9°/192/F18, xlarge). Cleanest
   continuity with pre-registered plan. Leaves the cone advantage
   unclaimed at the biggest scale.
B. Flagship as CONE + xlarge (recommended). Combines proven wins.
   Requires copy-paste cone_r0/v/p into xlarge.yaml; harness already
   supports scoring.
C. Skip flagship, pivot to Tier-3 static-geography conditioning at
   medium scale. Highest information gain per compute if C1 signals
   generalize to Tier-3; requires materialization v3 rebuild.
LOOP STOPS HERE. Flagship launch is a PI decision and never autonomous.
## 2026-07-14 — Cone-grid study at small scale (pre-registered)

Requested: a direct cone-vs-fixed comparison at the small slot (the
comparison workhorse). Setup: scaling-small-cone-50M — identical to
scaling-small-50M except grid_mode=cone: origin-centred canvas with
R(t) = 0.05 + 1.5e-4 deg/s * ELAPSED SECONDS (light-cone reachable-set
bound: maneuvers are contained by construction; only sustained speed
outliers ~>32 kn escape). AMENDED same day: growth is per TIME
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

## 2026-07-14 — C1 headroom gate: fields do NOT explain hard-tail difficulty

Ran on 47,237 val windows at 2h horizon, 6 sampled days Sep-Dec 2024,
regressing DR-residual (km) on origin-sampled ERA5 wind + CMEMS Baltic
current + wave height:
  R^2 overall (wind+cur):     0.011  (1.1%)
  R^2 hard-decile only:       0.0013 (0.1%)
  + waves:                    0.0117 (no change)
  Pearson r, hard, wind:     -0.034
  Pearson r, hard, current:  -0.016
Decile means (0=easiest → 9=hardest): median residual 0.0 → 44.6 km, but
mean wind 5.6 → 5.2 m/s (DECREASES), mean current 0.09 → 0.12 m/s
(flat). Hard cases are in LOWER-wind, average-current conditions —
opposite of what a "weather causes hardness" story predicts.

Interpretation: hard forecasting is DECISION-driven (maneuvers, route
choice, port approaches), NOT weather-driven. Reconciles with C2's
r=0.52 for cross-heading drift vs currents (fields DO push vessels
physically, but drift is small vs maneuver-driven residuals at 2h).

Implication for flagship decision — the Tier-4 conditioning story
(weather/currents/waves recovers the ~1.25 kinematic floor) looks much
weaker than earlier estimates. Tier-3 STATIC geography (lanes, ports,
land, TSS) is the stronger conditioning direction because it captures
DECISION structure — where routes are, where vessels turn — which is
what makes hard cases hard. The flagship's ~5% CE headroom via pure
capacity remains a real option and is not automatically dominated by
conditioning if conditioning turns out to be weather-dominated.

Caveats: local weather at origin only (not along trajectory or
forecast-integrated); linear R^2 misses nonlinearities; 47k windows in
6 autumn/winter days (not a storm sample); test doesn't rule out fields
being useful in a more sophisticated integration or as a feature to
help stratify hard-vs-noisy cases — only that the simple version fails.

## 2026-07-14 — 2x2 data-vs-capacity RESOLVED (neutral window, both confounds controlled)

All three small checkpoints scored on the v1-val window (2024-09-22..12-08)
that NONE trained on. Both confounds now separated:
- small-69d  (69d,  b384):  CE 1.2695, 2.421x tuned-DR
- small-b384 (26mo, b384):  CE 1.2519, 2.461x tuned-DR
- small-26mo (26mo, b1638): CE 1.3033, 2.358x tuned-DR
BATCH effect (26mo, b384 vs b1638): b384 +3.9% (smaller batch / 4.3x more
updates helped). DATA effect (b384, 69d vs 26mo, the CLEAN cell): 26mo
+1.4%. Verdict: (1) the earlier "69d wins" reads were the BATCH confound
(69d used b384), not data — retracted for good; (2) data diversity pays
MODESTLY and in the diversity direction (26mo>69d by 1.4%), opposite to
the saturation hypothesis; (3) batch/optimization (3.9%) > data diversity
(1.4%) at this scale — consistent with capacity near the ~1.25 floor:
when the model is near-saturated, how you spend samples matters more than
which samples. Actionable: revisit batch size for the scaling runs (the
50M-sample family used large batches / few updates; smaller-batch may
recover a few % everywhere). Data-axis claims must stay scale-qualified
and modest.

## 2026-07-14 — Head x Geometry matrix, judged on containment km^2 (h1 retired)

Old head ablation judged h1 (a ~2s forecast) on raw CE — wrong question,
wrong metric. Replaced by a 2x2 {geometry: fixed, cone} x {head: fourier,
direct} at small/26mo/50M/F12, judged on the CONTAINMENT metric in
PHYSICAL km^2-to-capture-90% per time bucket (15m/30m/1h/2h) + ceilings
reported separately. Raw CE / cell-counts are NOT cross-geometry
comparable (a cone cell @2h is ~17x a fixed cell; fixed clips ~20% of 2h
targets). New scorer: eval/xgeometry.py -> km^2@90, ceiling, ranks on one
physical basis; driver scripts/xgeom_matrix.py. Cells: fixed+fourier
(scaling-small-50M, done) and cone+fourier (queued) exist; the two
direct-head cells train in head_geom_matrix_queue.sh after the cone study,
then all four are scored.

Validated baseline (fixed+fourier, val, 2-batch smoke): 15m ceiling 0.999
km2@90=5.4; 30m 0.994 km2@90=18; 1h 0.922 km2@90=266; 2h ceiling 0.797
km2@90=UNREACHABLE (20% escape the +-0.3deg grid). The 2h-unreachable
cell IS the cone's reason to exist: cone ceiling @2h should be ~0.99 (it
contains 37kn straight-liners by construction), so it can reach 90% where
fixed cannot — at the cost of coarser cells (bigger km2 per cell). The
matrix answers, in one table: does the cone restore long-horizon
containment (ceiling), at what km^2 cost, and does the head choice change
under the cone (fourier's F12 band-limit blur at long horizons may be
rescued because the band limit is relative to the growing canvas).
Success = cone reaches 90% at 1h/2h where fixed can't, without inflating
km^2@90 at short horizons; the direct head is the control for whether the
continuous Fourier density costs long-horizon sharpness.

## 2026-07-14 — Rotation + anisotropy = the cone's phase 2 (translation/rotation question)

Motion forecasting canonicalizes per-query by translation AND rotation
(agent-centric, +12-24%). TrackFM already has translation (egocentric
displacement; DR-centered rejected earlier) and scale (the cone). The
missing axis is ROTATION — align the box to the origin course so along-/
cross-track become separate axes, which unlocks ANISOTROPY (R_along >
R_cross). Measured on v1 val (moving vessels, SOG>1kn): along-track p99 /
cross-track p99 = 3.8x @<15min, ~2x @2-4.5h — a course-aligned box can be
44-74% narrower cross-track, ~HALF the canvas area at every horizon. This
(a) directly attacks the review's #2 risk (isotropic shape collapse),
(b) dissolves most of the containment-box resolution cost (the wasted
canvas is almost all cross-track), (c) makes long-horizon boxes actually
efficient — the reachable set of a moving vessel IS a thin course-aligned
ellipse, not a disc. Caveat: COG is undefined at SOG~0 (stopped) — rotate
only above a speed threshold, fall back to axis-aligned; stopped vessels
have tiny boxes anyway. COG must stay a conditioning input (it is) so
absolute direction is not hidden. CORRECTION (2026-07-14): rotation is NOT a standalone phase 2 —
it BREAKS geography. A course-aligned frame makes "where is land / where
do lanes run" heading-dependent: the same coast maps to a different
canvas location per vessel, so the model must learn geography in absolute
terms and un-rotate it per query via COG. Cost is worst at LONG horizons
(where geography, not kinematics, sets position) — exactly the target
regime; benefit (anisotropy) is largest at SHORT horizons where geography
is irrelevant. Cost/benefit is backwards. Motion forecasting (HiVT/QCNet)
gets away with rotation ONLY because it rotates the MAP into the same
frame; TrackFM has no explicit geography input yet, so rotating the
output alone is the worst case. Scale (the cone) is geography-safe by
contrast: isotropic zoom, north stays north; only a landmark's canvas
RADIUS shifts with horizon (1D, recoverable), not its direction.
DECISION: cone stays SCALE-ONLY. Rotation+anisotropy is GATED on
co-rotated Tier-3 context grids (bathymetry/lanes/land-mask cropped and
rotated with the output) — revisit only in that package, never before.

## 2026-07-14 — Cone R(t): LINEAR containment bound (containment requirement overrides resolution fit)

CORRECTION to the concave recalibration earlier today. Requirement: "capture ALL
movement as a forecast at any reasonable horizon." A constant-speed
straight-line vessel has displacement v_ship*t (LINEAR); a concave box
(p<1) eventually falls beneath any straight line, so the fast straight-
liner ESCAPES at long horizons — the concave fit optimized average
resolution efficiency, the WRONG objective. Containment requires a linear
(reachable-set) box. Measured effective-speed distribution (v1 val):
p99=20.7kn, p99.9=37.0kn, p99.99=140kn, max=12500kn (GPS teleports).
Real movement tops out ~37kn; above that is sensor glitch, excluded by
design (physics-bound censor, not sized into the box). Chosen:
R(t) = 0.02 + 1.71e-4 * t  (LINEAR, cone_p=1; 1.71e-4 deg/s = 37kn =
p99.9 speed). Verified: 20/30/37kn straight-liners contained at every
horizon; off-canvas 0.00% beyond 30min, <1% short-horizon (glitch tail).
Unit test pins the straight-line-containment guarantee. TRADEOFF (logged
honestly): the linear box is larger at long horizons than the fixed grid
(R(2h)=1.25deg vs fixed 0.3), so coarser resolution at fixed num_freqs —
num_freqs is the lever to buy it back (same coverage-vs-resolution knob as
the flagship geometry). This is the honest coverage/resolution trade the
cone-vs-fixed study measures: cone = full containment/coarser lobes vs
fixed = fine lobes/25% escape at 2h. The power-law code (cone_p) is
retained as a generalization but defaults to 1.0.

## 2026-07-14 — Cone R(t) recalibrated CONCAVE from measured envelope + review verdict

Verified mini-review (docs/research/2026-07-scale-normalized-outputs.md,
35 works) on scale-normalized outputs answered the "changing-scale map
would confuse the model" worry: that failure (over-stationarization,
Non-stationary Transformers 2022 [CONFIRMED]) occurs ONLY when scale is
HIDDEN from the network; TrackFM feeds horizon t to the conditioning MLP,
so it is excluded by construction. Domain-matched precedent: a lead-time-
conditioned TC-track net showed no skill loss vs horizon-specialized nets
[CONFIRMED]; per-query canonicalization HELPS 12-24% in motion forecasting
(HiVT/QCNet [CONFIRMED]). The review relocated the real risk to R(t)
CALIBRATION (dominant ~2x failure across diffusion scaling papers).

Pre-flight CPU check (120k val windows, before the cone run): displacement-
from-origin vs elapsed time is CONCAVE, p99 ~ 1.24e-3 * t^0.67 — sub-linear
because vessels maneuver/stop (opposite of the NHC convex cone, which
tracks forecast ERROR not reachable displacement). The pre-registered
LINEAR R(t) was therefore over-scaled at long horizons (fill ratio
0.72->0.34: ~89% of canvas area holds ~1% of mass) — the review's #1
false-negative risk, caught before spending GPU. RECALIBRATED to
R(t) = 0.015 + 1.55e-3 * t^0.67 (p99 * 1.25 margin), holding fill ratio
~0.77 across all horizons = true scale-stationarity. cone_p=1 still
recovers the linear bound. Legitimate pre-registration (calibration data,
not model-outcome data). Full suite 86 green.

OBJECTIVE CLARIFIED (2026-07-14): the cone does NOT need to BEAT
the fixed grid. Its job is to ENABLE very-long-horizon forecasts (6-8h)
WITHOUT hurting quality at the horizons we already do. The alternative
way to reach 6-8h — one FIXED box sized for 8h (~±2.8deg) used for every
forecast including 15min — is the real waste the cone avoids: it would
be huge and coarse for short forecasts. So the comparison that matters is
cone vs FIXED-HUGE (cone wins by construction: small sharp box short, big
box long), not cone vs the current fixed-small grid (which cannot reach
6-8h at all). SUCCESS CRITERION: (a) parity with fixed-small at the
overlapping horizons (<=~2h) on shared-window CE/containment, and (b)
graceful, contained forecasts out to the data's horizon limit. A cone
that merely MATCHES fixed at short range and extends cleanly to long
range is a SUCCESS, not a tie. DATA LIMIT: current windows reach only
~1.3-4.5h (800 steps); TRUE 6-8h needs longer horizon targets
(materialization v4, time-targeted windows) — the small-scale study
tests the mechanism up to ~4.5h; 6-8h is gated on the data rebuild.

Falsifiable expectation for the study (from the review, unchanged by
recalibration): cone MATCHES/BEATS fixed on long-horizon calibrated
NLL/miss-rate with the gap WIDENING as horizon grows, roughly NEUTRAL at
short horizons. Falsified if fixed >= cone at the LONGEST horizons, or if
the cone edge does not grow with t. Guardrail: do NOT justify the cone on
short-horizon accuracy (every well-conditioned control says neutral-to-
slightly-negative there — a short-horizon win would be motivated
reasoning). Diagnostics to log in the harness: off-canvas target mass vs t
(under-scaling indicator) and density anisotropy (rho->0 shape-collapse,
the DISPUTED TC-track risk). Live follow-up levers if the calibrated cone
still loses long: anisotropic R (along/cross-track), per-horizon lambda(t)
weighting, beta-NLL beta~0.5.

## 2026-07-14 — Cross-geometry eval harness (flagship vs family), incl. the clamp bias

Family (±0.3/64) and flagship (±0.9/192) share CELL PITCH (0.009375°),
so rank/containment metrics compare in physical units. Raw CE does NOT
compare: bigger-grid normalization, and — a caught confound — the CLAMP BIAS:
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

**2x2 data-vs-capacity — CONTAMINATION CAUGHT, one cell retracted:**
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
this is pure density-dependent chaining. Selection criterion (the
cross-referencing methodology turned into a validation set): eps must
recover the 22 independently-verified major commercial ports. Sweep:
0.75 km -> 22/22 recovered (970 clusters, 456 ports); 1.0 -> 21/22;
1.5 -> 20/22; 3.0 -> catastrophic. Chose 0.75. Residual fragmentation is
terminal-level granularity (benign for origin/destination labels);
chaining is wrong-label (harmful). Note: eps is now a DENSITY-DEPENDENT
parameter — revalidate against the known-port set if the dwell corpus
changes materially. v2 table: 970 clusters = 456 ports / 514 anchorages;
Copenhagen = "Lynettehavnen" (nearest registry basin), 82,317 dwells.

## 2026-07-13 — FLOPs accounting for the scaling ladder (the FLOPs challenge)

Challenge: is medium's win over small is just MORE FLOPS. Analytic
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

## 2026-07-12 — Flagship GATED on scaling evidence (PI decision)

Prediction: small@26mo ~= medium@26mo — i.e. capacity is NOT the binding
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

## 2026-07-12 — Data-vs-capacity at the small end (the FLOPs challenge)

Claim "data volume is doing enormous work" from small@26mo ~= golden-
medium@69d was CONFOUNDED (varied model and data together). Decisive
cell launched: **small@69d, identical 50M-sample budget** (4.5 repeated
epochs of golden69 vs fresh 26-month windows; same model, protocol,
sample count — only data DIVERSITY differs). Combined with the running
medium@26mo vs golden-medium@69d pair this gives a 2x2 mini scaling grid
{small,medium} x {69d,26mo}.

Outcome interpretation:
- small@69d ~= small@26mo -> prediction right: Small saturates; capacity binds
  at the small end; the 26-month corpus pays only at scale (watch the
  medium/large/xlarge points for where diversity starts mattering).
  Data-scaling claims must then be made per-scale, never pooled.
- small@26mo clearly better (>~0.05 val CE or >0.15x tuned-DR ratio) ->
  data diversity pays even at 1M params; repeated-epoch training on
  small corpora underperforms fresh data at matched sample counts.
- Intermediate -> report the interaction; the 2x2 grid becomes a paper
  figure either way.

## 2026-07-12 — Saturation stop v2: median-block halves (noise-robust)

Iterated twice under review challenges: (1) a rate-floor stop kills
healthy power-law runs (relative gains decay below any floor); (2) a
window-local slope projection has SNR < 1 under realistic +-1% val noise
(measured 26 consecutive false positives). Final criterion: compare
MEDIANS of the two halves of the full validation history — half-over-half
gain < 0.4% for 4 consecutive validations, minimum 24h history. Median
blocks grow with the run so noise (1/sqrt n) cannot fake saturation;
healthy power laws show ~1.5% half-over-half. Seeded sims: 10/10 noisy
power laws survive full budget, 10/10 flat runs stop within ~24h of
eligibility. Doctrine: loss selects, medians thrift, ranking monitors.

## 2026-07-12 — Campaign plan: hybrid geometries (chosen plan)

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
