# TrackFM Architecture Review — Final Ranked Recommendation

**Verdict of the judge panel (3/3 votes):** the verified evidence does not overturn the current design — it vindicates every core bet and prescribes targeted upgrades. This document merges the winning "best-evidence" proposal with all judge grafts.

---

## 1. EXECUTIVE RECOMMENDATION

| Decision | Recommendation | Confidence |
|---|---|---|
| **Backbone** | **KEEP** decoder-only causal transformer over bare posits; add continuous-time p-RoPE (75% bands, normalized within-window timestamps) alongside dt-as-feature; keep all learned/conditioning tokens OUT of the RoPE'd sequence; target 20–120M params, do not chase 1B. | **High** |
| **Head** | **KEEP** the discretized 2D density family on the horizon-scaled cone canvas, softmax-normalized; upgrade to single-pass multi-horizon anchor output (8–16 log-spaced cones, MLP projector per anchor) as a *challenger* gated on per-horizon coverage; settle Fourier vs per-cell vs implicit vs MDN in one matched bake-off. | **High** (family) / **Medium** (single-pass variant) |
| **Objective** | **KEEP** CE on the normalized cone density as the sole core loss; add multi-resolution coarse CE, exponential lead-time weighting, and a dense-future auxiliary; calibration handled **post-hoc** via split-conformal HDR, reported as a standing eval column on every arm. | **High** |
| **Horizon conditioning** | Layer three mechanisms: cone geometry (keep), single-pass anchor heads + interpolation (adopt, gated), horizon-as-cross-attention-query injected from outside the RoPE'd sequence (ablate). Train anchors through 8h from day one. | **Medium** |
| **Context integration** | **KEEP** canvas-registered rasters → zero-init CNN → additive logit bias; add dual-resolution (wide-coarse/narrow-fine) crops; prioritize traffic rasters; mandatory rotation augmentation, feature-dropout ablatability, and held-out-region gating. | **High** (mechanism) / **Medium** (dual-crop gain) |

---

## 2. EVIDENCE CHAIN PER DECISION

### 2.1 Backbone: decoder-only transformer, ~20–120M, continuous-time p-RoPE

- **C01** (*An Empirical Study of Mamba-based Language Models*, NVIDIA): pure Mamba/Mamba-2 lose copying and in-context recall (5-shot MMLU 28.0 vs 46.3 at 1.1T tokens; Phonebook failure past ~500 tokens); only ~7% attention hybrids recover parity — and the deficit is widest in the under-trained regime that 1–14 day runs occupy. At 128-posit context, attention is O(128²) ≈ free on a 96GB card, so the SSM efficiency case is void.
- **C03** (*RoMAE*): a 4.67M **completely vanilla** transformer whose only irregularity mechanism is RoPE over continuous wall-clock time beats ContiFormer, S5, mTAN, ATAT (ELAsTiCC F 0.803 vs 0.627). Prop 4.2: any learned token in the RoPE'd sequence destroys the relative-position property — conditioning must enter via the head or cross-attention. App B.1: RoPE degrades at position-domain edges → normalize timestamps to a fixed range, train anchors through 8h.
- **GraFITi** (AAAI 2024): the entire Neural-ODE/CDE/flow family is dominated by plain attention over (value, time-embedding) tokens — up to 17% better accuracy AND 5× faster runtime — on exactly the irregular-time problems it was built for. Its MAB-vs-GAT ablation shows graph machinery adds nothing; what matters is which tokens attend to what.
- **C19** (*Scaling Laws of Motion Forecasting and Planning*, Waymo): N_opt ∝ C^0.63; compute-optimal motion models are ~50× smaller than LLMs at equal compute — the existing 4.5M–117M sweep likely spans the single-GPU optimum; the ≤1B constraint is non-binding. Warning: overlapping sliding windows fake an irreducible-loss floor → **deduplicate the ~200M windows before any scaling-law fit**.
- **SMART** (β = −0.157 clean power law to 101M in ~1 week) and **TrAISformer** (same Danish waters, one 1080 Ti): precedent that decoder-only NTP on trajectory tokens scales cleanly at exactly this compute class.
- **Local facts**: 4.5M–117M sweep already exists; dt-as-feature works — keep it and ablate the p-RoPE combination rather than assuming replacement.

### 2.2 Head: cone-canvas density, single-pass multi-horizon, matched bake-off

- **C05** (*TrAISformer*, Table II, same Danish waters): CE-classified density 0.94 nmi vs MSE regression 8.36–10.64 nmi at 2h — ~10× — because regression averages bimodal posteriors at waypoints. Strongest domain evidence in the pack; it backs the existing head family decisively.
- **C08 + UnO**: implicit occupancy heads (balanced BCE) are intrinsically unnormalized and uncalibrated — UnO's own ablation shows the balancing that buys +8.7 mAP transfer destroys the probability scale. The Fourier head's free softmax normalization is a real, citable advantage. UnO's 0.06M decoder also shows head capacity can be tiny when the query mechanism is right — consistent with the local MLP-projector win (val CE 1.044 vs 1.262 over linear: the win is nonlinearity, not size).
- **C06**: no 2023–2026 paper compares grid vs MDN vs implicit heads on containment metrics — the matched-backbone bake-off is both the internal decision procedure and a publishable contribution. **Graft (judge 3): the MDN arm is restored to the bake-off** — omitting it forfeits the C06 gap claim.
- **C09** (*RainPro-8*): 36.7M params, 13h on one H100, all 48 lead times in one forward pass, beating a 227M MetNet-3 reimplementation — single-pass all-horizons is both ~K× cheaper and slightly better than lead-time conditioning. **Graft (judges 1, 2): adopted as challenger, not default** — RainPro-8 never audited calibration, so per-horizon containment/coverage and off-anchor interpolation quality are explicit acceptance criteria before it displaces per-horizon decoding.
- **ImplicitO**: K=1 learned offset lifts fast-mover mAP +10%, K=4 buys nothing; continuous query sampling beats grid sampling; implicit-in-t beats grid+interpolation at off-grid times — defines the third bake-off arm and the interpolation stress test.
- **C21** (*SKETCH*): intent bottleneck worth ~3× MFD at ~12h (1.86 → 0.63), near-oracle with a learned predictor — but wrong-waypoint hard conditioning is worse than none (2.09 vs 1.86) → **marginalize p(Z|X), never MAP**. **MTR++**: hard assignment (nearest anchor to GT) is what makes many-mode training stable; latent unanchored mixtures degrade with count; k-means anchors +5.5% mAP over latent embeddings. **C11** (*ModeSeq/EDA*): 6 sequential modes beat MTR++'s 64 anchors on mAP (0.4507 vs 0.4382); adaptive anchors cut miss rate 13.5% — dense static anchors underfit the rare-maneuver tail that drives km²@90. **Graft (judges 1, 2): run the intent pilot both ways** — K~64 hard-assigned data-driven waypoint anchors AND a few-adaptive/sequential-modes arm, head-to-head, rather than presupposing the C11 reading. **Graft (judges 1, 3): temperature-calibrate mixture anchor logits on val** — MTR++'s own top weakness is 45.5% top-mode hit rate.
- **Local facts**: cone R(t)=r0+v·t beats fixed grid on full-population containment; per-cell logit head trains ~2× faster at slightly worse quality — **graft (all 3 judges): the per-cell head becomes the standing workhorse for all ablation arms where head identity is not the variable under test; the Fourier head runs final rankings and confirmation.**

### 2.3 Objective: CE core + cheap additions + post-hoc conformal calibration

- CE validated three independent ways: **C05** (10× on this data), **C19** (CE tracks operational metrics monotonically across 84 models), **SMART** (plain CE NTP scales as a clean power law).
- Additions: multi-resolution coarse CE (TrAISformer's fine+coarse trick as hierarchical label smoothing on G=64); exponential lead-time weighting (RainPro-8 ablation: free ~1.6%); dense-future auxiliary next-posit loss (MTR++ Table XI: +1.5% mAP at near-zero cost). Skip RainPro-8's ordinal chained-conditional loss — its own ablation shows plain CE ties it.
- **Calibration is the highest-leverage move in the whole plan.** *Rethinking Gaussian Trajectory Predictors* (arXiv:2603.10407): NLL/CE-optimal heads are systematically overconfident at 2–3σ — exactly where km²@90 lives — so val-CE wins (including the local 1.044-vs-1.262 MLP-projector gap) must pass an HPD coverage audit before being credited. **C14** (*CONTRA*): split-conformal HDR delivers 0.89–0.91 empirical coverage at 90% nominal with ~24% smaller regions than ellipses, in minutes of holdout compute; nonconformity score = predicted mass at cells with density ≥ density at truth; threshold at the ⌈(1−α)(n+1)⌉ quantile, stratified by horizon bin and speed class; disconnected HDR components allowed (crucial at junctions — CONTRA's own connected flow regions would inflate area).
- **Graft (all 3 judges): conformal HDR is a standing eval layer, not a one-off** — calibrated km²@90 is reported for every ablation arm from the null gate onward, so all cross-arm rankings are on calibrated containment, not raw tails.
- Verify the CE↔km²@90 correlation explicitly once (Waymo never tested a containment metric). Only if post-hoc conformal is insufficient: add the KDE-softened CDF-matching term as a small auxiliary on HPD-mass ranks — never replacing CE (proper scoring drives sharpness).

### 2.4 Horizon conditioning: three layered mechanisms

1. **Geometric** — keep the cone canvas (local fact: beats fixed grid).
2. **Architectural** — single-pass 8–16 log-spaced anchor cones (C09), arbitrary wall-clock queries via coefficient/canvas-space interpolation; **acceptance criteria (graft)**: per-horizon coverage curves AND interpolation quality at off-anchor query times, because the shared trunk trading far-horizon calibration for near-horizon CE is a failure mode RainPro-8 never tested. Ablate an ImplicitO-style continuous-Δt query head as the alternative (implicit-in-t beats grid+interpolation off-grid).
3. **Attention-level** — inject the queried horizon as a cross-attention query into history tokens from **outside** the RoPE'd sequence: GraFITi's target-edge ablation (+9–20% MSE when the query leaves attention) says the horizon should participate in attention; RoMAE Prop 4.2 forbids doing it as a learned in-sequence token. The tension resolves to cross-attention/head injection only.
- Train the anchor set through 8h from the start (RoMAE App B.1 extrapolation degradation), even while 6–8h quality is aspirational.

### 2.5 Context integration: keep the in-flight design, harden it

- **HOME/GOHOME pattern** (canvas-registered rasters → zero-init CNN → additive logit bias): composes with Fourier, per-cell, and implicit heads alike — keep.
- **RainPro-8**: wide-coarse/narrow-fine dual crops; extra context sources carry skill specifically beyond ~4h (radar-only −9% CSI) — TrackFM's aspirational regime.
- **Local facts**: traffic rasters are the strongest measured signal; weather ≈ 0% — prioritize accordingly. Bound the traffic ceiling cheaply with Schöller's oracle-future-neighbors probe before building anything heavier.
- **C24 + C23 discipline**: scene-prior memorization masquerades as learned dynamics (ETH-Hotel ADE 1.59 → 0.30 only after rotation augmentation); the context pathway stays strictly ablatable via feature dropout (bare-posit core preserved), rotation/heading-canonicalization augmentation is mandatory, and all conclusions are gated on held-out-region eval (spatial split within Danish waters + ≥1 external AIS region). This is what protects the non-AIS transferability goal.

### 2.6 Budget discipline (graft, judges 2 & 3)

Loader throughput and window deduplication are **first-class budget items**: profile actual samples/s locally first (the 855 samples/s figure circulating in the panel is unsourced — measure, don't adopt), dedup the overlapping ~200M windows (Waymo's spurious irreducible-loss warning) before scheduling any multi-day run or scaling-law fit, and budget marginal GPU-hours to loader/dedup before any width increase.

---

## 3. ORDERED ABLATION PLAN (each step ≤1 day at 50M samples unless marked eval-only)

**Day 0 (hours, pre-plan):** Profile loader samples/s and quantify window overlap; dedup/decorrelate before any multi-day run or scaling fit. *(Graft — Waymo overlap warning; measured, not assumed, throughput.)*

**Day 1 — Null baselines + history truncation** *(eval-only + tiny runs)*: dead-reckoning from last SOG/COG with horizon-growing Gaussian/angular-wedge density on the cone, scored km²@90 per horizon; history 128 vs 8 vs 2 posits (C23). **Gate: no transformer result is credited unless it beats this at every horizon.** Conformal-calibrated km²@90 reported from this step onward (standing column).

**Day 2 — Calibration audit + conformal HDR on the current best checkpoint** *(near-zero compute)*: HPD coverage curves α 0.5–0.99 stratified by horizon/speed; split-conformal HDR-mass threshold (C14); one-time CE↔km²@90 correlation check (C19 caveat). Likely the best km²@90-per-FLOP step in the plan. The local MLP-projector CE win is only credited as a containment win if it survives this audit.

**Day 3 — dt encoding**: continuous-time p-RoPE (75% bands, normalized timestamps) vs dt-as-feature vs both; matched 20–30M backbone (C03). Cheapest backbone-side win available (~13% throughput cost is the only risk). *Workhorse: per-cell logit head.*

**Day 4 — Horizon head**: single-pass 8–16 log-spaced anchors + lead-time weighting vs current per-horizon decoding; lead-time weighting on/off (C09). **Acceptance criteria: per-horizon coverage curves AND off-anchor interpolation quality** *(graft)* — the challenger only displaces the incumbent if calibration holds at far horizons. Winning cuts eval cost for all later steps. *Workhorse: per-cell logit head.*

**Day 5 — Matched head bake-off** (fills the C06 gap): Fourier+MLP projector vs direct per-cell logits vs ImplicitO-style continuous-(x,y,Δt)-query head (K=1 learned offset, quadrature-normalized) **vs small MDN** *(graft — restored)*; identical backbone, all normalized over the cone; scored on CE + calibrated km²@90 + wall-clock cost. This is the one day the Fourier head and challengers all run in full.

**Day 6 — Objective add-ons**: multi-resolution coarse CE (TrAISformer) and horizon-as-cross-attention-query outside the RoPE'd sequence (GraFITi target-edge); each independently on/off. *Workhorse: per-cell logit head.*

**Day 7 — Context**: zero-init CNN logit bias on/off; single vs dual-resolution crops (RainPro-8); rotation augmentation on/off — **all evaluated on a held-out spatial region** (C24); oracle-future-traffic upper-bound probe (Schöller). *Workhorse: per-cell logit head.*

**Day 8 (+1 if needed) — Long-horizon intent mixture (4–8h arm)**: data-driven waypoint clusters from turning-point geometry (bare-posit compatible; no ports/straits annotations, no retrieval DB); p(Z|X)-**marginalized** mixture of per-intent density fields; **two arms head-to-head** *(graft)*: (a) K~64 hard-assigned k-means waypoint anchors (MTR++ recipe), (b) few adaptive/sequential mode tokens (C11); **temperature-calibrate anchor logits on val** *(graft, MTR++ 45.5% top-mode weakness)*; compare both against the plain far-horizon anchor head on calibrated km²@90 at 4–8h.

**Afterwards (opportunistic):** iso-FLOP scaling-law fit (Waymo protocol, ~7 compute bands, parabolic fits) over the deduplicated data across the existing 4.5M–117M sweep — confirms where compute-optimal sits before any model is scaled up.

---

## 4. SKIP LIST

| Rejected option | Evidence-based reason |
|---|---|
| **Mamba/SSM re-platform** | At 128-posit context the efficiency case is worth nothing; pure SSMs demonstrably lose in-context recall/copying — the capability a density head needs to condition on specific past posits — with the deficit widest in the under-trained 1–14-day regime (C01). At most one ~8%-attention hybrid arm if history ever grows 10–100×. |
| **Neural ODE/CDE/ContiFormer/flow continuous-time machinery** | Dominated by plain attention over (value, time-embedding) tokens on both accuracy (up to 17%) and runtime (5×) on exactly the problems it was designed for (GraFITi; corroborated by RoMAE beating ContiFormer/S5 with vanilla attention). |
| **Diffusion / CNF / sampled-rollout primary density head** | ~1000× head-cost per cone heatmap vs one forward pass (C12/GenCast: 39 NFE × 50 members); Waymo's inference-scaling crossover prices sampled coverage as strictly worse per FLOP at this budget (C19). Keep only as a fairly-priced baseline arm. |
| **MAE/bidirectional pretraining objective** | RoMAE's own authors concede MAE struggles at out-of-distribution positions and causal models are better for forward prediction — take the positional mechanism, not the objective (C03). |
| **Discrete motion-token output + autoregressive rollout (SMART/TrAISformer-style)** | Cannot emit a calibrated joint 2D density at arbitrary wall-clock horizons without expensive sampling; erases irregular dt; region-locked bins break transferability; 2h horizons imply thousands of compounding steps. TrAISformer stays as the baseline to beat, not the head. |
| **Normalizing-flow conformal regions (CONTRA's flow)** | Regions are provably connected — wrong for multimodal vessel futures at junctions where disconnected HDRs win; the flow barely beat PCP on the 2D taxi task. Take only the split-conformal HDR calibration math (C14). |
| **64 static k-means anchors as the *sole* multimodality mechanism** | Dense fixed anchors underfit rare maneuvers (C11: ModeSeq 0.4507 vs MTR++ 0.4382 mAP with 6 modes; EDA −13.5% MR with adaptive anchors), and the CE-trained density head has no WTA collapse to fix. Anchors appear only inside the Day-8 intent-mixture pilot — and per the graft, tested head-to-head against the adaptive-modes alternative rather than assumed either way. |
| **Scaling toward 1B params** | Compute-optimal motion models are ~50× smaller than LLMs at equal compute (C19, N_opt ∝ C^0.63); the 4.5M–117M sweep already spans the likely single-GPU optimum. Spend the compute on dedup, head ablations, and calibration instead. |
| **Ordinal chained-conditional multi-horizon losses; heavy scene encoders (HD-map polylines, symmetric multi-agent encoding)** | RainPro-8's own ablation shows plain CE ties the ordinal machinery; map/multi-agent encoders violate the bare-posit constraint — traffic conditioning stays on the raster/dropout path. |

---

## 5. RISKS & OPEN QUESTIONS

**What the literature could not answer — only local experiments can settle:**

1. **Grid vs MDN vs implicit heads on containment metrics** (C06): no published comparison exists. Day 5 is both the decision procedure and a paper. Risk: the answer may be horizon-dependent (implicit wins off-grid, Fourier wins calibrated tails), forcing a hybrid.
2. **CE ↔ km²@90 correlation**: Waymo validated CE against minADE/mAP, never against a containment/area metric; *Rethinking Gaussian* says CE-optimal tails are too tight. If the Day-2 audit shows CE and calibrated km²@90 decouple, every CE-based selection in the plan (including the local MLP-projector win) needs re-ranking on the operational metric.
3. **Single-pass multi-horizon calibration**: RainPro-8 never audited calibration; whether a shared trunk trades far-horizon coverage for near-horizon CE is unknown and is the explicit Day-4 acceptance test.
4. **p-RoPE × dt-as-feature interaction**: RoMAE tested continuous RoPE alone; no paper tests it combined with explicit dt features, or on 15min–8h wall-clock spans. Extrapolation beyond trained horizon ranges is documented to degrade (RoMAE App B.1) — the 6–8h arm may hit this even with anchors trained through 8h.
5. **Does the intent bottleneck survive full-population traffic?** SKETCH is container ships on structured sea lanes, deterministic output, semi-manual waypoints. Whether data-driven waypoint clustering carries the ~3× long-horizon gain over Danish full-population traffic (fishing, pleasure craft) — and whether few-adaptive-modes or many-hard-anchors wins — is genuinely open (the Day-8 dual-arm exists because the judges ruled the C11 reading contestable against MTR++'s 6→64 ablation).
6. **Mixture-weight calibration under marginalization**: MTR++'s 45.5% top-mode hit rate shows anchor-probability calibration is the known weakness of every mixture recipe; temperature calibration is planned but unproven at maritime horizons.
7. **Transformer value over dead reckoning at short horizons**: C23's constant-velocity upset (vessels are *more* inertial than pedestrians) means the 15–30min band may be nearly saturated by the null model — the Day-1 gate can genuinely fail arms, and that outcome would redirect compute toward long horizons.
8. **Traffic-raster ceiling**: locally the strongest signal, but its value as a *static* raster vs dynamic neighbor state is unmeasured; the Schöller oracle probe bounds it cheaply but the answer is unknown.
9. **Transferability**: held-out-region and external-AIS evaluation may reveal that part of current skill is memorized Danish-waters prior (C24) — the single largest threat to the non-AIS research goal, and only our own spatial-split experiments can quantify it.
10. **Dedup effect size**: how much of the current val-loss floor is window-overlap artifact (Waymo warning) is unknown until the Day-0 dedup lands; scaling-law conclusions wait on it.

---

## 6. SOURCES

- TrAISformer — https://arxiv.org/abs/2109.03958
- Scaling Laws of Motion Forecasting and Planning (Waymo) — https://arxiv.org/abs/2506.08228
- SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction — https://arxiv.org/abs/2405.15677
- GraFITi: Graphs for Forecasting Irregularly Sampled Time Series — https://arxiv.org/abs/2305.12932
- RoMAE: Rotary Masked Autoencoders are Versatile Learners — https://arxiv.org/abs/2505.20535
- An Empirical Study of Mamba-based Language Models (NVIDIA) — https://arxiv.org/abs/2406.07887
- ImplicitO: Implicit Occupancy Flow Fields for Perception and Prediction in Self-Driving — https://arxiv.org/abs/2308.01471
- MTR++: Multi-Agent Motion Prediction with Symmetric Scene Modeling and Guided Intention Querying — https://arxiv.org/abs/2306.17770
- ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling — https://arxiv.org/abs/2411.11911
- SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction — https://arxiv.org/abs/2601.18537
- Rethinking Gaussian Trajectory Predictors: Calibrated Uncertainty for Safe Planning — https://arxiv.org/abs/2603.10407
- CONTRA: Conformal Prediction Region via Normalizing Flow Transformation — https://arxiv.org/abs/2605.08561
- UnO: Unsupervised Occupancy Fields for Perception and Forecasting (CVPR 2024)
- RainPro-8: single-pass multi-lead-time precipitation forecasting (C09 evidence pack entry)
- GenCast (diffusion weather forecasting; C12 evidence pack entry)
- HOME / GOHOME: heatmap output for motion forecasting (context-integration pattern)
- Schöller et al.: What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction (C23 evidence pack entry)
- EDA: Evolving and Distinct Anchors (C11 evidence pack entry)
- CTLPE: Continuous-Time Learned Positional Embeddings (C03 corollary evidence)
