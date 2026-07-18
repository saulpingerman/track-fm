# Fine-tuning TrackFM: strategy and protocol

## 1. Verdict on the staged-unfreezing hypothesis

**The mechanism is real, but narrower than hypothesized, and the remedy is "cheap insurance," not an expected win on standard evals.**

- The hypothesis is exactly LP-FT (Kumar et al.). Confirmed numbers: full FT beats linear probing ~2% ID but loses ~7% OOD; LP-FT beats full FT ~1% ID and ~10% OOD [C1]. A ~1-point ID delta is **within 30-trial Ax noise** — LP-FT only pays visibly if eval splits contain genuine shift (held-out regions, seasons, vessel classes) [C1].
- Feature distortion requires **three conditions simultaneously**: random-init head, *good* pretrained features, *large* distribution shift. When features are weak or shift is small, full FT wins even OOD [C2]. Prediction: the LP-FT advantage grows up the 4.5M→117M ladder and vanishes on same-distribution Danish AIS evals [C2].
- The *mechanism* of protection is not "avoiding large early gradients": NTK analysis shows it comes from the **large norm of the CE-converged head** suppressing feature change. Consequences: (a) small-std head init and LP-to-convergence are *different* interventions, not variants of one fix; (b) the mechanism is **CE-specific**, so the log1p-MSE EtaRegressor may not enjoy the protection; (c) LP-FT degrades calibration — temperature-scale port probabilities post-hoc [C3].
- The **strongest form** of the hypothesis ("head to convergence on frozen features suffices") has a documented failure mode: backbone outputs shift substantially once unfrozen, so the LP-stage head fits features that immediately move; PET-warmed heads (LoRA warmup) beat random-init heads across 9 GLUE/SuperGLUE tasks [C5].

**Default ruling:** LP-FT is the default for **port classification** (CE, where the mechanism applies), run head-to-convergence in the LP stage [C3], with a genuinely OOD eval split added so the expected benefit is measurable [C1]. For **ETA regression**, treat LP-FT as a falsifiable arm, not the default — expect weaker/absent benefit [C3]. For **anomaly detection**, staging is moot: try zero-shot first [C12]. Since FT runs are minutes-to-hours, the LP stage costs almost nothing — run it, but do not claim it as an expected ID improvement.

## 2. Strategy decision table

| Task | ~10k labels | ~100k–1M labels | Evidence chain |
|---|---|---|---|
| **Port classification** | **LP-FT** (LP to convergence under CE, then unfreeze). Small data = closest to the regime where distortion/instability matter. Temperature-scale afterwards. | **LP-FT default, full FT as primary rival** — the FT-vs-LP gap shrinks as labels grow [C1 note], so expect near-parity ID; LP-FT is insurance for OOD deployment. Add a **LoRA arm** (all matrices incl. MLPs, ~10x LR) as the one-phase structural rival [C14], and a **LoRA-warmed-head arm** [C5]. | C1 → C2 (moderators) → C3 (CE mechanism) → C5, C14 (rival arms) |
| **ETA regression** | **Linear probe first** (cheapest baseline), then full FT. LP-FT arm included but expected ≈ full FT. | **Full FT default.** The CE-norm protection mechanism doesn't obviously apply to log1p-MSE [C3]; no evidence staging helps regression. Run one LP-FT arm as the per-task asymmetry test — if LP-FT helps port but not ETA, C3's mechanism is confirmed locally. | C3 (CE-specificity) → C1 (data-scale hedge) |
| **Anomaly detection** | **Zero-shot density-head NLL** — no labels needed. Raw global NLL thresholding is known to fail (density is location-dependent: busy lanes vs open water); use **per-H3-cell a contrario normalization** with the existing hex machinery [C12]. | Same. Only revisit exp-12-style fine-tuning if normalized zero-shot underperforms the exp-12 result. Bonus: check whether a **port-tuned LoRA checkpoint retains pretrained density NLL** (LoRA forgets ~23% less; zero-init B starts at the exact pretrained function for the backbone+density path) [C14] — one checkpoint could serve two tasks. | C12 (GeoTrackNet precedent) → C14 (retention) |

The literature is genuinely split on whether LP-FT or PET-warmup is the better staging (C1/C2 vs C5, plus C3's refinement of the mechanism). Default pick: **plain LP-FT for CE tasks** because it's the exact hypothesis under test, is free at this compute scale, and has the strongest headline evidence; LoRA-warmup rides along as one arm, not the default.

## 3. Concrete hyperparameter protocol

- **FT LR:** search a band centered ~1/10 of the pretrain optimum (per-group after muP restoration, §4). Prefer **longer training at smaller LR** — that, not architectural tricks, is what moves means and cuts variance [C7]. **LoRA arm gets its own band ~10x the full-FT optimum** [C14].
- **Optimizer:** AdamW with bias correction — already satisfied (`finetune.py:168` uses `torch.optim.AdamW`, which always applies it); missing bias correction is the documented cause of degenerate runs [C7].
- **Schedule:** warmup + decay to zero; err toward more epochs with early stopping on val loss rather than fewer epochs at high LR [C7].
- **Head init:** standard init, and for the LP-FT arm **train the LP stage to convergence** — protection comes from the large converged head norm, not from starting small [C3]. Do **not** assume small-std init is an LP-FT substitute; if tested at all, it's a separate arm [C3]. LoRA B is zero-init by construction [C14].
- **Pooling:** **fix mean pooling** for pretrained encoders (local prior). Once the backbone adapts, mean vs last differs ≤0.014 F1 with task-dependent sign [C13] — but the bound is *not* established for frozen-backbone probes, where pooling gaps can be large [C13 note], so keep mean during the LP stage especially. Run pooling as a cheap 2-way ablation **outside Ax**.
- **Layer-wise LR decay:** not a separate condition — gains at BERT scale are within run-to-run noise. Fold in as **one Ax parameter, decay ∈ [0.9, 1.0]** [C7].
- **Early stop / epochs:** early-stop on val; budget generously since instability is fixed by bias correction + longer low-LR training, not by aggressive truncation [C7].
- **Calibration:** temperature-scale PortClassifier outputs post-hoc, especially after LP-FT [C3].

## 4. muP wiring implications

- **Restore the 3-group optimizer in finetune.py.** Full FT's optimal LR scales Θ(1/width) under the same SignSGD/Adam abstraction muP uses (muA, Thm A.23) — bypassing the grouped optimizer forfeits FT-LR transfer from 4.5M to 117M and multiplies Ax sweeps per width [C9].
- **MLPHead roles:** hidden layers (d→d/2→128) as *hidden* (η/m); final logits/regression layer as *readout*. Note the muA paper itself kept the classification-head LR fixed and unscaled, mirroring muP's separate readout treatment [C9] — consistent with the existing 3-group design.
- **LoRA A/B matrices need explicit muP role assignment** (the classifier already raises on unassigned params) [C14, C5 note].
- **Transfer is a hypothesis, not prior art:** no published muP work — Tensor Programs V included — validates HP transfer for fine-tuning pretrained checkpoints [C10]. The theoretical validation is at fixed width only [C9]. The 117M spot-check (§5) is a necessary falsification, not a formality.
- **What to sweep at d_base:** LR (per group), plus **weight decay and dropout per width** — regularizers do **not** µTransfer [C10]. Use **independent (LR-decoupled) weight decay**; evidence suggests it, not muP alignment, does cross-width stabilization after the early regime [C10]. The current dropout=0.3 must not be assumed optimal across the ladder.

## 5. What to ablate locally (ordered, each <2 GPU-h)

1. **Zero-shot anomaly, no training at all:** pretrained density-head NLL with (a) global threshold vs (b) per-H3-cell a contrario normalization, compared against exp-12's fine-tuned result [C12]. If (b) matches exp-12, fine-tuning for anomaly is dead. Pure inference — cheapest possible experiment.
2. **LP-FT vs one-shot full FT on port classification at 4.5M, with an OOD split** (held-out region or season). This is the direct hypothesis test with the moderator conditions actually present [C1, C2]. Record ID and OOD deltas separately.
3. **FT-LR transfer spot-check:** with muP groups restored, tune FT LR at 4.5M, then verify the optimum holds at the next width up (single LR sweep, few points) [C9, C10].
4. **Head-warmup bake-off on port classification:** LP-to-convergence vs LoRA-warmed head (then full FT) vs standard random-init full FT [C3, C5]. Piggyback the ETA asymmetry check (does LP-FT help ETA at all?) on the same budget if time allows [C3].

## 6. Skip list

- **Pooling as an Ax search dimension** — ≤0.014 F1 once the backbone adapts; fix mean per local prior, 2-way ablation outside Ax [C13].
- **LLRD as a separate experimental condition** — within noise at comparable scale; it's one cheap Ax parameter [C7].
- **Small-std / zero-init head as an assumed LP-FT substitute** — mechanistically distinct from LP-to-convergence; unsourced as a remedy [C3].
- **Fine-tuning for anomaly detection before zero-shot + per-cell normalization has been tried and failed** — GeoTrackNet precedent says the failure mode of raw NLL is normalization, not missing fine-tuning [C12].
- **LoRA as a compute-saving measure** — compute parity is moot when full FT takes minutes-to-hours; LoRA's only local justification is forgetting-retention / multi-task checkpointing [C14].
- **Learnable [CLS]-style pooling token** — catastrophic failure in the pooling ablation source [C13 note].
- **Per-width FT-LR sweeps as the default plan** — premature; run the transfer spot-check first and only fall back to per-width sweeps if it fails [C9, C10]. (Weight decay/dropout, by contrast, stay per-width [C10].)

## 7. Sources

- **C1, C2** — Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (LP-FT), ICLR 2022. https://arxiv.org/abs/2202.10054
- **C3** — Tomihari & Sato, NTK analysis of LP-FT (head-norm mechanism, calibration), NeurIPS 2024. https://arxiv.org/abs/2405.16747
- **C5** — Yang et al., *Parameter-Efficient Tuning Makes a Good Classification Head*, EMNLP 2022. https://arxiv.org/abs/2210.16771
- **C7** — Zhang et al., *Revisiting Few-sample BERT Fine-tuning* (bias correction, longer training, LLRD noise), 2020. https://arxiv.org/abs/2006.05987
- **C9** — Chen, Villar & Hayou, muA / Θ(1/width) full-FT LR scaling, 2026. https://arxiv.org/abs/2602.06204
- **C10** — *Weight Decay may matter more than muP for Learning Rate Transfer in Practice*, 2025. https://arxiv.org/abs/2510.19093
- **C12** — Nguyen et al., *GeoTrackNet* (unsupervised AIS anomaly detection, a contrario cell normalization), 2019. https://arxiv.org/abs/1912.00682
- **C13** — Pooling ablation for decoder-only classifiers, 2025. https://arxiv.org/abs/2512.12677
- **C14** — Biderman et al., *LoRA Learns Less and Forgets Less*, TMLR 2024. https://arxiv.org/abs/2405.09673
