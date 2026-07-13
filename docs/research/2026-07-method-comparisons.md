# Published-method comparison plan (July 2026)

**Status: active** — defines the paper's comparison section. Builds on the
verified findings in [2026-07-downstream-tasks.md](2026-07-downstream-tasks.md).

## The output-type problem (why comparison needs care)

Methods emit different objects; naive metric comparisons are apples/oranges:

| Method | Output type | Native eval |
|---|---|---|
| **TrackFM (ours)** | continuous 2D density (Fourier grid) per horizon, one-shot conditioned on elapsed time | soft-target KL / NLL |
| **TrAISformer** (CIA-Oceanix, code public) | discrete "four-hot" bins over lat/lon/SOG/COG, autoregressive; SAMPLES trajectories | best-of-N mean haversine vs truth |
| **GeoTrackNet** (CIA-Oceanix, code public) | VRNN probabilistic track representation | a-contrario anomaly detection |
| **AIS-LLM** (2025) | LLM decoder, multi-task text/numeric | per-task; code availability unverified |
| Seq2Seq/GRU/LSTM/Social-LSTM | point trajectory | ADE/FDE meters |
| DR / tuned-DR / Kalman (built) | point + Gaussian | our loss framework |

Two fair meeting grounds:
1. **Point-prediction metrics** (ADE/FDE km, per-horizon haversine): reduce
   every method to a point. Ours = density mode (or expectation) per horizon.
   TrAISformer's = best-of-N or mean of samples (report both; best-of-N
   flatters samplers — say so).
2. **Probabilistic metrics** (NLL of the true position; calibration
   coverage): ours natively; TrAISformer's factorized bin distribution IS a
   piecewise-constant density -> NLL directly comparable (bin-width
   correction required); point methods enter only via fitted Gaussians
   (our tuned-sigma machinery). Calibration curves (does the X% credible
   region cover truth X% of the time?) showcase the Fourier head's
   non-Gaussian value — no point method can win here, so lead with NLL +
   coverage, use ADE/FDE for breadth.

## Tiered execution plan

### Tier 1 — EnvShip-Bench protocol (LOW effort, HIGH credibility)
Run TrackFM under the released MIT pipeline (repo:
mark000071/EnvShip-Bench_Large_Dataset_Pipeline_and_datasets; DMA + NOAA
data) and report ADE/FDE against their published table (best ADE 59.57m
Seq2Seq; TrAISformer best FDE 125.41m, plus GRU/Bi-GRU/LSTM/Bi-LSTM/
Social-LSTM). Needs: density->point extraction util + protocol alignment
(their dt, input lengths). One table, zero training of others' models.

### Tier 2 — TrAISformer retrained on OUR split (MEDIUM effort, the marquee comparison)
Official repo targets DMA data already (paper region is a subset of ours).
Retrain on golden69 train split; evaluate BOTH models on golden69 test:
per-horizon haversine + NLL + calibration. This is the head-to-head that
answers "is the Fourier density better than discrete four-hot bins" on
identical data. Watch: their preprocessing (10-min resampling) differs from
our native cadence — run their protocol on their resampling, ours on ours,
and one bridging row (ours evaluated at their resampled horizons).

### Tier 3 — GeoTrackNet on the DTU anomaly task (MEDIUM, downstream chapter)
Learned anomaly baseline beside our fine-tuned encoder (AUPRC on the
verified 521/25 set) and the legacy exp-12 numbers. Their code is TF1-era —
budget porting pain; fall back to citing reported numbers if it fights us.

### Reported-numbers-only (no code)
AIS-LLM (trajectory + anomaly on DMA 2019): cite their table, note
non-identical split. The May-2025 survey's 34-paper table gives context
for method-family positioning.

## Code tasks this implies (small, before golden eval reuse)
1. `density_point_estimate(log_density)` — mode + expectation extraction.
2. Haversine per-horizon metric alongside the loss-based table in
   `evaluate_forecasting` (km errors are what external readers understand).
3. Calibration coverage curve from the Fourier density.
4. NLL-of-truth metric (log-density at true displacement, bin-corrected).

## Framing for the paper
Baseline ladder, weakest -> strongest, all on the same test split:
LP -> DR(fixed) -> DR(tuned per-horizon) -> Kalman(tuned) -> TrAISformer ->
TrackFM. Each rung answers one objection; the Kalman rung (added 2026-07-11)
kills the "strawman baseline" review, and the TrAISformer rung kills
"but learned methods." Report paper-comparable fixed-sigma ratios in an
appendix for continuity with the original Table 1.

## AISFormer full text read (2026-07-13, via PSU ILL; Yu et al., Ocean Eng 340:122098)

The paywalled closest-neighbor is now read. The single most important
finding: **AISFormer does NOT condition on wave fields as model input.**
x_t = [lat, lon, SOG, COG] throughout (their Eq. 1); the methodology
sections define only four-hot encoding + frequency-domain self-attention
(FFT over the SEQUENCE — spectral sequence modeling, unrelated to our
spatial Fourier density head) + a multi-resolution CE loss. The
"wave-aware" claim is a TRAINING CURRICULUM: pretrain on a
maneuver-rich/calm-sea subset, then fine-tune (lr 1e-5) on a high-wave
subset — with ERA5 waves used only to SELECT the subsets. And the
subsets are startlingly small: "1200 seconds (pretraining) and 15,000
seconds (retraining)" — 20 minutes and ~4.2 hours of data; "the complete
process takes approximately 20 minutes."

Consequences for us:
1. The fusion review's [CONFIRMED] caveat (">20% gain is vs TrAISformer,
   not a no-wave ablation") UNDERSOLD the issue — there is no wave input
   pathway to ablate. Their ablations cover four-hot/FSA/multi-res-loss
   only. **The maritime domain still has NO published environmental-field
   -conditioned trajectory forecaster; review gap #4 stands, stronger.**
2. Same data source as ours (DMA), ROI (55.5-58N, 10.3-13E) INSIDE our
   bbox, Jan-Mar 2019, cargo+tanker only, 10-min resampling, recursive
   sampling decode. A head-to-head is feasible on our splits.
3. Protocol cautions for any comparison: Table 1 (0.43/0.89/1.62 nmi at
   1/2/3h) reports the BEST OF 16 sampled trajectories; their own Table 2
   gives MEAN ADE@3h = 6.42 nmi — a 4x gap between their two protocols.
   Match protocol explicitly or report both; our capture/containment
   metrics are the honest generalization of best-of-N (capture@k asks
   "how much probability mass must I search," not "was one of 16 samples
   close").
4. Worth borrowing (cheap, testable): their multi-resolution CE
   (auxiliary coarse-grid loss, beta-weighted) — a plausible calibration
   /long-horizon aid for our grid CE; and their attention-map analysis
   (waypoint rows attend to prior waypoints) is a nice interpretability
   template.
5. Positioning: four-hot factorizes lat/lon into INDEPENDENT marginals
   per step — it cannot represent diagonal/correlated multimodality
   (e.g. two lanes crossing obliquely) without the correlation being
   carried implicitly by autoregressive sampling. Our joint 2D
   continuous density models exactly this; one paper paragraph.
