# Fine-tuning task & dataset review (July 2026)

Deep literature review of downstream tasks for single-track AIS encoders,
with verified datasets/recipes. All claims below survived 3-vote adversarial
verification against primary sources. Excludes tasks we already have
(forecasting, DTU anomaly, vessel type, port origin/destination, ETA).

## How the field evaluates (context for the paper)

- **Trajectory forecasting dominates**: 22 of ~34 transformer-AIS papers
  (survey, May 2025: arxiv.org/html/2505.07374v1) evaluate forecasting;
  behavior *detection* (8 papers) and behavior *prediction* (4) are the
  standard companion categories. TrackFM's suite (forecasting pretraining +
  behavior-style downstream tasks) matches peer expectations.
- **Peer models benchmark on OUR data source**:
  - **TrAISformer** (arxiv.org/abs/2109.03958): forecasting-only, DMA data
    Jan–Mar 2019, 55.5–58.0N/10.3–13.0E — a strict subset of our bbox.
  - **AIS-LLM** (Aug 2025, arxiv.org/html/2508.07668v1): Qwen2-1.5B+QLoRA
    doing forecasting + anomaly + collision-risk on the same DMA region
    (8,462 train / 1,397 test trajectories). Explicitly frames multi-vessel
    interaction as the field's open gap (supports deferring collision).
  - **EnvShip-Bench** (June 2026, arxiv.org/html/2606.15240v1): standardized
    short-term forecasting benchmark from DMA + MarineCadastre, MIT-licensed
    pipeline, published baselines (best ADE 59.57m Seq2Seq; TrAISformer best
    FDE 125.41m). Lets us score TrackFM against peers without new labels.

## Recommended additions (ranked)

### 1. Intentional AIS shutdown / dark-vessel gap detection  [self-supervised, HIGH]
Published recipe (Bernabé et al., IEEE T-ITS 25(2) 2024, arxiv 2310.15586;
transformer hit 99.65% acc): from a window of ~25 messages, predict whether
another message arrives within ~10 min — labels generated automatically from
the stream itself. Features: [t, lat, lon, SOG, dt, distance deltas,
distance-to-port, time-of-day]. Filters: ≥50 msgs/trajectory, loss >5km from
port. Complementary definition (Rodriguez et al., EPJ Data Sci 2024, arxiv
2211.04438): "silence anomalies" = gaps >24h outside ports (87% within 100km
of coast). Their corpora are private → self-generation is necessary AND
sufficient. **Fits us perfectly: we already detect gaps (segmentation) and
have the port index from the port-task work.** Effort: LOW.

### 2. Fishing-activity detection (per-position fishing / not-fishing)  [public labels, HIGH]
GFW training data: github.com/GlobalFishingWatch/training-data (CC-BY 4.0,
numpy, git-lfs) — expert point-labeled single-vessel tracks. Canonical
precedent: Kroodsma et al. 2018 (Science, 10.1126/science.aao5646): CNNs on
single tracks, >90% fishing detection, trained on ~500–1000 labeled tracks →
thousand-track supervision suffices. Sequence-labeling head (per-position
logits) — a nice architectural variety beyond pooled classification.
Effort: MEDIUM (new head type + label alignment).

### 3. Gear-type classification (6 classes)  [public labels, HIGH]
Same GFW repo: drifting longlines, purse seines, trawlers, fixed gear,
pole-and-line, trollers (+Unknown). Kroodsma CNNs: ~95% acc. Whole-track
classification — drop-in reuse of our existing PortClassifier head.
Effort: LOW (data prep only). Caveat: global tracks, not Danish-only —
fine-tune tests geographic transfer too (a selling point).

### 4. Loitering / encounter detection via GFW weak labels  [weak labels, MEDIUM]
GFW Events API (globalfishingwatch.org/our-apis/documentation): pre-computed
loitering, encounters (transshipment proxy), port visits, fishing events —
joinable to our corpus by MMSI. Also: gridded fishing effort by MMSI
(2012→96h ago; BigQuery global-fishing-watch.fishing_effort_v3) and Vessel
API registry gear/vessel types for Danish MMSIs (an independent check on our
ship_type field). Loitering is single-track. Encounters are two-vessel
events but detectABLE from one track's kinematics as "loitering with
company" — defer the interaction modeling, keep the label. Effort: MEDIUM
(API key, joins).

### 5. External forecasting benchmark (EnvShip-Bench)  [no labels needed, MEDIUM confidence]
Run TrackFM's forecaster on the published benchmark protocol; report
ADE/FDE against Seq2Seq/TrAISformer/LSTM baselines. Strengthens the paper's
comparison section with zero labeling work. Effort: LOW-MEDIUM (protocol
alignment).

## Explicitly unvetted (no verified benchmark found — usable but cite-free)
Spoofing detection, navigational-status prediction, draught/load-state
estimation, emissions, weather impact, trajectory imputation, route
extraction/clustering, destination free-text mining, SAR patterns, pilot
boarding, tug ops. Several are auto-labelable from raw fields we hold
(nav status, draught, destination text) — candidates for novel-task
contributions rather than benchmark comparisons. Re-research individually
before investing.

## Notes for implementation
- Raw DMA zips carry nav-status/draught/destination/ETA text that the
  cleaning pipeline currently drops; tasks needing them require a column
  addition + re-extraction pass (cheap: fields only, no re-cleaning).
- GFW data is global; restrict joins to our bbox/MMSIs. Fishing-effort
  labels are themselves CNN-derived (weak supervision, not ground truth) —
  frame accordingly in the paper.
