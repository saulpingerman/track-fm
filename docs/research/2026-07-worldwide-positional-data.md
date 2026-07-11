# Worldwide multi-domain positional datasets (July 2026)

**Status: someday** — forward-looking research for a future multi-domain
"world model of movement" (ships + aircraft + vehicles + humans + animals
under one (position, kinematics, dt) abstraction). We are NOT building this
soon; this is a verified planning catalog. All claims below survived 3-vote
adversarial verification against primary sources (9 verified, 3 refuted).

## Bottom line

An academic lab can assemble **>20B position reports across four domains
today, with no commercial license** — but *worldwide coverage in any single
domain is out of reach* without paying. Every open source is regionally
biased. Animal tracking was not resolved by this review (see gaps).

## Per-domain inventory (verified)

### Maritime AIS
| Source | Scale / coverage | Access | Notes |
|---|---|---|---|
| **NOAA MarineCadastre** (US) | 2009–2025, ~17y; 2024 = 366 daily CSVs / ~117GB | **Open, CC0, no registration** | US-coastal only (~40–50nm, ~200 USCG stations). 2025 switched to zstd; 2009–14 are monthly per-UTM-zone |
| **Kystverket** (Norway) | Norwegian EEZ + Svalbard/Jan Mayen | Open real-time stream, NLOD, no registration (TCP 153.44.253.27:5631) | Open feed EXCLUDES fishing <15m & recreational <45m; full/historical needs an application |
| **Our Danish DMA** | 26mo, ~8B points | Open | (current corpus) |
| Global Fishing Watch | derived only: gridded effort 2012–24 (~370M fishing-hrs, ~141K MMSIs), events API | Non-commercial; **raw AIS NOT available** (licensed from Orbcomm/Spire) | weak labels, joinable by MMSI |
| AISHub | ~1,517 stations, 80 countries, ~107K vessels/24h, real-time only | **Reciprocal**: must run a receiver (≥10 vessels, ≥90% uptime) | no bulk history; upper bound on volunteer terrestrial federation |

**Federation verdict**: US + Norway + Denmark confirmed open → strong
*regional* coverage, NOT global. Other countries (Brazil, Finland, NL…)
unverified. Satellite-AIS gaps (open ocean) remain.

### Aviation ADS-B — OpenSky Network (the research standard)
- Scale: since 2013, **>7,000 receivers, >40 trillion Mode S replies**;
  preserves raw unfiltered replies (unlike FR24/FlightAware).
- **Open, no application**: weekly "24h of state vectors" snapshots,
  Mondays **Jun 2017–Jun 2022**, exactly **10s dt**, fields
  time/icao24/lat/lon/velocity/heading/vertrate/callsign/altitude, in
  CSV/Avro/JSON (~5–6K aircraft/tick). Non-commercial, citation required.
  **This matches our (position, kinematics, dt) abstraction directly.**
  Plus the COVID-19 Flight Dataset on Zenodo (2019-01 → 2022-12).
- **Full archive via application**: Trino DB, university/gov/aviation-authority
  researchers; `state_vectors_data4` has unlimited retention (~10s), raw
  tables ~1y. NO offline full-snapshot download exists (refuted).
- Limit: crowdsourced coverage saturates in Europe/US; oceans, Africa,
  global South are gaps (2025 report adds satellite ADS-C to fill them).

### Ground vehicles — WorldTrace (the standout)
- **2.45M trajectories, ~8.8B raw GPS points (~0.88B after 1s-normalization),
  70 countries, Aug 2021–Dec 2023.** Built from OpenStreetMap public GPS
  traces (openpilot/dragonpilot/sunnypilot car uploads, orig. 10Hz).
- **Openly downloadable, ODbL**: huggingface.co/datasets/OpenTrace/WorldTrace
  (full release confirmed — earlier "sample-only" claim refuted).
- Released with **UniTraj** (NeurIPS 2025) — the closest existing attempt at
  a worldwide trajectory foundation model. Raw point count is comparable to
  our 8B AIS; processed corpus ~10× smaller.

### Human mobility — YJMob100K (largest fully-open, but coarsened)
- 100K users / 75 days (Dataset 1: 111.5M records) + 25K users (Dataset 2:
  29.4M, incl. 15 emergency days) = **~140M records** (~1–2% of our AIS).
- **CC-BY 4.0 on Zenodo, no registration** (10.5281/zenodo.10836269).
- **Privacy-coarsened**: 500m cells (200×200 grid), 30-min bins, no
  city/coords/speed/course. A fundamentally different regime — cell IDs +
  30-min dt, no kinematics — hard to mix with raw lat/lon streams.
- No verified near-worldwide human-mobility corpus exists (WorldMove's
  1,600-city claim failed verification).

### Animal tracking — UNRESOLVED
No claims about Movebank / Argos / GBIF survived or were submitted to
verification. Domain 5 is an open question, not a negative result.

## Heterogeneity challenges (the core research problem)
- **dt spans 3 orders of magnitude**: 1s (WorldTrace) → 10s (OpenSky) →
  30-min bins (YJMob100K). Our AIS is seconds-to-minutes.
- **Representation mismatch**: raw lat/lon + kinematics (AIS, ADS-B,
  WorldTrace) vs. 500m cell IDs with no kinematics (YJMob100K).
- **Geographic bias in every domain**: US/Nordic maritime, EU/US aviation,
  openpilot-user countries for vehicles.
- **Licensing**: several "open" sets are non-commercial/citation, not
  public-domain — matters if TrackFM successors are ever commercialized.

## Open questions to chase later
1. Animal tracking: Movebank aggregate scale + per-study licensing — can
   enough CC-licensed studies aggregate into a usable telemetry corpus?
2. Which other countries publish open historical AIS, and does the
   federation cover enough shipping lanes vs satellite gaps?
3. What did UniTraj/TrajFM/GTM actually find on cross-domain transfer
   (vehicle→vessel, mixing 1s GPS with 30-min bins)? (Not yet researched.)
4. OpenSky Trino practical throughput/quota for multi-year extraction.

## Immediately actionable if we wanted a v2 corpus
Start with the three that are open + kinematic + our abstraction:
**OpenSky 10s snapshots** (aviation), **WorldTrace** (vehicles, HF download),
and a **NOAA MarineCadastre** federation with our Danish data (maritime).
That alone is a multi-billion-point, three-domain corpus with zero licensing
friction — enough to test whether cross-domain pretraining helps vessel
tasks before investing in the harder domains.
