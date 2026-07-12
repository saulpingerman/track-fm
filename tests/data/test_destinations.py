"""Destination/ETA interpretation: registry matching and DMA year-fill quirks."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from trackfm.data.destinations import (DEFAULT_MIN_CONFIDENCE,
                                       normalize_destination, parse_eta)

REGISTRY = [
    {"name": "Aarhus Havn", "lat": 56.15, "lon": 10.22},
    {"name": "APM Container Terminal Aarhus", "lat": 56.153, "lon": 10.238},
    {"name": "Skagen Havn", "lat": 57.72, "lon": 10.59},
    {"name": "Helsingborg Hamn", "lat": 56.04, "lon": 12.69},
    {"name": "Thyborøn Lystbådehavn", "lat": 56.70, "lon": 8.21},
    {"name": "Onsevig Havn", "lat": 54.95, "lon": 11.06},
]


def _reg(tmp_path: Path) -> str:
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(REGISTRY))
    return str(p)


def test_exact_match_is_confidence_one(tmp_path):
    m = normalize_destination("AARHUS HAVN", registry_path=_reg(tmp_path))
    assert m.name == "Aarhus Havn" and m.confidence == 1.0 and m.method == "exact"


def test_city_token_matches_general_entry(tmp_path):
    m = normalize_destination("AARHUS", registry_path=_reg(tmp_path))
    assert m.name == "Aarhus Havn"          # shorter name wins the tie
    assert m.confidence >= DEFAULT_MIN_CONFIDENCE


def test_junk_scores_below_threshold(tmp_path):
    m = normalize_destination("TOW.SRV.EQPT.1NM CPA", registry_path=_reg(tmp_path))
    assert m.confidence < DEFAULT_MIN_CONFIDENCE


def test_from_to_convention_uses_segment_after_gt(tmp_path):
    m = normalize_destination("GDANSK>SKAGEN", registry_path=_reg(tmp_path))
    assert m.name == "Skagen Havn" and m.confidence >= DEFAULT_MIN_CONFIDENCE


def test_known_locode_resolves_then_matches(tmp_path):
    m = normalize_destination("DKSKA", registry_path=_reg(tmp_path))
    assert m.method == "locode" and m.name == "Skagen Havn"
    assert m.confidence >= DEFAULT_MIN_CONFIDENCE


def test_unknown_locode_surfaced_not_fuzzed(tmp_path):
    m = normalize_destination("DKXQZ", registry_path=_reg(tmp_path))
    assert m.method == "locode-unresolved" and m.confidence == 0.0


def test_real_city_name_not_mistaken_for_locode(tmp_path):
    # YSTAD is 5 letters but starts with no known country code.
    m = normalize_destination("YSTAD", registry_path=_reg(tmp_path))
    assert m.method == "fuzzy"


def test_nordic_transliteration(tmp_path):
    # AIS senders write Thyborøn as THYBORON; both sides fold to ASCII.
    m = normalize_destination("THYBORON", registry_path=_reg(tmp_path))
    assert m.name == "Thyborøn Lystbådehavn"
    assert m.confidence >= DEFAULT_MIN_CONFIDENCE


def test_near_miss_token_coverage(tmp_path):
    # ONSVIG (dropped E) should still cover ONSEVIG.
    m = normalize_destination("ONSVIG", registry_path=_reg(tmp_path))
    assert m.name == "Onsevig Havn"
    assert m.confidence >= DEFAULT_MIN_CONFIDENCE


def test_empty_destination(tmp_path):
    m = normalize_destination("", registry_path=_reg(tmp_path))
    assert m.method == "empty" and m.name is None


def test_eta_ok():
    p = parse_eta("05/01/2023 17:00:00", datetime(2023, 1, 1))
    assert p.flag == "ok" and p.hours_out == 113.0


def test_eta_year_roll_back():
    # December ETA reported Jan 1 with DMA's report-year fill -> last year.
    p = parse_eta("13/12/2023 22:00:00", datetime(2023, 1, 1))
    assert p.flag in ("year-rolled", "stale") and p.eta.year == 2022


def test_eta_year_roll_forward():
    p = parse_eta("02/01/2023 04:00:00", datetime(2023, 12, 30))
    assert p.eta.year == 2024 and p.flag == "year-rolled"


def test_eta_stale_and_unparseable():
    assert parse_eta("01/01/2023 00:00:00", datetime(2023, 1, 20)).flag == "stale"
    assert parse_eta("garbage", datetime(2023, 1, 1)).flag == "unparseable"
    assert parse_eta(None, datetime(2023, 1, 1)).flag == "unparseable"
