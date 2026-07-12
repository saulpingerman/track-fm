"""Shared, versioned interpretation of AIS free-text Destination and ETA.

Extraction (aux_fields.py) archives both fields verbatim — every rule that
*interprets* them lives here, in one place, so the port task and any
destination-conditioned experiment use the same mapping and re-running an
interpretation change costs seconds over the 150MB aux change-log, never a
re-read of the raw archives.

Destination matching scores a cleaned free-text string against the OSM
harbour registry (configs/data/port_registry_osm.json) and ALWAYS returns
the best candidate with a confidence in [0, 1]; callers threshold
(DEFAULT_MIN_CONFIDENCE) rather than this module deciding what counts as
matched. Foreign ports (outside the registry region) and junk strings
simply score low — that is signal, not failure.

ETA parsing handles the DMA quirk that AIS broadcasts ETA as month/day
(no year) and DMA fills in the *report* year: a December ETA reported in
January parses ~11 months in the future when it means last month. Off-by-
a-year candidates are rolled to the nearest year and flagged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

DEFAULT_MIN_CONFIDENCE = 0.55
INTERPRETATION_VERSION = 2   # bump on any rule change; store next to outputs

# AIS senders transliterate Nordic characters inconsistently (Thyborøn ->
# THYBORON / "THYBOR N"); fold both sides to plain ASCII before matching.
_TRANSLIT = str.maketrans({"Ø": "O", "Å": "A", "Æ": "A", "Ö": "O", "Ä": "A",
                           "Ü": "U", "É": "E"})

# UN/LOCODEs seen in regional traffic (calibrated on real aux data
# 2026-07-13). Resolution = map code -> city string -> normal fuzzy match,
# so out-of-registry cities (DEHAM) still score low, correctly.
_LOCODE_COUNTRIES = frozenset({"DK", "SE", "NO", "DE", "PL", "NL", "GB",
                               "FI", "RU", "LT", "LV", "EE"})
# English exonyms -> local registry spellings, applied per token.
_ALIASES = {"COPENHAGEN": "KOBENHAVN", "GOTHENBURG": "GOTEBORG",
            "ELSINORE": "HELSINGOR", "MALMOE": "MALMO"}
_LOCODE_MAP = {
    "DKCPH": "KOBENHAVN", "DKSKA": "SKAGEN", "DKAAR": "AARHUS",
    "DKESB": "ESBJERG", "DKAAL": "AALBORG", "DKFRC": "FREDERICIA",
    "DKGRE": "GRENAA", "DKANH": "ANHOLT", "DKRNN": "RONNE",
    "DKKAL": "KALUNDBORG", "DKODE": "ODENSE", "DKHVS": "HVIDE SANDE",
    "SEGOT": "GOTEBORG", "SEMMA": "MALMO", "SEHEL": "HELSINGBORG",
    "SEYST": "YSTAD", "SETRG": "TRELLEBORG", "SEHAD": "HALMSTAD",
    "DEHAM": "HAMBURG", "DEKEL": "KIEL", "DEBRV": "BREMERHAVEN",
    "DEROS": "ROSTOCK", "DELBC": "LUBECK",
    "NOOSL": "OSLO", "NOKRS": "KRISTIANSAND",
    "PLGDN": "GDANSK", "PLGDY": "GDYNIA", "PLSZZ": "SZCZECIN",
    "NLRTM": "ROTTERDAM", "NLAMS": "AMSTERDAM",
}

# Generic tokens carry no port identity ("Aarhus Havn" ~ "AARHUS").
_STOP_TOKENS = frozenset({
    "HAVN", "HARBOUR", "HARBOR", "PORT", "TERMINAL", "MARINA", "LYSTBADEHAVN",
    "THE", "OF", "AND", "VIA", "FOR", "ORDERS", "SEA", "ANCH", "ANCHORAGE",
    "DENMARK", "SWEDEN", "NORWAY", "GERMANY", "DK", "SE", "NO", "DE",
})
_LOCODE_RE = re.compile(r"^[A-Z]{2}[A-Z2-9]{3}$")


def _clean(dest: str) -> str:
    return re.sub(r"\s+", " ", dest.upper().translate(_TRANSLIT).strip())


def _tok_covered(t: str, tok_set: frozenset) -> bool:
    """Exact token hit, or a near-miss (ONSVIG~ONSEVIG) — guarded so
    SequenceMatcher only runs on plausible pairs."""
    if t in tok_set:
        return True
    return any(abs(len(rt) - len(t)) <= 2 and rt[0] == t[0]
               and SequenceMatcher(None, t, rt).ratio() >= 0.85
               for rt in tok_set)


def _tokens(s: str) -> list[str]:
    return [_ALIASES.get(t, t) for t in re.split(r"[^A-Z0-9]+", s)
            if len(t) >= 3 and t not in _STOP_TOKENS]


@lru_cache(maxsize=1)
def _registry_index(path_str: str | None = None) -> list[tuple[str, frozenset, str, float, float]]:
    """[(name, token_set, joined_tokens, lat, lon)] from the OSM registry."""
    from trackfm.datasets.ports import load_registry
    reg = load_registry(Path(path_str) if path_str else None)
    idx = []
    for name, lat, lon in zip(reg["name"], reg["lat"], reg["lon"]):
        toks = _tokens(_clean(name))
        if toks:
            idx.append((name, frozenset(toks), " ".join(toks), lat, lon))
    return idx


@dataclass(frozen=True)
class DestinationMatch:
    raw: str
    name: str | None          # best registry candidate (None only for empty input)
    lat: float | None
    lon: float | None
    confidence: float         # [0, 1]; caller thresholds
    method: str               # exact | fuzzy | locode-unresolved | empty


@lru_cache(maxsize=65536)   # destinations repeat heavily across reports
def normalize_destination(dest: str | None,
                          registry_path: str | None = None) -> DestinationMatch:
    raw = dest or ""
    cleaned = _clean(raw)
    # AIS FROM>TO convention: the segment after '>' is the destination.
    if ">" in cleaned:
        cleaned = cleaned.rsplit(">", 1)[1].strip()
    if not cleaned:
        return DestinationMatch(raw, None, None, None, 0.0, "empty")
    method = "fuzzy"
    compact = cleaned.replace(" ", "")
    if _LOCODE_RE.fullmatch(compact) and compact[:2] in _LOCODE_COUNTRIES:
        mapped = _LOCODE_MAP.get(compact)
        if mapped is None:
            return DestinationMatch(raw, None, None, None, 0.0,
                                    "locode-unresolved")
        cleaned, method = mapped, "locode"   # resolve, then match normally

    d_toks = _tokens(cleaned)
    best, best_score, exact = None, -1.0, False
    for name, tok_set, joined, lat, lon in _registry_index(registry_path):
        if cleaned == _clean(name):
            best, best_score, exact = (name, lat, lon), 1.0, True
            break
        coverage = (sum(_tok_covered(t, tok_set) for t in d_toks)
                    / len(d_toks)) if d_toks else 0.0
        ratio = SequenceMatcher(None, " ".join(d_toks) or cleaned, joined).ratio()
        score = 0.7 * coverage + 0.3 * ratio
        # Prefer the more general (shorter) name at equal score, e.g.
        # "Aarhus Havn" over "APM Container Terminal Aarhus" for "AARHUS".
        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9
                                         and best and len(name) < len(best[0])):
            best, best_score = (name, lat, lon), score
    if best is None:
        return DestinationMatch(raw, None, None, None, 0.0, method)
    return DestinationMatch(raw, best[0], best[1], best[2],
                            round(max(best_score, 0.0), 4),
                            "exact" if exact else method)


@dataclass(frozen=True)
class ParsedEta:
    eta: datetime | None
    flag: str                 # ok | year-rolled | stale | far-future | unparseable
    hours_out: float | None   # eta - report_time, after any year roll


def parse_eta(eta_str: str | None, report_time: datetime,
              stale_days: float = 2.0, far_days: float = 60.0) -> ParsedEta:
    """DMA renders AIS ETA as 'dd/mm/yyyy HH:MM:SS' with the year filled in
    from the report date; roll candidates >300d off to the nearest year."""
    if not eta_str:
        return ParsedEta(None, "unparseable", None)
    try:
        eta = datetime.strptime(eta_str.strip(), "%d/%m/%Y %H:%M:%S")
    except ValueError:
        return ParsedEta(None, "unparseable", None)
    flag = "ok"
    if (eta - report_time) > timedelta(days=300):
        eta, flag = eta.replace(year=eta.year - 1), "year-rolled"
    elif (report_time - eta) > timedelta(days=300):
        eta, flag = eta.replace(year=eta.year + 1), "year-rolled"
    hours_out = (eta - report_time).total_seconds() / 3600.0
    if hours_out < -stale_days * 24:
        flag = "stale"
    elif hours_out > far_days * 24:
        flag = "far-future"
    return ParsedEta(eta, flag, round(hours_out, 2))
