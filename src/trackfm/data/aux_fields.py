"""Vessel-intrinsic auxiliary-field extraction from raw DMA zips.

The cleaning pipeline keeps only kinematic/identity columns; this module
re-reads the raw zips and extracts the slow-moving vessel-intrinsic fields
(navigational status, draught, destination, ETA, dimensions, cargo type) as
a compact per-(mmsi, day) CHANGE-LOG: one row per report where any watched
field differs from the vessel's previous report that day, plus the first
report of each vessel each day.

Output mirrors the clean layout:
    <aux_dir>/year=YYYY/month=MM/day=DD/aux.parquet   (zstd)
    <aux_dir>/MANIFEST.json                           (per-day rows, git SHA)

Monthly zips (pre March 2024) bundle one CSV member per day; members are
streamed individually so peak memory stays bounded to a single day.
Resumable: a day whose aux.parquet already exists is skipped, and fully
processed zips are tracked with the same ProcessingCheckpoint pattern as
the cleaning pipeline (in a separate state dir).
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import zipfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import polars as pl

from .io.reader import list_raw_files, parse_timestamp
from .io.writer import write_parquet
from .state.checkpoint import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)

# Canonical name -> raw header pattern. DMA has renamed columns before
# (Latitude/lat etc.), so match tolerantly: optional separators, known
# alternate spellings (Draft, Rate of turn, Nav status).
AUX_COLUMN_PATTERNS: Dict[str, re.Pattern] = {
    "timestamp": re.compile(r"^#?\s*timestamp$", re.I),
    "mmsi": re.compile(r"^mmsi$", re.I),
    "nav_status": re.compile(r"^nav(igational)?[ _-]?status$", re.I),
    "rot": re.compile(r"^(rot|rate[ _-]?of[ _-]?turn)$", re.I),
    "width": re.compile(r"^width$", re.I),
    "length": re.compile(r"^length$", re.I),
    "draught": re.compile(r"^(draught|draft)$", re.I),
    "destination": re.compile(r"^destination$", re.I),
    "eta": re.compile(r"^eta$", re.I),
    "cargo_type": re.compile(r"^cargo[ _-]?type$", re.I),
}

# Fields whose change (vs. the vessel's previous report that day) triggers a row.
WATCHED_FIELDS = ["nav_status", "draught", "destination", "eta", "width", "length", "cargo_type"]

OUTPUT_SCHEMA: Dict[str, pl.DataType] = {
    "mmsi": pl.Int64,
    "timestamp": pl.Datetime("us"),
    "nav_status": pl.String,
    "rot": pl.Float32,
    "draught": pl.Float32,
    "destination": pl.String,
    "eta": pl.String,
    "width": pl.Float32,
    "length": pl.Float32,
    "cargo_type": pl.String,
}

# Plausibility bounds (values outside are nulled, rows are kept).
DRAUGHT_RANGE = (0.0, 30.0)
DIMENSION_RANGE = (0.0, 500.0)

MEMBER_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
MANIFEST_FILENAME = "MANIFEST.json"


def member_date(member_name: str) -> Optional[date]:
    """Extract the day covered by a CSV member from its filename."""
    m = MEMBER_DATE_RE.search(Path(member_name).stem)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def day_output_path(aux_dir: Path, day: date) -> Path:
    """Partition path for one day's change-log."""
    return (
        Path(aux_dir)
        / f"year={day.year}"
        / f"month={day.month:02d}"
        / f"day={day.day:02d}"
        / "aux.parquet"
    )


def map_aux_columns(source_columns: Sequence[str]) -> Dict[str, str]:
    """Map raw header names to canonical aux column names."""
    rename_map: Dict[str, str] = {}
    for canonical, pattern in AUX_COLUMN_PATTERNS.items():
        for source_col in source_columns:
            if pattern.match(source_col.strip()) and source_col not in rename_map:
                rename_map[source_col] = canonical
                break
    return rename_map


def read_aux_csv_from_zip(zf: zipfile.ZipFile, filename: str) -> Optional[pl.DataFrame]:
    """Read only the aux columns of one CSV member, timestamps parsed.

    The header is sniffed first so the read projects just the ~10 needed
    columns (out of 26), keeping one raw day around 1-2 GB instead of 10+.
    """
    with zf.open(filename) as stream:
        header = stream.readline().decode("utf-8", errors="ignore").strip()
    source_columns = [c.strip() for c in header.split(",")]
    rename_map = map_aux_columns(source_columns)

    missing = set(AUX_COLUMN_PATTERNS) - set(rename_map.values())
    if "timestamp" in missing or "mmsi" in missing:
        logger.warning(f"{filename}: no timestamp/mmsi column, skipping (header: {header[:120]})")
        return None
    if missing:
        logger.warning(f"{filename}: raw columns missing for {sorted(missing)}")

    string_fields = {"nav_status", "destination", "eta", "cargo_type"}
    overrides = {
        src: (pl.String if canon in string_fields else pl.Float64)
        for src, canon in rename_map.items()
        if canon not in ("timestamp", "mmsi")
    }
    overrides.update({src: pl.Int64 for src, canon in rename_map.items() if canon == "mmsi"})

    with zf.open(filename) as stream:
        df = pl.read_csv(
            stream,
            separator=",",
            columns=list(rename_map.keys()),
            schema_overrides=overrides,
            encoding="utf8-lossy",
            ignore_errors=True,
            truncate_ragged_lines=True,
            infer_schema_length=1000,
        )
    df = df.rename(rename_map)

    df = parse_timestamp(df)
    if df is None or df.is_empty():
        return None
    return df


def build_change_log(df: pl.DataFrame, day: Optional[date] = None) -> pl.DataFrame:
    """Reduce one day's reports to a per-mmsi change-log of watched fields.

    Keeps a row when any WATCHED_FIELDS value differs (null-aware) from the
    vessel's previous report, plus each vessel's first report of the day.
    Values are normalized first (plausibility bounds, destination casing) so
    changes are detected on the cleaned values.
    """
    for name, dtype in OUTPUT_SCHEMA.items():
        if name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(name))

    lf = df.lazy().filter(pl.col("mmsi").is_not_null() & pl.col("timestamp").is_not_null())
    if day is not None:
        lf = lf.filter(pl.col("timestamp").dt.date() == pl.lit(day))

    lf = lf.with_columns([
        pl.col("mmsi").cast(pl.Int64),
        pl.col("timestamp").cast(pl.Datetime("us")),
        pl.col("nav_status").cast(pl.String).str.strip_chars(),
        pl.col("rot").cast(pl.Float32),
        pl.col("draught")
        .cast(pl.Float32)
        .pipe(lambda c: pl.when(c.is_between(*DRAUGHT_RANGE)).then(c)),
        pl.col("destination").cast(pl.String).str.strip_chars().str.to_uppercase(),
        pl.col("eta").cast(pl.String),
        pl.col("width")
        .cast(pl.Float32)
        .pipe(lambda c: pl.when(c.is_between(*DIMENSION_RANGE)).then(c)),
        pl.col("length")
        .cast(pl.Float32)
        .pipe(lambda c: pl.when(c.is_between(*DIMENSION_RANGE)).then(c)),
        pl.col("cargo_type").cast(pl.String),
    ])

    changed = pl.any_horizontal([
        pl.col(f).ne_missing(pl.col(f).shift(1)).over("mmsi") for f in WATCHED_FIELDS
    ])
    first_of_day = pl.int_range(pl.len()).over("mmsi") == 0

    return (
        lf.sort(["mmsi", "timestamp"], maintain_order=True)
        .filter(changed | first_of_day)
        .select(list(OUTPUT_SCHEMA))
        .collect()
    )


def iter_aux_members(zip_path: Path, only: Optional[set] = None):
    """Yield (member, day, raw aux DataFrame) per CSV member, in date order.

    Streaming counterpart of the cleaning pipeline's iter_zip_csvs: monthly
    zips are never concatenated, so peak memory stays bounded to one day.
    only: optional set of member names; others are skipped without reading
    (resumed/day-filtered runs must not re-parse finished days).
    """
    zip_path = Path(zip_path)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            csv_members = sorted(
                m.filename for m in zf.infolist() if m.filename.lower().endswith(".csv")
            )
            if only is not None:
                csv_members = [m for m in csv_members if m in only]
            for member in csv_members:
                day = member_date(member)
                if day is None:
                    logger.warning(f"{zip_path.name}::{member}: cannot parse date, skipping")
                    continue
                try:
                    df = read_aux_csv_from_zip(zf, member)
                except Exception as e:
                    logger.warning(f"Error reading {zip_path.name}::{member}: {e}")
                    continue
                if df is None or df.is_empty():
                    continue
                yield member, day, df
    except Exception as e:
        logger.error(f"Error reading {zip_path}: {e}")


def write_aux_manifest(aux_dir: Path) -> Path:
    """Write <aux_dir>/MANIFEST.json with per-day row counts and the git SHA."""
    aux_dir = Path(aux_dir)
    day_files = sorted(aux_dir.glob("year=*/month=*/day=*/aux.parquet"))

    entries = []
    total_rows = 0
    for p in day_files:
        year, month, day = (part.split("=")[1] for part in p.parts[-4:-1])
        rows = pl.scan_parquet(p).select(pl.len()).collect().item()
        total_rows += rows
        entries.append({"date": f"{year}-{month}-{day}", "rows": rows, "bytes": p.stat().st_size})

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        sha = "unknown"

    manifest = {
        "n_days": len(entries),
        "total_rows": total_rows,
        "pipeline_git_sha": sha or "unknown",
        "files": entries,
    }
    manifest_path = aux_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=1))
    logger.info(f"Wrote {manifest_path} ({len(entries)} days, {total_rows:,} rows)")
    return manifest_path


def extract_aux_fields(
    raw_dir: Path,
    aux_dir: Path,
    state_dir: Path,
    resume: bool = True,
    max_files: Optional[int] = None,
    days: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Extract aux-field change-logs for all (or selected) days of raw zips.

    days: optional list of YYYY-MM-DD strings restricting both which zips are
    opened and which members are processed (smoke tests / backfills). Zips are
    only marked processed in the checkpoint when run without a day filter.
    """
    raw_dir, aux_dir, state_dir = Path(raw_dir), Path(aux_dir), Path(state_dir)
    aux_dir.mkdir(parents=True, exist_ok=True)

    wanted_days = {date.fromisoformat(d) for d in days} if days else None

    checkpoint = load_checkpoint(state_dir) if resume else None
    all_files = list_raw_files(raw_dir)
    if wanted_days is not None:
        stems = {f"aisdk-{d.isoformat()}" for d in wanted_days} | {
            f"aisdk-{d.year}-{d.month:02d}" for d in wanted_days
        }
        all_files = [p for p in all_files if p.stem in stems]

    pending = [str(p) for p in all_files]
    if checkpoint is not None:
        pending = checkpoint.get_pending_files(pending)
    if max_files:
        pending = pending[:max_files]

    logger.info(f"extract-aux: {len(pending)} zip(s) pending (of {len(all_files)} selected)")

    stats = {"zips_processed": 0, "days_written": 0, "days_skipped": 0, "rows_written": 0}

    for zip_key in pending:
        zip_path = Path(zip_key)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = sorted(
                    m.filename for m in zf.infolist() if m.filename.lower().endswith(".csv")
                )
            member_days = {m: member_date(m) for m in members}
            todo = [
                m for m, d in member_days.items()
                if d is not None
                and (wanted_days is None or d in wanted_days)
                and not (resume and day_output_path(aux_dir, d).exists())
            ]
            skipped = sum(
                1 for m, d in member_days.items()
                if d is not None
                and (wanted_days is None or d in wanted_days)
                and m not in todo
            )
            stats["days_skipped"] += skipped

            if todo:
                for member, day, raw_df in iter_aux_members(zip_path, only=set(todo)):
                    n_raw = raw_df.height
                    change_log = build_change_log(raw_df, day=day)
                    del raw_df
                    out_path = day_output_path(aux_dir, day)
                    write_parquet(change_log, out_path, compression="zstd")
                    logger.info(
                        f"  {zip_path.name}::{member}: {n_raw:,} raw -> "
                        f"{change_log.height:,} change rows"
                    )
                    stats["days_written"] += 1
                    stats["rows_written"] += change_log.height

            if checkpoint is not None and wanted_days is None:
                checkpoint.mark_processed(zip_key)
                save_checkpoint(checkpoint, state_dir)
            stats["zips_processed"] += 1

        except Exception as e:
            logger.error(f"Error processing {zip_key}: {e}")
            if checkpoint is not None and wanted_days is None:
                checkpoint.mark_failed(zip_key)
                save_checkpoint(checkpoint, state_dir)

    write_aux_manifest(aux_dir)
    logger.info(f"extract-aux done: {stats}")
    return stats
