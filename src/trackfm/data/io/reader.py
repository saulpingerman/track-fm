"""Reading ZIP/CSV files from local filesystem."""
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)

COLUMN_PATTERNS = {
    "timestamp": re.compile(r"^#?\s*timestamp$", re.I),
    "mmsi": re.compile(r"^mmsi$", re.I),
    "lat": re.compile(r"^lat(itude)?$", re.I),
    "lon": re.compile(r"^lon(gitude)?$", re.I),
    "sog": re.compile(r"^sog$", re.I),
    "cog": re.compile(r"^cog$", re.I),
    "heading": re.compile(r"^heading$", re.I),
    "ship_type": re.compile(r"^ship.?type$", re.I),
    "imo": re.compile(r"^imo$", re.I),
    "name": re.compile(r"^name$", re.I),
    "callsign": re.compile(r"^callsign$", re.I),
}

TIMESTAMP_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
]


def map_columns(source_columns: List[str]) -> Dict[str, str]:
    """Map source columns to canonical names."""
    rename_map: Dict[str, str] = {}
    for canonical, pattern in COLUMN_PATTERNS.items():
        for source_col in source_columns:
            if pattern.match(source_col):
                if source_col not in rename_map:
                    rename_map[source_col] = canonical
                    break
    return rename_map


def list_raw_files(raw_dir: Path) -> List[Path]:
    """List all aisdk-*.zip files under raw_dir, recursively, sorted by name."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        logger.error(f"Raw directory does not exist: {raw_dir}")
        return []

    zip_files = sorted(p for p in raw_dir.rglob("aisdk-*.zip") if p.is_file())
    return zip_files


def iter_zip_csvs(path: Path):
    """Yield one DataFrame per CSV member of a DMA zip, in filename (date) order.

    Monthly zips bundle one CSV per day; yielding members individually keeps
    peak memory bounded to a single day regardless of zip size.
    """
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as zf:
            csv_members = sorted(
                (m for m in zf.infolist() if m.filename.lower().endswith(".csv")),
                key=lambda m: m.filename,
            )
            for member in csv_members:
                try:
                    df = read_csv_from_zip(zf, member.filename)
                except Exception as e:
                    logger.warning(f"Error processing CSV {member.filename}: {e}")
                    continue
                if df is None or df.is_empty():
                    continue
                if "latitude" in df.columns:
                    df = df.rename({"latitude": "lat"})
                if "longitude" in df.columns:
                    df = df.rename({"longitude": "lon"})
                yield member.filename, df
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")


def read_zip(path: Path) -> Optional[pl.DataFrame]:
    """Read all CSVs out of a local DMA zip and return a combined DataFrame."""
    path = Path(path)
    try:
        logger.info(f"Reading {path}")

        all_data: List[pl.DataFrame] = []
        with zipfile.ZipFile(path) as zf:
            csv_members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]

            for member in csv_members:
                try:
                    df = read_csv_from_zip(zf, member.filename)
                    if df is not None and not df.is_empty():
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"Error processing CSV {member.filename}: {e}")
                    continue

        if not all_data:
            logger.warning(f"No valid data found in {path}")
            return None

        combined_df = pl.concat(all_data, how="diagonal")

        if "latitude" in combined_df.columns:
            combined_df = combined_df.rename({"latitude": "lat"})
        if "longitude" in combined_df.columns:
            combined_df = combined_df.rename({"longitude": "lon"})

        logger.info(f"Read {combined_df.height} records from {path}")
        return combined_df

    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None


def read_csv_from_zip(zf: zipfile.ZipFile, filename: str) -> Optional[pl.DataFrame]:
    """Read a single CSV file from a ZIP archive."""
    with zf.open(filename) as csv_stream:
        text_stream = io.TextIOWrapper(csv_stream, encoding="utf-8", errors="ignore")

        try:
            df = pl.read_csv(
                text_stream,
                separator=",",
                ignore_errors=True,
                truncate_ragged_lines=True,
                infer_schema_length=1000,
            )

            rename_map = map_columns(df.columns)
            if not rename_map:
                return None

            cols_to_keep = [col for col in rename_map.keys() if col in df.columns]
            if not cols_to_keep:
                return None

            df = df.select(cols_to_keep).rename(rename_map)

            if "timestamp" in df.columns:
                df = parse_timestamp(df)
                if df is None:
                    return None

            return df

        except Exception as e:
            logger.warning(f"Error reading CSV {filename}: {e}")
            return None


def parse_timestamp(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """Parse timestamp column trying multiple formats."""
    for fmt in TIMESTAMP_FORMATS:
        try:
            parsed_df = df.with_columns([
                pl.col("timestamp").str.strptime(
                    pl.Datetime,
                    format=fmt,
                    strict=False,
                )
            ]).filter(pl.col("timestamp").is_not_null())

            if not parsed_df.is_empty():
                return parsed_df
        except Exception:
            continue

    return None
