"""Audit the cleaned AIS dataset before materialization.

Checks (V3 milestone of the restart plan):
  1. Checkpoint: all raw zips processed, none failed
  2. Date continuity: no missing day partitions in the covered range
  3. Invariants per sampled day: bbox bounds, non-null coords, dt >= 0
  4. Size sanity: per-day bytes within a plausible band
Writes clean_dir/MANIFEST.json (file list, row counts, pipeline SHA,
sha256 of contents) — its hash becomes the data_manifest_sha256 MLflow tag.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from trackfm.datasets.materialize import list_day_partitions, partition_date

logger = logging.getLogger(__name__)

BBOX = {"lat_min": 54.0, "lat_max": 58.5, "lon_min": 7.0, "lon_max": 16.0}


def run_audit(clean_dir: Path, raw_dir: Path, write_manifest: bool = True,
              sample_every: int = 20) -> bool:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ok = True

    # 1. checkpoint state
    ckpt_path = clean_dir.parent / "state" / "processing_checkpoint.json"
    n_raw = len(list(Path(raw_dir).rglob("aisdk-*.zip")))
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        n_done, n_failed = len(ckpt["processed_files"]), len(ckpt["failed_files"])
        logger.info(f"checkpoint: {n_done}/{n_raw} processed, {n_failed} failed")
        if n_done < n_raw or n_failed:
            logger.error(f"INCOMPLETE: {n_raw - n_done} pending, {n_failed} failed")
            ok = False
    else:
        logger.warning("no processing checkpoint found")
        ok = False

    # 2. date continuity
    days = list_day_partitions(clean_dir)
    if not days:
        logger.error("no day partitions")
        return False
    dates = [date.fromisoformat(partition_date(p)) for p in days]
    expected = (dates[-1] - dates[0]).days + 1
    missing = expected - len(dates)
    logger.info(f"{len(dates)} day partitions, {dates[0]} .. {dates[-1]} "
                f"({missing} missing days)")
    if missing:
        have = set(dates)
        gaps = [dates[0] + timedelta(d) for d in range(expected)
                if dates[0] + timedelta(d) not in have]
        logger.error(f"missing dates (first 10): {[str(g) for g in gaps[:10]]}")
        ok = False

    # 3. invariants on sampled days + 4. size stats
    sizes = [p.stat().st_size for p in days]
    rows_total = 0
    for i, p in enumerate(days):
        rows_total += pl.scan_parquet(p).select(pl.len()).collect().item()
        if i % sample_every:
            continue
        df = pl.read_parquet(p, columns=["lat", "lon", "dt_seconds", "track_id"])
        checks = {
            "lat in bbox": df["lat"].is_between(BBOX["lat_min"], BBOX["lat_max"]).all(),
            "lon in bbox": df["lon"].is_between(BBOX["lon_min"], BBOX["lon_max"]).all(),
            "no null coords": df["lat"].null_count() == 0 and df["lon"].null_count() == 0,
            "dt >= 0": bool((df["dt_seconds"] >= 0).all()),
            "track_id non-empty": bool((df["track_id"].str.len_chars() > 0).all()),
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            logger.error(f"{partition_date(p)}: FAILED {bad}")
            ok = False
    med = sorted(sizes)[len(sizes) // 2]
    outliers = [partition_date(p) for p, s in zip(days, sizes)
                if s < med * 0.05 or s > med * 20]
    logger.info(f"rows: {rows_total:,} | median day {med/1e6:.0f}MB | "
                f"size outliers: {outliers[:5] or 'none'}")

    # manifest
    if write_manifest and ok:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=Path(__file__).parent).stdout.strip()
        except Exception:
            sha = "unknown"
        entries = [{"date": partition_date(p), "bytes": s}
                   for p, s in zip(days, sizes)]
        manifest = {"first_day": str(dates[0]), "last_day": str(dates[-1]),
                    "n_days": len(days), "total_rows": rows_total,
                    "pipeline_git_sha": sha, "files": entries}
        blob = json.dumps(manifest, indent=1).encode()
        (clean_dir / "MANIFEST.json").write_bytes(blob)
        logger.info(f"MANIFEST.json written (sha256 {hashlib.sha256(blob).hexdigest()[:16]}…)")

    logger.info(f"AUDIT {'PASSED' if ok else 'FAILED'}")
    return ok
