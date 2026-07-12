#!/usr/bin/env python3
"""Danish Maritime Authority AIS Data Downloader (local filesystem).

Downloads AIS data from Danish Maritime Authority for specified date ranges
and writes them under <raw_dir>/<year>/aisdk-<date>.zip. Skips files that
already exist locally. Handles both monthly (pre-March 2024) and daily
(March 2024+) data automatically.

Usage:
    python download_ais_data.py --start-date 2024-01-01 --end-date 2024-01-31
    python download_ais_data.py --start-date 2020-01-01 --end-date 2020-12-31
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import requests

# Allow running without installing the package: add src/ to path.
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trackfm.data.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DMA_BASE_URL = "http://aisdata.ais.dk/"


class AISDataDownloader:
    """Downloads AIS data from the Danish Maritime Authority into a local directory."""

    def __init__(self, raw_dir: Path, base_url: str = DMA_BASE_URL):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url

    def local_path_for(self, remote_filename: str) -> Path:
        """Compute the local destination for a remote DMA filename like '2024/aisdk-2024-01-15.zip'."""
        return self.raw_dir / remote_filename

    def generate_file_list(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Generate list of expected DMA filenames (with year prefix) for date range.

        Daily files started in March 2024; earlier dates use monthly aggregates.
        """
        files: List[str] = []
        current_date = start_date

        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            day = current_date.day

            if year >= 2024 and (year > 2024 or month >= 3):
                filename = f"{year}/aisdk-{year:04d}-{month:02d}-{day:02d}.zip"
                files.append(filename)
                current_date += timedelta(days=1)
            else:
                filename = f"{year}/aisdk-{year:04d}-{month:02d}.zip"
                if filename not in files:
                    files.append(filename)
                if month == 12:
                    current_date = datetime(year + 1, 1, 1)
                else:
                    current_date = datetime(year, month + 1, 1)

        return files

    def check_file_exists_remote(self, filename: str) -> bool:
        url = f"{self.base_url}{filename}"
        try:
            response = requests.head(url, timeout=30)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Could not check {filename}: {e}")
            return False

    def download_file(self, filename: str) -> bool:
        """Download a single file directly to its final local path (atomic via .part)."""
        url = f"{self.base_url}{filename}"
        local_path = self.local_path_for(filename)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(local_path.suffix + ".part")

        try:
            logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            last_logged_pct = -10

            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int(downloaded / total_size * 100)
                        if pct >= last_logged_pct + 10:
                            logger.info(f"  Progress: {pct}%")
                            last_logged_pct = pct

            tmp_path.replace(local_path)
            file_size = local_path.stat().st_size
            logger.info(f"Downloaded {filename} ({file_size / 1024 / 1024:.1f} MB)")
            return True

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        force_redownload: bool = False,
    ) -> Tuple[int, int]:
        """Download AIS data for the specified date range.

        Returns:
            (successful_downloads, skipped_files)
        """
        logger.info(f"Processing AIS data from {start_date.date()} to {end_date.date()}")

        expected_files = self.generate_file_list(start_date, end_date)
        logger.info(f"Expected {len(expected_files)} files for date range")

        files_to_download: List[str] = []
        skipped = 0
        for filename in expected_files:
            local_path = self.local_path_for(filename)
            if not force_redownload and local_path.exists():
                logger.info(f"Skipping {filename} (already exists locally)")
                skipped += 1
            else:
                files_to_download.append(filename)

        if not files_to_download:
            logger.info("All files already exist locally")
            return 0, skipped

        logger.info(f"Need to download {len(files_to_download)} files")

        successful = 0
        for filename in files_to_download:
            if not self.check_file_exists_remote(filename):
                logger.warning(f"{filename} not found on server, skipping")
                continue

            if self.download_file(filename):
                successful += 1

        logger.info(f"Summary: {successful} downloaded, {skipped} skipped")
        return successful, skipped


def parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(
        description="Download AIS data from Danish Maritime Authority to a local directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download January 2024 (daily files for >= March 2024)
  python download_ais_data.py --start-date 2024-01-01 --end-date 2024-01-31

  # Download year 2020 (monthly files)
  python download_ais_data.py --start-date 2020-01-01 --end-date 2020-12-31

  # Force redownload existing files
  python download_ais_data.py --start-date 2024-01-01 --end-date 2024-01-01 --force
        """,
    )
    parser.add_argument("--start-date", type=parse_date, required=True,
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=parse_date, required=True,
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--raw-dir",
                        help="Override raw_dir from config (path to write files into)")
    parser.add_argument("--config", default="config/production.yaml",
                        help="Configuration file path (used when --raw-dir not given)")
    parser.add_argument("--force", action="store_true",
                        help="Force redownload even if files exist locally")
    args = parser.parse_args()

    if args.start_date > args.end_date:
        logger.error("Start date must be before or equal to end date")
        sys.exit(1)

    if args.raw_dir:
        raw_dir = Path(args.raw_dir).expanduser()
    else:
        try:
            config = load_config(args.config)
            raw_dir = config.storage.raw_path
        except Exception as e:
            logger.error(f"Could not load config: {e}")
            sys.exit(1)

    logger.info(f"Output directory: {raw_dir}")
    downloader = AISDataDownloader(raw_dir)

    try:
        downloaded, skipped = downloader.download_date_range(
            args.start_date, args.end_date, args.force
        )
        if downloaded > 0:
            logger.info(f"Successfully downloaded {downloaded} files to {raw_dir}")
        else:
            logger.info("No new files to download")
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
