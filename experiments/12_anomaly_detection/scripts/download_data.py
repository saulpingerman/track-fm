#!/usr/bin/env python3
"""
Download DTU Danish Waters AIS dataset for anomaly detection.

Dataset: AIS Trajectories from Danish Waters for Abnormal Behavior Detection
URL: https://data.dtu.dk/articles/dataset/AIS_Trajectories_from_Danish_Waters_for_Abnormal_Behavior_Detection
"""

import argparse
import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


# DTU dataset download URL (may need to be updated)
DTU_DOWNLOAD_URL = "https://data.dtu.dk/ndownloader/articles/19446300/versions/1"

# Alternative: Direct links to specific files if available
ALTERNATIVE_URLS = [
    "https://data.dtu.dk/ndownloader/files/34445916",  # Example file ID
]


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, output_dir: Path):
    """Extract zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Download DTU dataset')
    parser.add_argument('--output', type=str, default='data/raw/',
                        help='Output directory')
    parser.add_argument('--url', type=str, default=DTU_DOWNLOAD_URL,
                        help='Download URL')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DTU Danish Waters AIS Dataset Downloader")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Download URL: {args.url}")

    # Check if data already exists
    existing_files = list(output_dir.glob('**/*.csv')) + list(output_dir.glob('**/*.parquet'))
    if existing_files:
        print(f"\nFound {len(existing_files)} existing data files.")
        response = input("Download anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Skipping download.")
            return

    print("\n" + "-" * 60)
    print("IMPORTANT: Manual download may be required")
    print("-" * 60)
    print("""
The DTU dataset requires accepting terms of use before download.
If automatic download fails, please:

1. Visit: https://data.dtu.dk/articles/dataset/AIS_Trajectories_from_Danish_Waters_for_Abnormal_Behavior_Detection/19446300

2. Accept the terms of use

3. Download the dataset files manually

4. Extract to: {output_dir}

The dataset should contain:
- CSV or Parquet files with AIS trajectories
- Labels indicating normal vs abnormal behavior
""".format(output_dir=output_dir))

    # Try automatic download
    try:
        print("\nAttempting automatic download...")
        zip_path = output_dir / "dtu_dataset.zip"

        download_file(args.url, zip_path)

        # Check if it's a valid zip
        if zipfile.is_zipfile(zip_path):
            extract_zip(zip_path, output_dir)
            print("\nDownload successful!")
        else:
            print("\nDownloaded file is not a zip. It may be the dataset directly.")
            # Rename to appropriate extension
            zip_path.rename(output_dir / "dtu_data.csv")

    except requests.exceptions.HTTPError as e:
        print(f"\nAutomatic download failed: {e}")
        print("Please download manually from the URL above.")
        return
    except Exception as e:
        print(f"\nError during download: {e}")
        print("Please download manually from the URL above.")
        return

    # Verify download
    data_files = list(output_dir.glob('**/*.csv')) + list(output_dir.glob('**/*.parquet'))
    if data_files:
        print(f"\nFound {len(data_files)} data files:")
        for f in data_files[:10]:
            print(f"  - {f.name}")
        if len(data_files) > 10:
            print(f"  ... and {len(data_files) - 10} more")
    else:
        print("\nWarning: No CSV or Parquet files found after extraction.")
        print("Please check the download and extract manually if needed.")


if __name__ == '__main__':
    main()
