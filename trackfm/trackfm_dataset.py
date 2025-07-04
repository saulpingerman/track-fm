#!/usr/bin/env python
"""
Track-FM dataset loader for multi-horizon trajectory forecasting
"""

import polars as pl
import torch
from torch.utils.data import IterableDataset
import boto3
import numpy as np
import random
from typing import List, Tuple, Optional

def latlon_to_local_uv(lat: float, lon: float, centre_lat: float, centre_lon: float) -> Tuple[float, float]:
    """Convert lat/lon to local UV coordinates centered at centre_lat/centre_lon"""
    # Approximate conversion for small distances
    # 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
    
    lat_diff = lat - centre_lat
    lon_diff = lon - centre_lon
    
    # Convert to meters then normalize to [-1, 1] range
    # Assume max distance of ~100 km for normalization
    u = lon_diff * 111000 * np.cos(np.radians(centre_lat)) / 100000
    v = lat_diff * 111000 / 100000
    
    return float(u), float(v)

class StreamingMultiHorizonAISDataset(IterableDataset):
    """
    Streaming dataset for multi-horizon trajectory forecasting from S3
    """
    
    def __init__(
        self,
        bucket_name: str,
        seq_len: int = 20,
        horizon: int = 10,
        chunk_size: int = 200_000,
        prefix: str = "cleaned/",
    ):
        super().__init__()
        self.bucket_name = bucket_name
        self.seq_len = seq_len
        self.horizon = horizon
        self.chunk_size = chunk_size
        self.prefix = prefix
        
        # List all parquet files
        self.parquet_files = self._list_parquet_files()
        print(f"🟢 Found {len(self.parquet_files)} parquet files for streaming")
        
    def _list_parquet_files(self) -> List[str]:
        """List all parquet files in the S3 bucket"""
        print(f"🔍 Listing objects in s3://{self.bucket_name}/{self.prefix}")
        
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=self.prefix
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.parquet'):
                    files.append(obj['Key'])
        
        return files
    
    def __iter__(self):
        """Generate batches of trajectory sequences"""
        while True:  # Infinite iteration
            # Shuffle files for each epoch
            files = self.parquet_files.copy()
            random.shuffle(files)
            
            for file_key in files:
                yield from self._process_file(file_key)
    
    def _process_file(self, file_key: str):
        """Process a single parquet file and yield trajectory sequences"""
        print(f"📂 Processing {file_key}")
        
        # Read file info to determine chunking
        s3_path = f"s3://{self.bucket_name}/{file_key}"
        df_scan = pl.scan_parquet(s3_path)
        total_rows = df_scan.select(pl.count()).collect().item()
        
        print(f"📊 File has {total_rows:,} rows, processing in chunks of {self.chunk_size:,}")
        
        # Process in chunks
        for chunk_start in range(0, total_rows, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_rows)
            print(f"📦 Processing chunk {chunk_start:,}-{chunk_end:,}")
            
            # Read chunk
            df_chunk = df_scan.slice(chunk_start, chunk_end - chunk_start).collect()
            
            # Group by MMSI and process trajectories
            for mmsi, group_df in df_chunk.group_by("mmsi"):
                yield from self._extract_sequences(group_df)
    
    def _extract_sequences(self, df: pl.DataFrame):
        """Extract valid sequences from a vessel's trajectory"""
        if len(df) < self.seq_len + self.horizon:
            return
        
        # Sort by timestamp
        df = df.sort("timestamp")
        
        # Extract coordinates
        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()
        
        # Convert to local coordinates using trajectory center
        centre_lat = float(np.mean(lats))
        centre_lon = float(np.mean(lons))
        
        uvs = []
        for lat, lon in zip(lats, lons):
            u, v = latlon_to_local_uv(lat, lon, centre_lat, centre_lon)
            uvs.append([u, v])
        
        uvs = np.array(uvs)
        
        # Extract sequences
        for i in range(len(uvs) - self.seq_len - self.horizon + 1):
            input_seq = uvs[i:i + self.seq_len]  # (seq_len, 2)
            target_seq = uvs[i + self.seq_len:i + self.seq_len + self.horizon]  # (horizon, 2)
            
            # Compute Jacobian for coordinate transformation
            # Simplified: assume uniform scaling
            logJ = np.log(1.0)  # Placeholder
            
            # Causal position (always predict from last position)
            causal_pos = self.seq_len - 1
            
            yield (
                torch.tensor(input_seq, dtype=torch.float32),
                torch.tensor([centre_lat, centre_lon], dtype=torch.float32),
                torch.tensor(target_seq, dtype=torch.float32),
                torch.tensor(logJ, dtype=torch.float32),
                torch.tensor(causal_pos, dtype=torch.long)
            )