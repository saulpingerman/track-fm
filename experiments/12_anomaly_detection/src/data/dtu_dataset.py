"""
DTU Danish Waters AIS Dataset for Anomaly Detection.

This module loads and processes the DTU abnormal behavior dataset.
Dataset: https://data.dtu.dk/articles/dataset/Labelled_evaluation_datasets_of_AIS_Trajectories_from_Danish_Waters_for_Abnormal_Behavior_Detection
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle


class DTUDataset(Dataset):
    """
    PyTorch Dataset for DTU abnormal behavior detection.

    Features are preprocessed to match TrackFM format:
    [lat, lon, sog, cog_sin, cog_cos, dt]
    """

    def __init__(
        self,
        trajectories: List[np.ndarray],
        labels: np.ndarray,
        indices: np.ndarray,
        config: dict,
        augmentor=None
    ):
        """
        Args:
            trajectories: List of trajectory arrays, each (seq_len, 6)
            labels: Array of binary labels (0=normal, 1=anomaly)
            indices: Indices into trajectories/labels to use for this split
            config: Configuration dict with normalization params
            augmentor: Optional TrajectoryAugmentor for training data
        """
        self.trajectories = [trajectories[i] for i in indices]
        self.labels = labels[indices]
        self.config = config
        self.augmentor = augmentor
        self.max_seq_length = config['data']['max_seq_length']

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx].copy()
        label = self.labels[idx]

        # Apply augmentation if available (training only)
        if self.augmentor is not None:
            traj = self.augmentor(traj)

        # Pad or truncate to max_seq_length
        seq_len = len(traj)
        if seq_len > self.max_seq_length:
            traj = traj[:self.max_seq_length]
            seq_len = self.max_seq_length
        elif seq_len < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - seq_len, traj.shape[1]))
            traj = np.vstack([traj, padding])

        return {
            'features': torch.tensor(traj, dtype=torch.float32),
            'length': torch.tensor(seq_len, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def load_dtu_data(raw_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load DTU dataset from pickle files.

    The DTU dataset contains AIS trajectories labeled as normal or abnormal.
    Data is stored as multiple pickled dictionaries in the data file.

    Args:
        raw_path: Path to raw data directory

    Returns:
        trajectories: List of trajectory arrays (seq_len, 6)
        labels: Binary labels (0=normal, 1=anomaly)
    """
    raw_path = Path(raw_path)

    # Check for preprocessed cache
    cache_path = raw_path.parent / 'processed' / 'dtu_cache.pkl'
    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['trajectories'], data['labels']

    # Find pickle files
    pkl_files = list(raw_path.glob('data_*.pkl'))
    info_files = list(raw_path.glob('datasetInfo_*.pkl'))

    if pkl_files and info_files:
        # Load from DTU pickle format
        return _load_dtu_pickle(raw_path, pkl_files[0], info_files[0], cache_path)

    # Fallback to CSV/Parquet format
    csv_files = list(raw_path.glob('**/*.csv'))
    parquet_files = list(raw_path.glob('**/*.parquet'))

    if parquet_files:
        print(f"Found {len(parquet_files)} parquet files")
        trajectories, labels = _load_from_dataframes(parquet_files, 'parquet')
    elif csv_files:
        print(f"Found {len(csv_files)} CSV files")
        trajectories, labels = _load_from_dataframes(csv_files, 'csv')
    else:
        raise FileNotFoundError(
            f"No data files found in {raw_path}. "
            "Please download the DTU dataset (data_*.pkl and datasetInfo_*.pkl files)."
        )

    labels = np.array(labels)

    # Cache for future use
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'trajectories': trajectories, 'labels': labels}, f)

    return trajectories, labels


def _load_dtu_pickle(
    raw_path: Path,
    data_file: Path,
    info_file: Path,
    cache_path: Path
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load DTU dataset from pickle format.

    The data file contains multiple pickled trajectory dictionaries.
    The info file contains metadata including outlier labels.
    """
    print(f"Loading data from {data_file.name}")
    print(f"Loading info from {info_file.name}")

    # Load all trajectories from data file (multiple pickled objects)
    raw_trajectories = []
    with open(data_file, 'rb') as f:
        try:
            while True:
                raw_trajectories.append(pickle.load(f))
        except EOFError:
            pass

    print(f"Loaded {len(raw_trajectories)} raw trajectories")

    # Load dataset info (contains labels)
    with open(info_file, 'rb') as f:
        info = pickle.load(f)

    labels = np.array(info['outlierLabels'])

    # Convert raw trajectories to feature arrays
    trajectories = []
    for traj_dict in raw_trajectories:
        traj_array = _convert_trajectory(traj_dict)
        if traj_array is not None:
            trajectories.append(traj_array)

    print(f"Converted {len(trajectories)} trajectories")
    print(f"Labels: {np.sum(labels == 0)} normal, {np.sum(labels == 1)} anomaly")

    # Verify lengths match
    if len(trajectories) != len(labels):
        print(f"Warning: trajectory count ({len(trajectories)}) != label count ({len(labels)})")
        min_len = min(len(trajectories), len(labels))
        trajectories = trajectories[:min_len]
        labels = labels[:min_len]

    # Cache for future use
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'trajectories': trajectories, 'labels': labels}, f)
    print(f"Cached processed data to {cache_path}")

    return trajectories, labels


def _convert_trajectory(traj_dict: Dict) -> Optional[np.ndarray]:
    """
    Convert a trajectory dictionary to feature array.

    Args:
        traj_dict: Dictionary with keys: lat, lon, speed, course, timestamp

    Returns:
        Array of shape (seq_len, 6) with features:
        [lat, lon, sog, cog_sin, cog_cos, dt]
    """
    try:
        lat = np.array(traj_dict['lat'])
        lon = np.array(traj_dict['lon'])
        sog = np.array(traj_dict['speed'])
        cog = np.array(traj_dict['course'])
        timestamp = np.array(traj_dict['timestamp'])

        # Convert COG to sin/cos
        cog_rad = np.radians(cog)
        cog_sin = np.sin(cog_rad)
        cog_cos = np.cos(cog_rad)

        # Calculate dt (time delta in seconds)
        if len(timestamp) > 0:
            if isinstance(timestamp[0], (int, float, np.integer, np.floating)):
                # Unix timestamps
                dt = np.diff(timestamp, prepend=timestamp[0])
                dt[0] = 0  # First point has no delta
            else:
                # Try to convert to datetime
                times = pd.to_datetime(timestamp)
                dt = times.diff().dt.total_seconds().fillna(0).values
        else:
            dt = np.zeros(len(lat))

        # Clip dt to reasonable values (0 to 3600 seconds)
        dt = np.clip(dt, 0, 3600)

        # Stack features: [lat, lon, sog, cog_sin, cog_cos, dt]
        trajectory = np.stack([lat, lon, sog, cog_sin, cog_cos, dt], axis=1)

        # Handle NaN values
        trajectory = np.nan_to_num(trajectory, nan=0.0)

        return trajectory.astype(np.float32)

    except Exception as e:
        print(f"Error converting trajectory: {e}")
        return None


def _load_from_dataframes(files: List[Path], file_type: str) -> Tuple[List[np.ndarray], List[int]]:
    """Load trajectories from CSV or Parquet files."""
    trajectories = []
    labels = []

    for file_path in files:
        if file_type == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        trajs, lbls = _parse_dtu_dataframe(df)
        trajectories.extend(trajs)
        labels.extend(lbls)

    return trajectories, labels


def _parse_dtu_dataframe(df: pd.DataFrame) -> Tuple[List[np.ndarray], List[int]]:
    """Parse DTU dataframe into trajectories and labels."""
    trajectories = []
    labels = []

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Find trajectory ID column
    id_col = None
    for col in ['mmsi', 'track_id', 'trajectory_id', 'id', 'vesselid']:
        if col in df.columns:
            id_col = col
            break

    if id_col is None:
        traj, label = _extract_single_trajectory(df)
        if traj is not None:
            trajectories.append(traj)
            labels.append(label)
        return trajectories, labels

    # Group by trajectory ID
    for traj_id, group in df.groupby(id_col):
        traj, label = _extract_single_trajectory(group)
        if traj is not None:
            trajectories.append(traj)
            labels.append(label)

    return trajectories, labels


def _extract_single_trajectory(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], int]:
    """Extract a single trajectory from a dataframe group."""
    lat_col = next((c for c in ['lat', 'latitude'] if c in df.columns), None)
    lon_col = next((c for c in ['lon', 'longitude'] if c in df.columns), None)
    sog_col = next((c for c in ['sog', 'speed', 'speed_over_ground'] if c in df.columns), None)
    cog_col = next((c for c in ['cog', 'course', 'course_over_ground'] if c in df.columns), None)
    time_col = next((c for c in ['timestamp', 'time', 'datetime', 'basedatetime'] if c in df.columns), None)
    label_col = next((c for c in ['label', 'abnormal', 'anomaly', 'is_anomaly'] if c in df.columns), None)

    if lat_col is None or lon_col is None:
        return None, 0

    if time_col:
        df = df.sort_values(time_col)

    lat = df[lat_col].values
    lon = df[lon_col].values
    sog = df[sog_col].values if sog_col else np.zeros(len(df))
    cog = df[cog_col].values if cog_col else np.zeros(len(df))

    cog_rad = np.radians(cog)
    cog_sin = np.sin(cog_rad)
    cog_cos = np.cos(cog_rad)

    if time_col:
        times = pd.to_datetime(df[time_col])
        dt = times.diff().dt.total_seconds().fillna(0).values
    else:
        dt = np.ones(len(df)) * 10.0

    label = int(df[label_col].iloc[0]) if label_col else 0

    trajectory = np.stack([lat, lon, sog, cog_sin, cog_cos, dt], axis=1)
    trajectory = np.nan_to_num(trajectory, nan=0.0)

    return trajectory.astype(np.float32), label


def get_dataset_info(raw_path: str) -> Dict:
    """
    Get dataset information including pre-defined splits.

    Args:
        raw_path: Path to raw data directory

    Returns:
        Dictionary with dataset info including train/test indices
    """
    raw_path = Path(raw_path)
    info_files = list(raw_path.glob('datasetInfo_*.pkl'))

    if not info_files:
        return {}

    with open(info_files[0], 'rb') as f:
        info = pickle.load(f)

    return {
        'n_trajectories': len(info.get('indicies', [])),
        'n_anomalies': int(np.sum(info.get('outlierLabels', []))),
        'train_indices': info.get('trainIndicies', []),
        'test_indices': info.get('testIndicies', []),
        'mean': info.get('mean', []),
        'std': info.get('std', []),
    }
