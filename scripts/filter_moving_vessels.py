#!/usr/bin/env python3
"""
Filter AIS data to keep only tracks with average speed > 5 knots
"""
import polars as pl
import numpy as np
from datetime import datetime
import os


def calculate_track_speeds(df):
    """Calculate average speed for each track"""
    
    print("Calculating average speeds for all tracks...")
    
    # Group by track_id and calculate speeds
    track_stats = []
    
    for track_id in df['track_id'].unique():
        track_df = df.filter(pl.col('track_id') == track_id).sort('timestamp')
        
        if len(track_df) < 2:
            continue
            
        lats = track_df['lat'].to_numpy()
        lons = track_df['lon'].to_numpy()
        timestamps = track_df['timestamp'].to_numpy()
        
        # Calculate total distance and time
        total_distance_nm = 0  # nautical miles
        total_time_hours = 0
        
        for i in range(1, len(lats)):
            # Haversine distance
            lat1, lon1 = np.radians(lats[i-1]), np.radians(lons[i-1])
            lat2, lon2 = np.radians(lats[i]), np.radians(lons[i])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            # Earth radius in nautical miles
            R_nm = 3440.065  
            distance_nm = R_nm * c
            total_distance_nm += distance_nm
            
            # Time in hours - handle datetime objects
            if isinstance(timestamps[i], np.datetime64):
                # Convert numpy datetime64 to seconds
                time_diff_seconds = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 's')
            else:
                # Assume milliseconds
                time_diff_seconds = (timestamps[i] - timestamps[i-1]) / 1000.0
            time_diff_hours = time_diff_seconds / 3600.0
            total_time_hours += time_diff_hours
        
        # Average speed in knots
        if total_time_hours > 0:
            avg_speed_knots = total_distance_nm / total_time_hours
        else:
            avg_speed_knots = 0
            
        track_stats.append({
            'track_id': track_id,
            'num_points': len(track_df),
            'total_distance_nm': total_distance_nm,
            'total_time_hours': total_time_hours,
            'avg_speed_knots': avg_speed_knots
        })
    
    return pl.DataFrame(track_stats)


def main():
    # Load interpolated data
    input_path = "/home/ec2-user/ais_data/interpolated_ais_2025-07-06_01-34-27.parquet"
    print(f"Loading data from: {input_path}")
    
    df = pl.read_parquet(input_path)
    print(f"Total rows: {len(df):,}")
    print(f"Total unique tracks: {df['track_id'].n_unique():,}")
    
    # Calculate speeds for all tracks
    track_speeds = calculate_track_speeds(df)
    
    # Print speed distribution
    print("\nSpeed distribution:")
    print(f"Min speed: {track_speeds['avg_speed_knots'].min():.2f} knots")
    print(f"Max speed: {track_speeds['avg_speed_knots'].max():.2f} knots")
    print(f"Mean speed: {track_speeds['avg_speed_knots'].mean():.2f} knots")
    print(f"Median speed: {track_speeds['avg_speed_knots'].median():.2f} knots")
    
    # Speed brackets
    speed_brackets = [0, 1, 5, 10, 15, 20, 25, 30, 100]
    print("\nSpeed brackets:")
    for i in range(len(speed_brackets)-1):
        lower, upper = speed_brackets[i], speed_brackets[i+1]
        count = len(track_speeds.filter((pl.col('avg_speed_knots') >= lower) & 
                                       (pl.col('avg_speed_knots') < upper)))
        pct = count / len(track_speeds) * 100
        print(f"  {lower:3d}-{upper:3d} knots: {count:6d} tracks ({pct:5.1f}%)")
    
    # Filter tracks with speed > 5 knots
    moving_tracks = track_speeds.filter(pl.col('avg_speed_knots') > 5)['track_id'].to_list()
    print(f"\nTracks with avg speed > 5 knots: {len(moving_tracks):,} ({len(moving_tracks)/len(track_speeds)*100:.1f}%)")
    
    # Filter the original dataframe
    df_filtered = df.filter(pl.col('track_id').is_in(moving_tracks))
    print(f"\nFiltered data:")
    print(f"  Rows: {len(df_filtered):,} ({len(df_filtered)/len(df)*100:.1f}% of original)")
    print(f"  Tracks: {df_filtered['track_id'].n_unique():,}")
    
    # Save filtered data
    output_path = "/home/ec2-user/ais_data/interpolated_ais_moving_vessels_5knots.parquet"
    df_filtered.write_parquet(output_path)
    print(f"\nSaved filtered data to: {output_path}")
    
    # Save track statistics for reference
    stats_path = "/home/ec2-user/ais_data/track_speed_statistics.parquet"
    track_speeds.write_parquet(stats_path)
    print(f"Saved track statistics to: {stats_path}")
    
    # Print some example tracks
    print("\nExample moving tracks (>5 knots):")
    for row in track_speeds.filter(pl.col('avg_speed_knots') > 5).head(5).iter_rows(named=True):
        print(f"  {row['track_id']}: {row['avg_speed_knots']:.1f} knots, "
              f"{row['total_distance_nm']:.1f} nm in {row['total_time_hours']:.1f} hours")


if __name__ == "__main__":
    main()