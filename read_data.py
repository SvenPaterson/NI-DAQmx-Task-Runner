"""
Simple script to read a TDMS file and display the data as a pandas DataFrame.

Usage:
    python read_tdms.py

Install dependencies first:
    pip install nptdms pandas
"""

import sys
from pathlib import Path
from nptdms import TdmsFile
import pandas as pd
import tkinter as tk
from tkinter import filedialog


def read_tdms_file(file_path):
    """Read TDMS file and convert to pandas DataFrame."""
    
    # Read the TDMS file
    print(f"Reading TDMS file: {file_path}\n")
    tdms_file = TdmsFile.read(file_path)
    
    # Print file structure
    print("=" * 60)
    print("FILE STRUCTURE")
    print("=" * 60)
    
    groups = tdms_file.groups()
    print(f"Number of groups: {len(groups)}\n")
    
    for group in groups:
        print(f"Group: '{group.name}'")
        print(f"  Channels ({len(group.channels())}):")
        for channel in group.channels():
            print(f"    - {channel.name} ({len(channel)} samples)")
            # Print channel properties if they exist
            if channel.properties:
                print(f"      Properties: {dict(list(channel.properties.items())[:3])}...")
        print()
    
    # Convert to DataFrame with time index
    print("=" * 60)
    print("CONVERTING TO DATAFRAME")
    print("=" * 60)
    
    # Try to create DataFrame with time index first
    try:
        df = tdms_file.as_dataframe(time_index=True, absolute_time=True)
        has_time_index = True
    except:
        # Fall back to regular DataFrame if time_index fails
        df = tdms_file.as_dataframe()
        has_time_index = False
    
    # Add timestamp and relative time columns
    if not df.empty:
        if has_time_index and hasattr(df.index, 'to_pydatetime'):
            # Has proper datetime index
            df.insert(0, 'Timestamp', df.index)
            start_time = df.index[0]
            df.insert(1, 'Time (s)', (df.index - start_time).total_seconds())
            
            print(f"\nDataFrame shape: {df.shape}")
            print(f"Columns: {len(df.columns)}")
            print(f"Rows: {len(df)}")
            print(f"\nTime range:")
            print(f"  Start: {df.index[0]}")
            print(f"  End: {df.index[-1]}")
            print(f"  Duration: {df.index[-1] - df.index[0]}")
            print(f"\nFirst timestamp: {df['Timestamp'].iloc[0]}")
            print(f"Last timestamp: {df['Timestamp'].iloc[-1]}")
            print(f"Time range: 0.0 s to {df['Time (s)'].iloc[-1]:.3f} s")
        else:
            # No time index - create synthetic time from sampling rate
            print(f"\nDataFrame shape: {df.shape}")
            print(f"Columns: {len(df.columns)}")
            print(f"Rows: {len(df)}")
            print("\nNo time index in TDMS file. Attempting to extract timing info...")
            
            # Try to get timing info from channel properties
            sample_rate = None
            start_time = None
            
            for group in groups:
                for channel in group.channels():
                    props = channel.properties
                    # Check for waveform timing properties
                    if 'wf_increment' in props:
                        sample_rate = 1.0 / float(props['wf_increment'])
                        print(f"  Found sample rate: {sample_rate} Hz")
                    if 'wf_start_time' in props:
                        start_time = props['wf_start_time']
                        print(f"  Found start time: {start_time}")
                    if sample_rate:
                        break
                if sample_rate:
                    break
            
            # Create time columns
            if sample_rate:
                time_seconds = [i / sample_rate for i in range(len(df))]
                df.insert(0, 'Time (s)', time_seconds)
                if start_time:
                    timestamps = pd.to_datetime(start_time) + pd.to_timedelta(time_seconds, unit='s')
                    df.insert(0, 'Timestamp', timestamps)
                    print(f"\nCreated Timestamp and Time (s) columns")
                    print(f"Time range: 0.0 s to {time_seconds[-1]:.3f} s")
                else:
                    print(f"\nCreated Time (s) column (no absolute timestamp available)")
                    print(f"Time range: 0.0 s to {time_seconds[-1]:.3f} s")
            else:
                # Last resort - just use sample index
                df.insert(0, 'Time (s)', [i for i in range(len(df))])
                print(f"\nWarning: Could not determine sample rate.")
                print(f"Created Time (s) as sample index (0, 1, 2, ...)")
    else:
        print(f"\nDataFrame is empty!")
    
    # Print column names
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Print head of DataFrame
    print("\n" + "=" * 60)
    print("FIRST 10 ROWS OF DATA")
    print("=" * 60)
    print(df.head(10))
    
    # Print basic statistics (excluding time columns)
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    stats_cols = [col for col in df.columns if col not in ['Timestamp', 'Time (s)']]
    if stats_cols:
        print(df[stats_cols].describe())
    else:
        print(df.describe())
    
    return df


def main():
    # Create a hidden root window for the file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    # Open file dialog to select TDMS file
    print("Opening file dialog...")
    file_path = filedialog.askopenfilename(
        title="Select TDMS file",
        filetypes=[
            ("TDMS files", "*.tdms"),
            ("All files", "*.*")
        ],
        initialdir=Path.cwd() / "logs"  # Start in logs folder if it exists
    )
    
    root.destroy()  # Clean up the hidden window
    
    # Check if user canceled
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)
    
    file_path = Path(file_path)
    
    try:
        df = read_tdms_file(file_path)
        
        # Optionally save to CSV
        print(f"\n\nWant to export to CSV? (y/n): ", end='')
        response = input().strip().lower()
        if response == 'y':
            csv_path = file_path.with_suffix('.csv')
            df.to_csv(csv_path)
            print(f"Exported to: {csv_path}")
            
    except Exception as e:
        print(f"\nError reading TDMS file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()