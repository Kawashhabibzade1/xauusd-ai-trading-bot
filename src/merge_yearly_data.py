"""
Optional helper for merging multiple yearly CSV files into one raw dataset.

This is no longer the primary pipeline entrypoint. The default repo flow uses
`data/xauusd_m1_2022_2025.csv` directly.
"""

import pandas as pd
import os
from datetime import datetime

def merge_yearly_files():
    """
    Merge separate yearly CSV files into one master file.
    """
    
    print("=" * 70)
    print("MERGE YEARLY XAUUSD DATA FILES")
    print("=" * 70)
    print()
    
    # Define input directory and files
    data_dir = 'data/raw'
    year_files = [
        'XAUUSD_1m_2023.csv',
        'XAUUSD_1m_2024.csv',
        'XAUUSD_1m_2025.csv'
    ]
    
    # Check which files exist
    print("🔍 Checking for yearly files...")
    available_files = []
    for filename in year_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✓ {filename} ({size_mb:.1f} MB)")
            available_files.append(filepath)
        else:
            print(f"   ✗ {filename} - NOT FOUND")
    
    if not available_files:
        print("\n❌ No yearly files found in data/raw/")
        print("\nExpected files:")
        for f in year_files:
            print(f"   - {f}")
        return False
    
    print(f"\n📊 Found {len(available_files)} file(s) to merge")
    print()
    
    # Load and merge files
    print("📥 Loading data files...")
    dfs = []
    total_rows = 0
    
    for filepath in available_files:
        filename = os.path.basename(filepath)
        print(f"\n   Loading: {filename}")
        
        try:
            df = pd.read_csv(filepath)
            
            # Show info
            print(f"      Rows: {len(df):,}")
            print(f"      Columns: {list(df.columns)}")
            
            # Parse datetime if present
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                print(f"      Date range: {df['time'].min()} → {df['time'].max()}")
            
            dfs.append(df)
            total_rows += len(df)
            print(f"      ✓ Loaded successfully")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            return False
    
    # Concatenate all dataframes
    print("\n🔗 Merging files...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    print(f"   Combined rows: {len(merged_df):,}")
    
    # Sort by time
    if 'time' in merged_df.columns:
        print("   Sorting by time...")
        merged_df = merged_df.sort_values('time')
        merged_df = merged_df.reset_index(drop=True)
    
    # Remove duplicates
    print("   Removing duplicates...")
    before_dedup = len(merged_df)
    if 'time' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['time'], keep='first')
    duplicates_removed = before_dedup - len(merged_df)
    print(f"      Removed: {duplicates_removed:,} duplicate rows")
    
    # Validate data quality
    print("\n✅ Data Quality Checks:")
    
    # Check for missing values
    missing = merged_df.isnull().sum()
    if missing.sum() > 0:
        print(f"   ⚠️  Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"      {col}: {count}")
    else:
        print(f"   ✓ No missing values")
    
    # Check OHLC consistency
    if all(col in merged_df.columns for col in ['open', 'high', 'low', 'close']):
        high_valid = (merged_df['high'] >= merged_df['open']).all() and \
                     (merged_df['high'] >= merged_df['close']).all()
        low_valid = (merged_df['low'] <= merged_df['open']).all() and \
                    (merged_df['low'] <= merged_df['close']).all()
        
        if high_valid and low_valid:
            print(f"   ✓ OHLC consistency: PASS")
        else:
            print(f"   ⚠️  OHLC consistency: FAIL (check data)")
    
    # Save merged file
    output_file = os.path.join(data_dir, 'xauusd_m1_2023_2025_merged.csv')
    print(f"\n💾 Saving merged file...")
    print(f"   Output: {output_file}")
    
    merged_df.to_csv(output_file, index=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"   File size: {file_size_mb:.1f} MB")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ MERGE COMPLETE!")
    print("=" * 70)
    print(f"\n📊 SUMMARY:")
    print(f"   Input files: {len(available_files)}")
    print(f"   Total rows: {len(merged_df):,}")
    if 'time' in merged_df.columns:
        print(f"   Date range: {merged_df['time'].min().date()} → {merged_df['time'].max().date()}")
        days = (merged_df['time'].max() - merged_df['time'].min()).days
        print(f"   Days covered: {days}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {file_size_mb:.1f} MB")
    
    # Show sample
    print(f"\n📋 Sample data (first 5 rows):")
    print(merged_df.head())
    
    print("\n🎯 Next Steps:")
    print("   1. Run: python validate_merged_data.py")
    print("   2. Run: python filter_overlap.py")
    print("   3. Continue to feature engineering")
    
    return True

if __name__ == "__main__":
    success = merge_yearly_files()
    
    if not success:
        print("\n💡 Troubleshooting:")
        print("   - Make sure files are in data/raw/ folder")
        print("   - Check file names match: XAUUSD_1m_2023.csv, etc.")
        print("   - Verify CSV format is correct")
