"""
Validate and standardize raw XAUUSD CSV data into a canonical UTC OHLCV format.
"""

from __future__ import annotations

import argparse

import pandas as pd

from pipeline_contract import (
    BASE_COLUMNS,
    DEFAULT_RAW_INPUT,
    DEFAULT_STANDARDIZED_OUTPUT,
    display_path,
    ensure_parent_dir,
    resolve_repo_path,
)


LEGACY_COLUMN_MAPPING = {
    "UTC": "time",
    "Gmt time": "time",
    "Date": "time",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_RAW_INPUT, help="Raw CSV input path.")
    parser.add_argument("--output", default=DEFAULT_STANDARDIZED_OUTPUT, help="Standardized CSV output path.")
    return parser.parse_args()


def parse_utc_time_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(" UTC", "", regex=False).str.strip()

    for fmt in ("%d.%m.%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        parsed = pd.to_datetime(cleaned, format=fmt, errors="coerce", utc=True)
        if parsed.notna().all():
            return parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    parsed = pd.to_datetime(cleaned, errors="coerce", utc=True)
    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise ValueError(f"Failed to parse {bad_count} timestamps from raw input.")

    return parsed.dt.tz_convert("UTC").dt.tz_localize(None)


def load_and_standardize(input_path: str) -> pd.DataFrame:
    path = resolve_repo_path(input_path)
    df = pd.read_csv(path)
    df = df.rename(columns=LEGACY_COLUMN_MAPPING)

    missing = [column for column in BASE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input is missing canonical columns after normalization: {missing}")

    standardized = df.loc[:, list(BASE_COLUMNS)].copy()
    standardized["time"] = parse_utc_time_column(standardized["time"])
    standardized = standardized.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

    for column in BASE_COLUMNS[1:]:
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    if standardized.isna().any().any():
        null_summary = standardized.isna().sum()
        null_summary = null_summary[null_summary > 0].to_dict()
        raise ValueError(f"Standardized data contains nulls after parsing: {null_summary}")

    return standardized


def validate_standardized(df: pd.DataFrame) -> dict:
    high_valid = (df["high"] >= df["open"]).all() and (df["high"] >= df["close"]).all()
    low_valid = (df["low"] <= df["open"]).all() and (df["low"] <= df["close"]).all()
    hl_valid = (df["high"] >= df["low"]).all()
    if not (high_valid and low_valid and hl_valid):
        raise ValueError("OHLC validation failed after standardization.")

    time_diff = df["time"].diff().dt.total_seconds().div(60.0)
    gaps = df.loc[time_diff > 1.0, "time"]

    return {
        "rows": len(df),
        "start": df["time"].min(),
        "end": df["time"].max(),
        "zero_volume": int((df["volume"] == 0).sum()),
        "gaps": int(len(gaps)),
        "largest_gap_minutes": float(time_diff.max()) if len(gaps) else 0.0,
    }


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("VALIDATE & STANDARDIZE RAW XAUUSD DATA")
    print("=" * 70)
    print(f"Input : {display_path(args.input)}")
    print(f"Output: {display_path(args.output)}")
    print()

    standardized = load_and_standardize(args.input)
    stats = validate_standardized(standardized)

    output_path = ensure_parent_dir(args.output)
    standardized.to_csv(output_path, index=False)

    print(f"Rows         : {stats['rows']:,}")
    print(f"Date range   : {stats['start']} -> {stats['end']}")
    print(f"Zero volume  : {stats['zero_volume']:,}")
    print(f"Time gaps    : {stats['gaps']:,}")
    if stats["gaps"]:
        print(f"Largest gap  : {stats['largest_gap_minutes']:.0f} minutes")
    print(f"Saved        : {display_path(output_path)}")


if __name__ == "__main__":
    main()
