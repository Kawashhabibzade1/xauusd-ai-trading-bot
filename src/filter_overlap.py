"""
Filter standardized XAUUSD data for the London-NY overlap window (13:00-16:59 UTC).
"""

from __future__ import annotations

import argparse

import pandas as pd

from pipeline_contract import (
    DEFAULT_OVERLAP_OUTPUT,
    DEFAULT_STANDARDIZED_OUTPUT,
    display_path,
    ensure_parent_dir,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_STANDARDIZED_OUTPUT, help="Standardized OHLCV CSV input.")
    parser.add_argument("--output", default=DEFAULT_OVERLAP_OUTPUT, help="Filtered overlap CSV output.")
    return parser.parse_args()


def filter_overlap_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame["hour"] = frame["time"].dt.hour
    frame["dayofweek"] = frame["time"].dt.dayofweek

    overlap = frame.loc[
        frame["dayofweek"].lt(5) & frame["hour"].between(13, 16, inclusive="both"),
        ["time", "open", "high", "low", "close", "volume"],
    ].copy()

    if overlap.empty:
        raise ValueError("Overlap filter removed all rows; input or time parsing is wrong.")

    hours = sorted(overlap["time"].dt.hour.unique().tolist())
    if hours != [13, 14, 15, 16]:
        raise ValueError(f"Filtered output contains unexpected UTC hours: {hours}")
    if overlap["time"].dt.dayofweek.ge(5).any():
        raise ValueError("Filtered output still contains weekend rows.")

    return overlap.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("FILTER LONDON-NY OVERLAP")
    print("=" * 70)
    print(f"Input : {display_path(args.input)}")
    print(f"Output: {display_path(args.output)}")
    print()

    input_path = resolve_repo_path(args.input)
    df = pd.read_csv(input_path)
    overlap = filter_overlap_frame(df)

    bars_per_day = overlap.groupby(overlap["time"].dt.date).size()
    representative = int(bars_per_day.mode().iloc[0]) if not bars_per_day.empty else 0

    output_path = ensure_parent_dir(args.output)
    overlap.to_csv(output_path, index=False)

    print(f"Rows              : {len(overlap):,}")
    print(f"Date range        : {overlap['time'].min()} -> {overlap['time'].max()}")
    print(f"Representative day: {representative} bars")
    print(f"Saved             : {display_path(output_path)}")


if __name__ == "__main__":
    main()
