"""
Filter standardized XAUUSD data for trading windows.
"""

from __future__ import annotations

import argparse
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline_contract import (
    DEFAULT_OVERLAP_OUTPUT,
    DEFAULT_STANDARDIZED_OUTPUT,
    display_path,
    ensure_parent_dir,
    resolve_repo_path,
)

UTC = ZoneInfo("UTC")
DEFAULT_HUNT_TIMEZONE = "Europe/Berlin"
DEFAULT_HUNT_WINDOWS = [
    {"name": "Morning Hunt", "start": "07:45", "end": "13:00", "max_trades": 2},
    {"name": "Afternoon Hunt", "start": "15:00", "end": "17:00", "max_trades": 2},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_STANDARDIZED_OUTPUT, help="Standardized OHLCV CSV input.")
    parser.add_argument("--output", default=DEFAULT_OVERLAP_OUTPUT, help="Filtered overlap CSV output.")
    return parser.parse_args()


def _time_to_minutes(value: str) -> int:
    hour_text, minute_text = str(value).strip().split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid time value: {value}")
    return hour * 60 + minute


def normalize_hunt_windows(windows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw_window in enumerate(windows or DEFAULT_HUNT_WINDOWS):
        start = str(raw_window.get("start", "")).strip()
        end = str(raw_window.get("end", "")).strip()
        if not start or not end:
            raise ValueError(f"Hunt window {index} is missing start or end.")
        normalized.append(
            {
                "name": str(raw_window.get("name", f"Hunt {index + 1}")).strip() or f"Hunt {index + 1}",
                "start": start,
                "end": end,
                "start_minute": _time_to_minutes(start),
                "end_minute": _time_to_minutes(end),
                "max_trades": int(raw_window.get("max_trades", 0)),
            }
        )
    return normalized


def annotate_hunt_windows(
    frame: pd.DataFrame,
    timezone_name: str = DEFAULT_HUNT_TIMEZONE,
    windows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["time"] = pd.to_datetime(annotated["time"])
    if getattr(annotated["time"].dt, "tz", None) is None:
        local_times = annotated["time"].dt.tz_localize(UTC).dt.tz_convert(timezone_name)
    else:
        local_times = annotated["time"].dt.tz_convert(timezone_name)

    annotated["local_time"] = local_times
    annotated["local_dayofweek"] = local_times.dt.dayofweek
    annotated["local_minutes"] = local_times.dt.hour * 60 + local_times.dt.minute
    annotated["hunt_window_name"] = ""
    annotated["hunt_window_allowed"] = False
    annotated["hunt_window_trade_limit"] = 0

    normalized_windows = normalize_hunt_windows(windows)
    weekday_mask = annotated["local_dayofweek"].lt(5)
    for window in normalized_windows:
        window_mask = (
            weekday_mask
            & annotated["local_minutes"].between(window["start_minute"], window["end_minute"], inclusive="both")
        )
        annotated.loc[window_mask, "hunt_window_name"] = window["name"]
        annotated.loc[window_mask, "hunt_window_allowed"] = True
        annotated.loc[window_mask, "hunt_window_trade_limit"] = window["max_trades"]

    return annotated


def filter_hunt_windows_frame(
    df: pd.DataFrame,
    timezone_name: str = DEFAULT_HUNT_TIMEZONE,
    windows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    annotated = annotate_hunt_windows(df, timezone_name=timezone_name, windows=windows)
    filtered = annotated.loc[
        annotated["hunt_window_allowed"],
        ["time", "open", "high", "low", "close", "volume"],
    ].copy()
    if filtered.empty:
        raise ValueError("Hunt-window filter removed all rows; input or local time-window config is wrong.")
    return filtered.reset_index(drop=True)


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
