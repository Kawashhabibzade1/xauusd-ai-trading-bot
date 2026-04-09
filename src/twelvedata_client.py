"""
Minimal Twelve Data client helpers for the local dashboard.
"""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from env_utils import resolve_env_value
from pipeline_contract import display_path, ensure_parent_dir, resolve_repo_path


DEFAULT_TWELVEDATA_ENV = "TWELVEDATA_API_KEY"
BASE_URL = "https://api.twelvedata.com/time_series"
def resolve_api_key(env_name: str = DEFAULT_TWELVEDATA_ENV) -> str | None:
    return resolve_env_value(env_name)


def fetch_time_series(
    api_key: str,
    symbol: str = "XAU/USD",
    interval: str = "1min",
    outputsize: int = 30,
    timezone: str = "UTC",
    order: str = "asc",
) -> dict:
    params = urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "timezone": timezone,
            "order": order,
            "apikey": api_key,
        }
    )
    request = Request(
        f"{BASE_URL}?{params}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("status") != "ok":
        message = payload.get("message", "Unknown Twelve Data error")
        raise RuntimeError(f"Twelve Data request failed: {message}")

    values = payload.get("values") or []
    if not values:
        raise RuntimeError("Twelve Data returned no values for the requested symbol.")

    normalized_values = []
    has_volume = any("volume" in value and value.get("volume") not in (None, "") for value in values)
    for value in values:
        normalized_values.append(
            {
                "datetime": value["datetime"],
                "open": float(value["open"]),
                "high": float(value["high"]),
                "low": float(value["low"]),
                "close": float(value["close"]),
                "volume": float(value["volume"]) if "volume" in value and value.get("volume") not in (None, "") else None,
            }
        )

    return {
        "meta": payload.get("meta", {}),
        "values": normalized_values,
        "has_volume": has_volume,
    }


def synthesize_volume(frame: pd.DataFrame, mode: str = "constant") -> pd.Series:
    if mode == "constant":
        return pd.Series(1000.0, index=frame.index, dtype="float64")

    if mode == "range_proxy":
        true_range = (frame["high"] - frame["low"]).abs()
        baseline = true_range.rolling(30, min_periods=1).median().replace(0, np.nan)
        proxy = (true_range / baseline).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        proxy = proxy.clip(lower=0.25, upper=4.0) * 1000.0
        return proxy.astype("float64")

    raise ValueError(f"Unsupported Twelve Data volume mode: {mode}")


def build_ohlcv_frame(time_series: dict, volume_mode: str = "constant") -> tuple[pd.DataFrame, dict]:
    values = time_series["values"]
    frame = pd.DataFrame(values).rename(columns={"datetime": "time"})
    frame = frame.loc[:, ["time", "open", "high", "low", "close", "volume"]].copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    synthesized = False
    if time_series["has_volume"]:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    else:
        frame["volume"] = synthesize_volume(frame, mode=volume_mode)
        synthesized = True

    if frame.isna().any().any():
        summary = frame.isna().sum()
        summary = summary[summary > 0].to_dict()
        raise ValueError(f"Twelve Data frame contains nulls after normalization: {summary}")

    frame = frame.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    return frame, {
        "rows": len(frame),
        "start": str(frame["time"].min()),
        "end": str(frame["time"].max()),
        "synthesized_volume": synthesized,
        "volume_mode": volume_mode if synthesized else "source",
    }


def fetch_and_write_time_series_csv(
    api_key: str,
    output_path: str,
    symbol: str = "XAU/USD",
    interval: str = "1min",
    outputsize: int = 5000,
    timezone: str = "UTC",
    order: str = "asc",
    volume_mode: str = "constant",
) -> dict:
    time_series = fetch_time_series(
        api_key=api_key,
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
        timezone=timezone,
        order=order,
    )
    frame, frame_stats = build_ohlcv_frame(time_series, volume_mode=volume_mode)
    output_file = ensure_parent_dir(output_path)
    frame.to_csv(output_file, index=False)
    return {
        "output_path": output_file,
        "output_display": display_path(output_file),
        "symbol": time_series["meta"].get("symbol", symbol),
        "interval": time_series["meta"].get("interval", interval),
        "timezone": timezone,
        "has_source_volume": time_series["has_volume"],
        **frame_stats,
    }
