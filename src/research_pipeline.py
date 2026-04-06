"""
Build the XAUUSD research dataset, train the baseline stack, and prepare Streamlit-ready artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from feature_engineering import compute_feature_frame
from pipeline_contract import (
    BASE_COLUMNS,
    DEFAULT_RESEARCH_BASELINE_DIR,
    DEFAULT_RESEARCH_CONFIG,
    DEFAULT_RESEARCH_ENSEMBLE_DIR,
    DEFAULT_RESEARCH_LABEL_OUTPUT,
    DEFAULT_RESEARCH_NEURAL_DIR,
    DEFAULT_RESEARCH_OVERLAYS_OUTPUT,
    DEFAULT_RESEARCH_PAPER_LEDGER_OUTPUT,
    DEFAULT_RESEARCH_PREDICTIONS_OUTPUT,
    DEFAULT_RESEARCH_READY_OUTPUT,
    DEFAULT_RESEARCH_REPORT_OUTPUT,
    DEFAULT_RESEARCH_STANDARDIZED_OUTPUT,
    display_path,
    ensure_parent_dir,
    resolve_repo_path,
)
from validate_merged_data import load_and_standardize, validate_standardized


CLASS_NAMES = ["SHORT", "HOLD", "LONG"]
CLASS_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_CLASS = {0: -1, 1: 0, 2: 1}
SETUP_NAME_MAP = {-1: "SHORT_SETUP", 0: "NO_TRADE", 1: "LONG_SETUP"}
PRIMARY_CORE_FEATURES = (
    "atr_14",
    "atr_5",
    "rsi_14",
    "ema_12",
    "ema_26",
    "ema_12_slope",
    "ema_26_slope",
    "sma_50",
    "sma_200",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_lower",
    "bb_middle",
    "bb_width",
    "bb_position",
    "stoch_k",
    "stoch_d",
    "swing_high_dist",
    "swing_low_dist",
    "bullish_ob",
    "bearish_ob",
    "fvg_up",
    "fvg_down",
    "fvg_size",
    "liquidity_sweep_high",
    "liquidity_sweep_low",
    "premium_discount",
    "bar_direction",
    "price_change",
    "price_range_norm",
    "hour",
    "minute",
    "dayofweek",
    "minutes_since_london",
    "minutes_since_ny",
    "session_position",
    "atr_percentile",
    "tick_volatility",
    "range_expansion",
    "volatility_regime",
    "true_range",
    "tr_percentile",
    "price_velocity",
    "price_acceleration",
    "returns_1m",
    "returns_5m",
    "returns_15m",
    "momentum",
    "dist_to_high",
    "dist_to_low",
    "h4_bias",
    "in_discount",
    "in_premium",
    "inducement_taken",
    "entry_zone_present",
    "smc_quality_score",
    "asia",
    "london",
    "new_york",
    "overlap",
    "session_allowed",
    "session_open_minutes",
    "killzone_count",
    "impulse_strength",
    "liquidity_proximity",
    "trap_state",
    "continuation_state",
    "exhaustion_state",
    "mean_reversion_risk",
    "time_of_day_bias",
    "session_bias_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Raw research OHLCV or broker CSV input.")
    parser.add_argument("--config", default=DEFAULT_RESEARCH_CONFIG, help="Research pipeline YAML config.")
    parser.add_argument("--standardized-output", default=DEFAULT_RESEARCH_STANDARDIZED_OUTPUT, help="Standardized research CSV output.")
    parser.add_argument("--research-output", default=DEFAULT_RESEARCH_READY_OUTPUT, help="Research-ready feature CSV output.")
    parser.add_argument("--label-output", default=DEFAULT_RESEARCH_LABEL_OUTPUT, help="Research label CSV output.")
    parser.add_argument("--overlays-output", default=DEFAULT_RESEARCH_OVERLAYS_OUTPUT, help="Research overlay JSON output.")
    parser.add_argument("--predictions-output", default=DEFAULT_RESEARCH_PREDICTIONS_OUTPUT, help="Research prediction CSV output.")
    parser.add_argument("--paper-output", default=DEFAULT_RESEARCH_PAPER_LEDGER_OUTPUT, help="Paper trading ledger CSV output.")
    parser.add_argument("--report-output", default=DEFAULT_RESEARCH_REPORT_OUTPUT, help="Research report JSON output.")
    parser.add_argument("--skip-neural", action="store_true", help="Skip the optional PatchTST neural model.")
    return parser.parse_args()


def load_research_config(path_like: str = DEFAULT_RESEARCH_CONFIG) -> dict[str, Any]:
    with resolve_repo_path(path_like).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_broker_history_csv(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=1, engine="python")
    frame = frame.loc[:, [column for column in frame.columns if not str(column).startswith("Unnamed")]].copy()
    frame = frame.rename(
        columns={
            "Date": "time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Change(Pips)": "change_pips",
            "Change(%)": "change_pct",
        }
    )

    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"Broker history CSV is missing required columns: {missing}")

    standardized = frame.loc[:, ["time", "open", "high", "low", "close"]].copy()
    standardized["time"] = pd.to_datetime(standardized["time"], format="%m/%d/%Y %H:%M", errors="coerce", utc=True)
    if standardized["time"].isna().any():
        raise ValueError("Broker history CSV contains unparseable timestamps.")
    standardized["time"] = standardized["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    for column in ("open", "high", "low", "close"):
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    if "volume" in frame.columns:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
    elif config["data"].get("infer_volume_from_change_pips", True) and "change_pips" in frame.columns:
        volume = pd.to_numeric(frame["change_pips"], errors="coerce").abs()
    else:
        volume = (standardized["high"] - standardized["low"]).abs()

    volume = volume.replace(0, np.nan).fillna(volume.median() if volume.notna().any() else 1.0)
    standardized["volume"] = volume.clip(lower=1e-6)
    standardized = standardized.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    return standardized


def load_research_source(input_path_like: str, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = resolve_repo_path(input_path_like)
    if not path.exists():
        raise SystemExit(f"Research input not found: {display_path(path)}")

    try:
        standardized = load_and_standardize(input_path_like)
        source_info = {
            "input_path": display_path(path),
            "source_kind": "canonical_ohlcv",
            "volume_kind": "source_volume",
        }
    except Exception:
        standardized = parse_broker_history_csv(path, config)
        source_info = {
            "input_path": display_path(path),
            "source_kind": "broker_history_fallback",
            "volume_kind": "change_pips_proxy",
        }

    stats = validate_standardized(standardized)
    source_info.update(
        {
            "rows": int(stats["rows"]),
            "start": str(stats["start"]),
            "end": str(stats["end"]),
            "gaps": int(stats["gaps"]),
        }
    )
    return standardized, source_info


def session_label(hour: int, config: dict[str, Any]) -> str:
    sessions = config["sessions"]
    if hour in sessions["overlap_hours"]:
        return "Overlap"
    if hour in sessions["new_york_hours"]:
        return "New York"
    if hour in sessions["london_hours"]:
        return "London"
    if hour in sessions["asia_hours"]:
        return "Asia"
    return "Off Hours"


def add_session_and_psychology_columns(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["time"] = pd.to_datetime(enriched["time"])
    hour = enriched["time"].dt.hour
    minute = enriched["time"].dt.minute
    sessions = config["sessions"]

    enriched["asia"] = hour.isin(sessions["asia_hours"]).astype(int)
    enriched["london"] = hour.isin(sessions["london_hours"]).astype(int)
    enriched["new_york"] = hour.isin(sessions["new_york_hours"]).astype(int)
    enriched["overlap"] = hour.isin(sessions["overlap_hours"]).astype(int)
    enriched["session_name"] = hour.map(lambda value: session_label(int(value), config))
    enriched["session_allowed"] = enriched["session_name"].isin(allowed_sessions(config)).astype(int)
    enriched["session_open_minutes"] = np.where(
        enriched["session_name"].eq("Asia"),
        hour * 60 + minute - sessions["killzones"]["asia_open"][0] * 60,
        np.where(
            enriched["session_name"].eq("London"),
            hour * 60 + minute - sessions["killzones"]["london_open"][0] * 60,
            np.where(
                enriched["session_name"].eq("New York"),
                hour * 60 + minute - sessions["killzones"]["new_york_open"][0] * 60,
                -1,
            ),
        ),
    )

    killzone_parts = []
    killzone_parts.append(np.where(hour.between(*sessions["killzones"]["asia_open"], inclusive="left"), "ASIA_KZ", ""))
    killzone_parts.append(np.where(hour.between(*sessions["killzones"]["london_open"], inclusive="left"), "LONDON_KZ", ""))
    killzone_parts.append(np.where(hour.between(*sessions["killzones"]["new_york_open"], inclusive="left"), "NY_KZ", ""))
    killzone_frame = pd.DataFrame(killzone_parts).T
    enriched["killzone_flags"] = killzone_frame.apply(lambda row: "|".join(part for part in row if part), axis=1)
    enriched["killzone_count"] = killzone_frame.ne("").sum(axis=1)

    price_range = (enriched["high"] - enriched["low"]).clip(lower=1e-6)
    bullish_body = (enriched["close"] - enriched["open"]).clip(lower=0.0)
    bearish_body = (enriched["open"] - enriched["close"]).clip(lower=0.0)
    enriched["buy_pressure_proxy"] = enriched["volume"] * bullish_body.div(price_range)
    enriched["sell_pressure_proxy"] = enriched["volume"] * bearish_body.div(price_range)
    total_pressure = enriched["buy_pressure_proxy"] + enriched["sell_pressure_proxy"] + 1e-6
    enriched["session_dominance"] = (enriched["buy_pressure_proxy"] - enriched["sell_pressure_proxy"]).div(total_pressure)
    enriched["impulse_strength"] = (enriched["close"] - enriched["open"]).abs().div(enriched["atr_14"] + 1e-6)
    enriched["liquidity_proximity"] = np.minimum(enriched["swing_high_dist"], enriched["swing_low_dist"]).div(enriched["atr_14"] + 1e-6)
    enriched["trap_state"] = (
        ((enriched["liquidity_sweep_high"] == 1) & (enriched["bar_direction"] < 0))
        | ((enriched["liquidity_sweep_low"] == 1) & (enriched["bar_direction"] > 0))
    ).astype(int)
    enriched["continuation_state"] = (
        (enriched["h4_bias"] == enriched["bar_direction"]) & (enriched["range_expansion"] > 1.0)
    ).astype(int)
    enriched["exhaustion_state"] = (
        ((enriched["rsi_14"] >= 70) | (enriched["rsi_14"] <= 30))
        | ((enriched["volume_ratio"] > 1.5) & (enriched["price_range_norm"] < 0.9))
    ).astype(int)
    enriched["mean_reversion_risk"] = (
        (enriched["bb_position"] - 0.5).abs() * 2.0 + (enriched["rsi_14"] - 50.0).abs() / 50.0
    ) / 2.0

    hour_returns = enriched["returns_1m"].shift(1).groupby(hour).expanding().mean().reset_index(level=0, drop=True)
    enriched["time_of_day_bias"] = hour_returns.fillna(0.0)
    enriched["session_bias_score"] = (
        enriched["overlap"] * 0.35 + enriched["london"] * 0.20 + enriched["new_york"] * 0.15 - enriched["asia"] * 0.05
    )

    enriched["regime"] = np.select(
        [
            (enriched["h4_bias"] > 0) & (enriched["volatility_regime"] >= 0.5),
            (enriched["h4_bias"] < 0) & (enriched["volatility_regime"] >= 0.5),
            enriched["volatility_regime"] >= 1.0,
        ],
        ["Trend Up", "Trend Down", "Volatile Range"],
        default="Range",
    )
    return enriched


def build_confluence_tags(frame: pd.DataFrame) -> pd.Series:
    tags: list[str] = []
    for _, row in frame.iterrows():
        parts = [row["session_name"].replace(" ", "_").upper()]
        if row["entry_zone_present"] == 1:
            parts.append("ENTRY_ZONE")
        if row["bullish_ob"] == 1:
            parts.append("BULLISH_OB")
        if row["bearish_ob"] == 1:
            parts.append("BEARISH_OB")
        if row["fvg_up"] > 0:
            parts.append("BULLISH_FVG")
        if row["fvg_down"] > 0:
            parts.append("BEARISH_FVG")
        if row["liquidity_sweep_high"] == 1:
            parts.append("SWEEP_HIGH")
        if row["liquidity_sweep_low"] == 1:
            parts.append("SWEEP_LOW")
        if row["trap_state"] == 1:
            parts.append("TRAP")
        if row["continuation_state"] == 1:
            parts.append("CONTINUATION")
        if row["exhaustion_state"] == 1:
            parts.append("EXHAUSTION")
        tags.append("|".join(parts))
    return pd.Series(tags, index=frame.index, dtype="object")


def allowed_sessions(config: dict[str, Any]) -> set[str]:
    return {str(name) for name in config.get("trading", {}).get("allowed_sessions", ["London", "New York", "Overlap"])}


def class_distribution(values: np.ndarray) -> dict[str, int]:
    return {
        "SHORT": int((values == 0).sum()),
        "HOLD": int((values == 1).sum()),
        "LONG": int((values == 2).sum()),
    }


def directional_signal_from_probs(short_prob: pd.Series, hold_prob: pd.Series, long_prob: pd.Series) -> np.ndarray:
    return np.select(
        [
            (long_prob > short_prob) & (long_prob >= hold_prob),
            (short_prob > long_prob) & (short_prob >= hold_prob),
        ],
        [1, -1],
        default=0,
    )


def join_reasons(*reasons: pd.Series) -> pd.Series:
    frame = pd.concat(reasons, axis=1)
    return frame.apply(lambda row: "|".join(part for part in row if isinstance(part, str) and part), axis=1)


def build_research_ready_frame(standardized: pd.DataFrame, feature_config: str, config: dict[str, Any]) -> pd.DataFrame:
    features = compute_feature_frame(standardized, feature_config)
    features = add_session_and_psychology_columns(features, config)
    features["confluence_tags"] = build_confluence_tags(features)
    return features.reset_index(drop=True)


def future_extrema(series: pd.Series, window: int, fn: str) -> pd.Series:
    reversed_series = series.iloc[::-1]
    if fn == "max":
        aggregated = reversed_series.rolling(window, min_periods=1).max()
    else:
        aggregated = reversed_series.rolling(window, min_periods=1).min()
    return aggregated.iloc[::-1].shift(-1)


def evaluate_first_touch(
    highs: np.ndarray,
    lows: np.ndarray,
    entry_prices: np.ndarray,
    stop_distances: np.ndarray,
    tp1_distances: np.ndarray,
    directions: np.ndarray,
    window: int,
) -> np.ndarray:
    quality = np.full(len(entry_prices), np.nan)
    for index in range(len(entry_prices)):
        direction = directions[index]
        if direction == 0:
            quality[index] = 0.0
            continue

        start = index + 1
        end = min(len(entry_prices), index + 1 + window)
        if start >= end:
            continue

        entry = entry_prices[index]
        stop_distance = stop_distances[index]
        tp_distance = tp1_distances[index]
        if direction > 0:
            stop_level = entry - stop_distance
            tp_level = entry + tp_distance
        else:
            stop_level = entry + stop_distance
            tp_level = entry - tp_distance

        result = 0.0
        for pointer in range(start, end):
            if direction > 0:
                hit_stop = lows[pointer] <= stop_level
                hit_tp = highs[pointer] >= tp_level
            else:
                hit_stop = highs[pointer] >= stop_level
                hit_tp = lows[pointer] <= tp_level

            if hit_tp and hit_stop:
                result = -1.0
                break
            if hit_tp:
                result = 1.0
                break
            if hit_stop:
                result = -1.0
                break
        quality[index] = result
    return quality


def create_research_labels(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    labeled = frame.copy()
    labels_cfg = config["labels"]
    effective_cost = labels_cfg["spread_bps"] + labels_cfg["slippage_bps"]

    for horizon in labels_cfg["horizons"]:
        horizon = int(horizon)
        forward_return = labeled["close"].shift(-horizon).div(labeled["close"]) - 1.0
        threshold = float(labels_cfg["return_thresholds"][str(horizon)]) + effective_cost
        labeled[f"forward_return_{horizon}m"] = forward_return
        labeled[f"direction_{horizon}m"] = np.where(
            forward_return > threshold,
            1,
            np.where(forward_return < -threshold, -1, 0),
        )

    labeled["forward_return_15m"] = labeled["forward_return_15m"].astype(float)
    labeled["effective_cost"] = effective_cost

    base_stop_distance = np.maximum(labeled["atr_14"] * labels_cfg["stop_atr_multiple"], labeled["true_range"] + 1e-6)
    labeled["long_stop_distance"] = np.maximum(base_stop_distance, labeled["swing_low_dist"].clip(lower=1e-6))
    labeled["short_stop_distance"] = np.maximum(base_stop_distance, labeled["swing_high_dist"].clip(lower=1e-6))
    labeled["stop_distance"] = base_stop_distance
    labeled["tp1_distance"] = base_stop_distance * labels_cfg["tp1_rr"]
    labeled["tp2_distance"] = base_stop_distance * labels_cfg["tp2_rr"]
    labeled["preferred_direction"] = np.where(
        labeled["h4_bias"] > 0,
        1,
        np.where(labeled["h4_bias"] < 0, -1, np.sign(labeled["bar_direction"])),
    )

    preferred_stop_distance = np.where(
        labeled["preferred_direction"] >= 0,
        labeled["long_stop_distance"],
        labeled["short_stop_distance"],
    )
    setup_quality = evaluate_first_touch(
        highs=labeled["high"].to_numpy(dtype=float),
        lows=labeled["low"].to_numpy(dtype=float),
        entry_prices=labeled["close"].to_numpy(dtype=float),
        stop_distances=preferred_stop_distance.astype(float),
        tp1_distances=labeled["tp1_distance"].to_numpy(dtype=float),
        directions=labeled["preferred_direction"].to_numpy(dtype=float),
        window=int(labels_cfg["entry_quality_window"]),
    )
    long_setup_result = evaluate_first_touch(
        highs=labeled["high"].to_numpy(dtype=float),
        lows=labeled["low"].to_numpy(dtype=float),
        entry_prices=labeled["close"].to_numpy(dtype=float),
        stop_distances=labeled["long_stop_distance"].to_numpy(dtype=float),
        tp1_distances=labeled["tp1_distance"].to_numpy(dtype=float),
        directions=np.ones(len(labeled), dtype=float),
        window=int(labels_cfg["entry_quality_window"]),
    )
    short_setup_result = evaluate_first_touch(
        highs=labeled["high"].to_numpy(dtype=float),
        lows=labeled["low"].to_numpy(dtype=float),
        entry_prices=labeled["close"].to_numpy(dtype=float),
        stop_distances=labeled["short_stop_distance"].to_numpy(dtype=float),
        tp1_distances=labeled["tp1_distance"].to_numpy(dtype=float),
        directions=-np.ones(len(labeled), dtype=float),
        window=int(labels_cfg["entry_quality_window"]),
    )
    labeled["setup_quality"] = setup_quality
    labeled["long_setup_result"] = long_setup_result
    labeled["short_setup_result"] = short_setup_result
    labeled["entry_quality"] = (
        (labeled["smc_quality_score"] / 4.0) * 0.55
        + labeled["entry_zone_present"] * 0.15
        + (labeled["continuation_state"] * 0.10)
        + (labeled["trap_state"] * 0.05)
        + ((labeled["setup_quality"] == 1).astype(float) * 0.15)
    ).clip(0.0, 1.0)
    labeled["trade_setup_label"] = np.select(
        [
            labeled["session_allowed"].eq(1) & labeled["long_setup_result"].eq(1) & labeled["short_setup_result"].ne(1),
            labeled["session_allowed"].eq(1) & labeled["short_setup_result"].eq(1) & labeled["long_setup_result"].ne(1),
        ],
        [1, -1],
        default=0,
    )
    labeled["trade_setup_name"] = labeled["trade_setup_label"].map(SETUP_NAME_MAP)
    labeled["label"] = labeled["trade_setup_label"].astype(float)

    required = [
        "forward_return_5m",
        "forward_return_15m",
        "forward_return_60m",
        "direction_5m",
        "direction_15m",
        "direction_60m",
        "setup_quality",
        "trade_setup_label",
    ]
    labeled = labeled.dropna(subset=required).reset_index(drop=True)
    return labeled


def build_overlay_objects(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    working = frame.copy().reset_index(drop=True)
    rolling_high = working["high"].rolling(20).max().shift(1)
    rolling_low = working["low"].rolling(20).min().shift(1)

    overlays: dict[str, list[dict[str, Any]]] = {
        "fvg_zones": [],
        "bos_events": [],
        "choch_events": [],
        "order_blocks": [],
        "entry_markers": [],
        "liquidity_levels": [],
    }

    for index, row in working.iterrows():
        time_value = str(row["time"])
        if row["fvg_up"] > 0:
            overlays["fvg_zones"].append(
                {
                    "time": time_value,
                    "direction": "bullish",
                    "low": float(row["low"] - row["fvg_up"]),
                    "high": float(row["low"]),
                    "size": float(row["fvg_up"]),
                }
            )
        if row["fvg_down"] > 0:
            overlays["fvg_zones"].append(
                {
                    "time": time_value,
                    "direction": "bearish",
                    "low": float(row["high"]),
                    "high": float(row["high"] + row["fvg_down"]),
                    "size": float(row["fvg_down"]),
                }
            )

        if pd.notna(rolling_high.iloc[index]) and row["close"] > rolling_high.iloc[index]:
            overlays["bos_events"].append(
                {
                    "time": time_value,
                    "direction": "bullish",
                    "level": float(rolling_high.iloc[index]),
                }
            )
        if pd.notna(rolling_low.iloc[index]) and row["close"] < rolling_low.iloc[index]:
            overlays["bos_events"].append(
                {
                    "time": time_value,
                    "direction": "bearish",
                    "level": float(rolling_low.iloc[index]),
                }
            )

        previous_bias = working["h4_bias"].iloc[index - 1] if index > 0 else row["h4_bias"]
        if index > 0 and row["h4_bias"] != previous_bias:
            overlays["choch_events"].append(
                {
                    "time": time_value,
                    "direction": "bullish" if row["h4_bias"] > previous_bias else "bearish",
                    "from_bias": int(previous_bias),
                    "to_bias": int(row["h4_bias"]),
                }
            )

        if row["bullish_ob"] == 1 or row["bearish_ob"] == 1:
            overlays["order_blocks"].append(
                {
                    "time": time_value,
                    "direction": "bullish" if row["bullish_ob"] == 1 else "bearish",
                    "low": float(row["low"]),
                    "high": float(row["high"]),
                }
            )

        if row["entry_zone_present"] == 1:
            overlays["entry_markers"].append(
                {
                    "time": time_value,
                    "price": float(row["close"]),
                    "score": float(row["smc_quality_score"]),
                }
            )

        if row["liquidity_sweep_high"] == 1:
            overlays["liquidity_levels"].append(
                {
                    "time": time_value,
                    "direction": "sell_side",
                    "level": float(row["high"]),
                }
            )
        if row["liquidity_sweep_low"] == 1:
            overlays["liquidity_levels"].append(
                {
                    "time": time_value,
                    "direction": "buy_side",
                    "level": float(row["low"]),
                }
            )

    return overlays


def select_research_feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {
        *BASE_COLUMNS,
        "session_name",
        "killzone_flags",
        "regime",
        "confluence_tags",
        "preferred_direction",
    }
    exclude.update(column for column in frame.columns if column.startswith("forward_return_"))
    exclude.update(column for column in frame.columns if column.startswith("direction_"))
    exclude.update(
        {
            "label",
            "setup_quality",
            "entry_quality",
            "trade_setup_label",
            "trade_setup_name",
            "long_setup_result",
            "short_setup_result",
        }
    )
    feature_columns = [
        column
        for column in PRIMARY_CORE_FEATURES
        if column in frame.columns and column not in exclude and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not feature_columns:
        raise ValueError("No primary core features were available for the research baseline.")
    return feature_columns


def aligned_predict_proba(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([0, 1, 2]))
    aligned = np.zeros((len(X), 3), dtype=float)
    for idx, cls in enumerate(classes):
        aligned[:, int(cls)] = raw[:, idx]
    sums = aligned.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return aligned / sums


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    one_hot = np.eye(3)[y_true]
    return float(np.mean(np.sum((one_hot - proba) ** 2, axis=1)))


def confidence_bins(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    predicted = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    output = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (confidence >= start) & (confidence < end if end < 1.0 else confidence <= end)
        if not mask.any():
            continue
        output.append(
            {
                "bin_start": float(start),
                "bin_end": float(end),
                "count": int(mask.sum()),
                "avg_confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            }
        )
    return output


def fit_calibrated_classifier(X_train: pd.DataFrame | np.ndarray, y_train: np.ndarray, config: dict[str, Any]) -> Any:
    baseline = config["baseline"]
    estimator = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        num_leaves=int(baseline["num_leaves"]),
        learning_rate=float(baseline["learning_rate"]),
        n_estimators=int(baseline["n_estimators"]),
        subsample=float(baseline["subsample"]),
        colsample_bytree=float(baseline["colsample_bytree"]),
        random_state=int(baseline["random_state"]),
        min_child_samples=int(baseline["min_child_samples"]),
        class_weight=baseline.get("class_weight"),
        verbose=-1,
    )

    unique_classes, counts = np.unique(y_train, return_counts=True)
    if len(unique_classes) < 2 or counts.min() < 8 or len(y_train) < 150:
        estimator.fit(X_train, y_train)
        return estimator

    inner_splits = min(3, max(2, len(y_train) // 120))
    try:
        calibrator = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=inner_splits),
        )
        calibrator.fit(X_train, y_train)
        return calibrator
    except Exception:
        estimator.fit(X_train, y_train)
        return estimator


def build_time_splits(n_rows: int, config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    wf = config["walk_forward"]
    splitter = TimeSeriesSplit(n_splits=int(wf["n_splits"]), gap=int(wf["gap"]))
    splits = []
    for train_idx, test_idx in splitter.split(np.arange(n_rows)):
        if len(train_idx) < int(wf["min_train_rows"]):
            continue
        splits.append((train_idx, test_idx))
    if not splits:
        raise ValueError("Not enough rows to build walk-forward splits for the research pipeline.")
    return splits


def directional_trade_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    predicted_actionable = predicted != 1
    return {
        "signal_rate": float(predicted_actionable.mean()),
        "hold_rate": float((predicted == 1).mean()),
        "trade_precision": float(
            precision_score(y_true, predicted, labels=[0, 2], average="macro", zero_division=0)
        ),
        "trade_recall": float(
            recall_score(y_true, predicted, labels=[0, 2], average="macro", zero_division=0)
        ),
    }


def evaluate_session_metrics(result_frame: pd.DataFrame, target_column: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for group_name, subset in result_frame.groupby("session_name"):
        if subset.empty:
            continue
        y_true = subset[target_column].to_numpy(dtype=int) + 1
        predicted = subset[[f"prob_short_{target_column}", f"prob_hold_{target_column}", f"prob_long_{target_column}"]].to_numpy().argmax(axis=1)
        trade_metrics = directional_trade_metrics(y_true, predicted)
        metrics[group_name] = {
            "samples": int(len(subset)),
            "accuracy": float(accuracy_score(y_true, predicted)),
            "signal_rate": trade_metrics["signal_rate"],
            "trade_precision": trade_metrics["trade_precision"],
        }
    return metrics


def train_walk_forward_baseline(frame: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    splits = build_time_splits(len(frame), config)
    horizon_targets = {5: "direction_5m", 15: "trade_setup_label", 60: "direction_60m"}
    predictions = frame[["time", "close", "volume", "session_name", "regime", "confluence_tags", "smc_quality_score", "entry_zone_present", "h4_bias", "atr_14"]].copy()
    report: dict[str, Any] = {"horizons": {}, "walk_forward_splits": len(splits)}

    X = frame[feature_columns].astype(float)

    for horizon, target_column in horizon_targets.items():
        y = frame[target_column].to_numpy(dtype=int) + 1
        oof = np.full((len(frame), 3), np.nan, dtype=float)
        fold_metrics: list[dict[str, Any]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            model = fit_calibrated_classifier(X_train, y[train_idx], config)
            proba = aligned_predict_proba(model, X_test)
            oof[test_idx] = proba

            predicted = proba.argmax(axis=1)
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "samples": int(len(test_idx)),
                    "accuracy": float(accuracy_score(y[test_idx], predicted)),
                    "log_loss": float(log_loss(y[test_idx], proba, labels=[0, 1, 2])),
                    "brier": multiclass_brier_score(y[test_idx], proba),
                }
            )

        valid_mask = ~np.isnan(oof).any(axis=1)
        oof_valid = oof[valid_mask]
        y_valid = y[valid_mask]
        if len(y_valid) == 0:
            raise ValueError(f"No out-of-fold predictions were generated for {target_column}.")
        predicted_valid = oof_valid.argmax(axis=1)
        predicted_full = np.full(len(frame), np.nan, dtype=float)
        predicted_full[valid_mask] = predicted_valid

        predictions[f"prob_short_{horizon}m"] = np.where(valid_mask, oof[:, 0], np.nan)
        predictions[f"prob_hold_{horizon}m"] = np.where(valid_mask, oof[:, 1], np.nan)
        predictions[f"prob_long_{horizon}m"] = np.where(valid_mask, oof[:, 2], np.nan)
        predictions[f"predicted_class_{horizon}m"] = predicted_full

        final_model = fit_calibrated_classifier(X, y, config)
        model_dir = ensure_parent_dir(Path(DEFAULT_RESEARCH_BASELINE_DIR) / f"h{horizon}" / "model.joblib")
        dump(final_model, model_dir)

        predictions[f"prob_short_{target_column}"] = predictions[f"prob_short_{horizon}m"]
        predictions[f"prob_hold_{target_column}"] = predictions[f"prob_hold_{horizon}m"]
        predictions[f"prob_long_{target_column}"] = predictions[f"prob_long_{horizon}m"]

        report["horizons"][str(horizon)] = {
            "target": target_column,
            "fold_metrics": fold_metrics,
            "overall_accuracy": float(accuracy_score(y_valid, predicted_valid)),
            "overall_log_loss": float(log_loss(y_valid, oof_valid, labels=[0, 1, 2])),
            "overall_brier": multiclass_brier_score(y_valid, oof_valid),
            "short_precision": float(precision_score(y_valid, predicted_valid, labels=[0], average="macro", zero_division=0)),
            "hold_precision": float(precision_score(y_valid, predicted_valid, labels=[1], average="macro", zero_division=0)),
            "long_precision": float(precision_score(y_valid, predicted_valid, labels=[2], average="macro", zero_division=0)),
            "long_recall": float(recall_score(y_valid, predicted_valid, labels=[2], average="macro", zero_division=0)),
            "signal_rate": directional_trade_metrics(y_valid, predicted_valid)["signal_rate"],
            "trade_precision": directional_trade_metrics(y_valid, predicted_valid)["trade_precision"],
            "trade_recall": directional_trade_metrics(y_valid, predicted_valid)["trade_recall"],
            "class_distribution_true": class_distribution(y_valid),
            "class_distribution_predicted": class_distribution(predicted_valid),
            "confidence_bins": confidence_bins(y_valid, oof_valid),
            "session_metrics": evaluate_session_metrics(
                pd.concat([predictions.loc[valid_mask].copy(), frame.loc[valid_mask, [target_column]]], axis=1),
                target_column,
            ),
            "prediction_coverage": float(valid_mask.mean()),
            "model_path": display_path(model_dir),
        }

    return predictions, report


def calculate_signal_levels(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    predictions = frame.copy()
    dominant_long_15 = predictions["prob_long_15m"].fillna(0.0)
    dominant_short_15 = predictions["prob_short_15m"].fillna(0.0)
    dominant_hold_15 = predictions["prob_hold_15m"].fillna(0.0)

    bias_15 = directional_signal_from_probs(dominant_short_15, dominant_hold_15, dominant_long_15)
    bias_60 = directional_signal_from_probs(
        predictions["prob_short_60m"].fillna(0.0),
        predictions["prob_hold_60m"].fillna(0.0),
        predictions["prob_long_60m"].fillna(0.0),
    )
    timing_5 = directional_signal_from_probs(
        predictions["prob_short_5m"].fillna(0.0),
        predictions["prob_hold_5m"].fillna(0.0),
        predictions["prob_long_5m"].fillna(0.0),
    )
    direction = np.where((bias_15 != 0) & (bias_60 == bias_15) & (timing_5 != -bias_15), bias_15, 0)

    predictions["signal_5m"] = timing_5
    predictions["signal_15m"] = bias_15
    predictions["signal_60m"] = bias_60
    predictions["signal_class"] = direction
    predictions["signal_name"] = pd.Series(direction).map({-1: "SHORT", 0: "HOLD", 1: "LONG"}).values
    predictions["setup_candidate"] = pd.Series(bias_15).map({-1: "SHORT_SETUP", 0: "NO_TRADE", 1: "LONG_SETUP"}).values
    predictions["session_allowed"] = predictions["session_name"].isin(allowed_sessions(config)).astype(int)

    predictions["entry_price"] = predictions["open"].shift(-1).fillna(predictions["close"])
    stop_distance = np.where(
        direction >= 0,
        np.maximum(predictions["long_stop_distance"], predictions["atr_14"] * float(config["labels"]["stop_atr_multiple"])),
        np.maximum(predictions["short_stop_distance"], predictions["atr_14"] * float(config["labels"]["stop_atr_multiple"])),
    )
    predictions["planned_stop_distance"] = stop_distance
    predictions["stop_loss"] = np.where(direction >= 0, predictions["entry_price"] - stop_distance, predictions["entry_price"] + stop_distance)
    predictions["tp1"] = np.where(
        direction >= 0,
        predictions["entry_price"] + stop_distance * float(config["labels"]["tp1_rr"]),
        predictions["entry_price"] - stop_distance * float(config["labels"]["tp1_rr"]),
    )
    predictions["tp2"] = np.where(
        direction >= 0,
        predictions["entry_price"] + stop_distance * float(config["labels"]["tp2_rr"]),
        predictions["entry_price"] - stop_distance * float(config["labels"]["tp2_rr"]),
    )

    session_score = predictions["overlap"] * 1.0 + predictions["london"] * 0.75 + predictions["new_york"] * 0.60
    directional_confidence = np.where(
        direction > 0,
        dominant_long_15,
        np.where(direction < 0, dominant_short_15, np.maximum(dominant_long_15, dominant_short_15)),
    )
    adverse_probability = np.where(
        direction > 0,
        dominant_short_15 + dominant_hold_15 * 0.50,
        np.where(direction < 0, dominant_long_15 + dominant_hold_15 * 0.50, 1.0),
    )
    expected_value = np.where(
        direction != 0,
        directional_confidence * float(config["labels"]["tp1_rr"]) - adverse_probability,
        0.0,
    )
    predictions["directional_confidence"] = directional_confidence
    predictions["expected_value"] = expected_value
    predictions["setup_score"] = (
        directional_confidence * float(config["ensemble"]["setup_score_weights"]["probability"])
        + (predictions["entry_quality"] * float(config["ensemble"]["setup_score_weights"]["confluence"]))
        + (session_score.clip(0, 1) * float(config["ensemble"]["setup_score_weights"]["session"]))
        + (np.maximum(expected_value, 0.0) / max(float(config["labels"]["tp1_rr"]), 1.0) * float(config["ensemble"]["setup_score_weights"]["expected_value"]))
    ).clip(0.0, 1.0)

    no_session = np.where(predictions["session_allowed"].eq(0), "SESSION_BLOCKED", "")
    no_bias = np.where(bias_15 == 0, "MAIN_BIAS_HOLD", "")
    no_confirmation = np.where((bias_15 != 0) & (bias_60 != bias_15), "H60_NOT_CONFIRMED", "")
    bad_timing = np.where((bias_15 != 0) & (timing_5 == -bias_15), "M5_TIMING_OPPOSES", "")
    low_entry = np.where(
        predictions["entry_quality"] < float(config["trading"]["min_entry_quality"]),
        "LOW_ENTRY_QUALITY",
        "",
    )
    low_confidence = np.where(
        directional_confidence < float(config["ensemble"]["confidence_threshold"]),
        "LOW_CONFIDENCE",
        "",
    )
    negative_ev = np.where(expected_value <= 0.0, "NEGATIVE_EV", "")
    low_setup = np.where(
        predictions["setup_score"] < float(config["trading"]["min_setup_score"]),
        "LOW_SETUP_SCORE",
        "",
    )
    predictions["reason_blocked"] = join_reasons(
        pd.Series(no_session, index=predictions.index),
        pd.Series(no_bias, index=predictions.index),
        pd.Series(no_confirmation, index=predictions.index),
        pd.Series(bad_timing, index=predictions.index),
        pd.Series(low_entry, index=predictions.index),
        pd.Series(low_confidence, index=predictions.index),
        pd.Series(negative_ev, index=predictions.index),
        pd.Series(low_setup, index=predictions.index),
    )
    predictions["recommended_trade"] = np.where(predictions["reason_blocked"].eq(""), predictions["signal_name"], "WATCH")
    predictions["trade_setup_name"] = np.where(
        predictions["recommended_trade"].eq("LONG"),
        "LONG_SETUP",
        np.where(predictions["recommended_trade"].eq("SHORT"), "SHORT_SETUP", "NO_TRADE"),
    )
    predictions["gate_status"] = np.where(predictions["recommended_trade"].isin(["LONG", "SHORT"]), "READY", "BLOCKED")
    return predictions


def build_paper_trading_ledger(predictions: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paper_cfg = config["trading"]["paper"]
    working = predictions.sort_values("time").reset_index(drop=True).copy()
    working["paper_status"] = np.where(working["recommended_trade"].isin(["LONG", "SHORT"]), "SIGNAL_READY", "BLOCKED")
    working["paper_reason_blocked"] = working["reason_blocked"].where(working["reason_blocked"].ne(""), "")
    working["paper_trade_id"] = pd.Series([pd.NA] * len(working), dtype="object")
    working["paper_planned_entry_time"] = working["time"].shift(-1).astype(str)

    equity = float(paper_cfg["starting_equity"])
    risk_per_trade = float(paper_cfg["risk_per_trade"])
    max_holding_bars = int(paper_cfg["max_holding_bars"])
    max_trades_per_day = int(paper_cfg["max_trades_per_day"])
    daily_loss_stop = int(paper_cfg["daily_loss_streak_stop"])

    daily_trade_count: dict[str, int] = {}
    daily_loss_streak: dict[str, int] = {}
    open_trade: dict[str, Any] | None = None
    ledger_rows: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    trade_id = 0

    for index in range(len(working)):
        row = working.iloc[index]
        row_day = str(pd.to_datetime(row["time"]).date())
        daily_trade_count.setdefault(row_day, 0)
        daily_loss_streak.setdefault(row_day, 0)

        if open_trade is not None and index >= int(open_trade["entry_index"]):
            hit_stop = False
            hit_tp1 = False
            hit_tp2 = False
            if open_trade["direction"] == "LONG":
                hit_stop = float(row["low"]) <= float(open_trade["stop_price"])
                hit_tp1 = float(row["high"]) >= float(open_trade["tp1"])
                hit_tp2 = float(row["high"]) >= float(open_trade["tp2"])
            else:
                hit_stop = float(row["high"]) >= float(open_trade["stop_price"])
                hit_tp1 = float(row["low"]) <= float(open_trade["tp1"])
                hit_tp2 = float(row["low"]) <= float(open_trade["tp2"])

            if not open_trade["tp1_hit"]:
                if hit_stop:
                    open_trade["realized_r"] = -1.0
                    open_trade["exit_reason"] = "STOP_LOSS"
                    open_trade["exit_price"] = float(open_trade["stop_price"])
                    open_trade["exit_time"] = str(row["time"])
                elif hit_tp1:
                    open_trade["tp1_hit"] = True
                    open_trade["realized_r"] += 0.75
                    open_trade["stop_price"] = float(open_trade["entry_price"])
                    open_trade["tp1_time"] = str(row["time"])

            if open_trade["tp1_hit"] and open_trade.get("exit_reason") is None:
                if hit_stop:
                    open_trade["exit_reason"] = "BREAKEVEN_AFTER_TP1"
                    open_trade["exit_price"] = float(open_trade["entry_price"])
                    open_trade["exit_time"] = str(row["time"])
                elif hit_tp2:
                    open_trade["realized_r"] += 1.25
                    open_trade["tp2_hit"] = True
                    open_trade["exit_reason"] = "TP2"
                    open_trade["exit_price"] = float(open_trade["tp2"])
                    open_trade["exit_time"] = str(row["time"])

            bars_open = index - int(open_trade["entry_index"]) + 1
            if open_trade.get("exit_reason") is None and bars_open >= max_holding_bars:
                if open_trade["direction"] == "LONG":
                    move_r = (float(row["close"]) - float(open_trade["entry_price"])) / float(open_trade["stop_distance"])
                else:
                    move_r = (float(open_trade["entry_price"]) - float(row["close"])) / float(open_trade["stop_distance"])
                if open_trade["tp1_hit"]:
                    open_trade["realized_r"] += 0.5 * move_r
                else:
                    open_trade["realized_r"] = move_r
                open_trade["exit_reason"] = "TIME_EXIT"
                open_trade["exit_price"] = float(row["close"])
                open_trade["exit_time"] = str(row["time"])

            if open_trade.get("exit_reason") is not None:
                risk_cash = float(open_trade["equity_before"]) * risk_per_trade
                pnl_cash = risk_cash * float(open_trade["realized_r"])
                equity_after = float(open_trade["equity_before"]) + pnl_cash
                equity = equity_after
                loss_day = open_trade["signal_day"]
                if float(open_trade["realized_r"]) < 0:
                    daily_loss_streak[loss_day] = daily_loss_streak.get(loss_day, 0) + 1
                else:
                    daily_loss_streak[loss_day] = 0

                ledger_rows.append(
                    {
                        "trade_id": open_trade["trade_id"],
                        "signal_time": open_trade["signal_time"],
                        "entry_time": open_trade["entry_time"],
                        "exit_time": open_trade["exit_time"],
                        "direction": open_trade["direction"],
                        "session_name": open_trade["session_name"],
                        "regime": open_trade["regime"],
                        "entry_price": open_trade["entry_price"],
                        "stop_loss": open_trade["initial_stop_price"],
                        "tp1": open_trade["tp1"],
                        "tp2": open_trade["tp2"],
                        "exit_price": open_trade["exit_price"],
                        "exit_reason": open_trade["exit_reason"],
                        "tp1_hit": bool(open_trade["tp1_hit"]),
                        "tp2_hit": bool(open_trade["tp2_hit"]),
                        "setup_score": open_trade["setup_score"],
                        "expected_value": open_trade["expected_value"],
                        "entry_quality": open_trade["entry_quality"],
                        "realized_r": float(open_trade["realized_r"]),
                        "pnl_cash": float(pnl_cash),
                        "equity_before": float(open_trade["equity_before"]),
                        "equity_after": float(equity_after),
                    }
                )
                equity_curve.append({"time": open_trade["exit_time"], "equity": float(equity_after)})
                open_trade = None

        if row["recommended_trade"] not in ("LONG", "SHORT"):
            continue

        risk_blockers: list[str] = []
        if index + 1 >= len(working):
            risk_blockers.append("NO_NEXT_BAR")
        if open_trade is not None:
            risk_blockers.append("OPEN_TRADE_ACTIVE")
        if daily_trade_count[row_day] >= max_trades_per_day:
            risk_blockers.append("DAILY_TRADE_LIMIT")
        if daily_loss_streak[row_day] >= daily_loss_stop:
            risk_blockers.append("DAILY_LOSS_LIMIT")

        if risk_blockers:
            working.at[index, "paper_status"] = "BLOCKED_BY_RISK"
            existing = str(working.at[index, "paper_reason_blocked"])
            combined = "|".join(part for part in [existing, *risk_blockers] if part)
            working.at[index, "paper_reason_blocked"] = combined
            continue

        trade_id += 1
        next_row = working.iloc[index + 1]
        entry_price = float(next_row["open"])
        stop_price = float(row["stop_loss"])
        tp1 = float(row["tp1"])
        tp2 = float(row["tp2"])
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 1e-8:
            working.at[index, "paper_status"] = "BLOCKED_BY_RISK"
            existing = str(working.at[index, "paper_reason_blocked"])
            working.at[index, "paper_reason_blocked"] = "|".join(part for part in [existing, "INVALID_STOP_DISTANCE"] if part)
            continue

        open_trade = {
            "trade_id": f"T{trade_id:04d}",
            "signal_time": str(row["time"]),
            "signal_day": row_day,
            "entry_time": str(next_row["time"]),
            "entry_index": index + 1,
            "direction": str(row["recommended_trade"]),
            "session_name": str(row["session_name"]),
            "regime": str(row["regime"]),
            "entry_price": entry_price,
            "stop_price": stop_price,
            "initial_stop_price": stop_price,
            "tp1": tp1,
            "tp2": tp2,
            "stop_distance": stop_distance,
            "equity_before": float(equity),
            "realized_r": 0.0,
            "tp1_hit": False,
            "tp2_hit": False,
            "exit_reason": None,
            "exit_price": np.nan,
            "exit_time": None,
            "setup_score": float(row["setup_score"]),
            "expected_value": float(row["expected_value"]),
            "entry_quality": float(row["entry_quality"]),
        }
        daily_trade_count[row_day] += 1
        working.at[index, "paper_status"] = "OPENED"
        working.at[index, "paper_trade_id"] = open_trade["trade_id"]
        working.at[index, "paper_reason_blocked"] = ""

    ledger_columns = [
        "trade_id",
        "signal_time",
        "entry_time",
        "exit_time",
        "direction",
        "session_name",
        "regime",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "exit_price",
        "exit_reason",
        "tp1_hit",
        "tp2_hit",
        "setup_score",
        "expected_value",
        "entry_quality",
        "realized_r",
        "pnl_cash",
        "equity_before",
        "equity_after",
    ]
    ledger = pd.DataFrame(ledger_rows, columns=ledger_columns)
    if ledger.empty:
        summary = {
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "expectancy_cash": 0.0,
            "max_drawdown": 0.0,
            "precision": 0.0,
            "equity_curve": [],
            "session_breakdown": {},
            "blocked_reason_counts": working.loc[working["paper_reason_blocked"].ne(""), "paper_reason_blocked"].value_counts().to_dict(),
            "open_positions": [open_trade] if open_trade is not None else [],
        }
        return working, ledger, summary

    pnl_cash = ledger["pnl_cash"].astype(float)
    realized_r = ledger["realized_r"].astype(float)
    gross_profit = pnl_cash[pnl_cash > 0].sum()
    gross_loss = abs(pnl_cash[pnl_cash < 0].sum())
    equity_series = ledger["equity_after"].astype(float)
    running_peak = equity_series.cummax()
    drawdown = (equity_series - running_peak) / running_peak.replace(0, np.nan)
    summary = {
        "trade_count": int(len(ledger)),
        "win_rate": float((realized_r > 0).mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else 0.0,
        "expectancy_r": float(realized_r.mean()),
        "expectancy_cash": float(pnl_cash.mean()),
        "max_drawdown": float(abs(drawdown.min())) if len(drawdown) else 0.0,
        "precision": float((realized_r > 0).mean()),
        "equity_curve": equity_curve,
        "session_breakdown": ledger["session_name"].value_counts().to_dict(),
        "blocked_reason_counts": working.loc[working["paper_reason_blocked"].ne(""), "paper_reason_blocked"].value_counts().to_dict(),
        "open_positions": [open_trade] if open_trade is not None else [],
    }
    return working, ledger, summary


def build_session_profile(frame: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for session_name, subset in frame.groupby("session_name"):
        if subset.empty:
            continue
        profile[session_name] = {
            "bars": int(len(subset)),
            "avg_return_1m": float(subset["returns_1m"].mean()),
            "avg_volume_ratio": float(subset["volume_ratio"].mean()),
            "avg_session_dominance": float(subset["session_dominance"].mean()),
            "avg_impulse_strength": float(subset["impulse_strength"].mean()),
            "entry_zone_rate": float(subset["entry_zone_present"].mean()),
        }
    return profile


def maybe_run_neural_model(frame: pd.DataFrame, feature_columns: list[str], config: dict[str, Any], skip_neural: bool) -> dict[str, Any]:
    if skip_neural:
        return {"status": "skipped_by_flag", "notes": ["PatchTST training was skipped by CLI flag."]}
    if not config.get("neural", {}).get("enabled", True):
        return {"status": "disabled_in_config", "notes": ["PatchTST training is disabled in the research config."]}

    try:
        from research_neural_patchtst import train_patchtst_research_model
    except Exception as exc:
        return {
            "status": "unavailable",
            "notes": [
                "PatchTST model could not be loaded. Install requirements-research.txt to enable the neural stack.",
                str(exc),
            ],
        }

    try:
        return train_patchtst_research_model(
            frame=frame,
            feature_columns=feature_columns,
            config=config,
            output_dir=str(resolve_repo_path(DEFAULT_RESEARCH_NEURAL_DIR)),
        )
    except Exception as exc:
        return {
            "status": "failed",
            "notes": [
                "PatchTST training failed, so the cockpit stayed on the calibrated baseline.",
                str(exc),
            ],
        }


def blend_predictions(baseline: pd.DataFrame, neural_result: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensemble = baseline.copy()
    notes = ["Ensemble defaults to the calibrated baseline when neural predictions are unavailable."]
    if neural_result.get("status") != "ok":
        return ensemble, {"status": neural_result.get("status", "baseline_only"), "notes": notes + neural_result.get("notes", [])}

    neural_predictions = pd.read_csv(resolve_repo_path(neural_result["predictions_path"]))
    merged = ensemble.merge(neural_predictions, on="time", suffixes=("", "_neural"), how="left")
    baseline_weight = float(config["ensemble"]["baseline_weight"])
    neural_weight = float(config["ensemble"]["neural_weight"])

    for horizon in (5, 15, 60):
        for label in ("short", "hold", "long"):
            base_col = f"prob_{label}_{horizon}m"
            neural_col = f"{base_col}_neural"
            merged[base_col] = (
                merged[base_col].fillna(0.0) * baseline_weight + merged[neural_col].fillna(0.0) * neural_weight
            )
        prob_cols = [f"prob_short_{horizon}m", f"prob_hold_{horizon}m", f"prob_long_{horizon}m"]
        total = merged[prob_cols].sum(axis=1).replace(0, np.nan)
        merged[prob_cols] = merged[prob_cols].div(total, axis=0).fillna(0.0)

    notes = ["Ensemble combines calibrated LightGBM baseline with PatchTST probabilities."]
    return merged[ensemble.columns], {"status": "ok", "notes": notes, "predictions_path": neural_result["predictions_path"]}


def build_report(
    source_info: dict[str, Any],
    dataset_summary: dict[str, Any],
    feature_columns: list[str],
    baseline_report: dict[str, Any],
    neural_report: dict[str, Any],
    ensemble_report: dict[str, Any],
    latest_signal: dict[str, Any],
    session_profile: dict[str, Any],
    overlays: dict[str, Any],
    paper_trading: dict[str, Any],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "source": source_info,
        "dataset_summary": dataset_summary,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "baseline": baseline_report,
        "neural": neural_report,
        "ensemble": ensemble_report,
        "latest_signal": latest_signal,
        "session_profile": session_profile,
        "overlay_counts": {key: len(value) for key, value in overlays.items()},
        "paper_trading": paper_trading,
        "backtest": paper_trading,
        "outputs": output_paths,
        "notes": [
            "This research cockpit is signal-plus-paper-trading only and does not auto-trade.",
            "Primary model features are Twelve Data compatible core features; volume-proxy fields remain research-only context.",
            "Trade quality should be judged by precision of emitted signals, expectancy, profit factor, drawdown, and calibration.",
        ],
    }


def run_research_pipeline(
    input_path_like: str,
    config_path: str = DEFAULT_RESEARCH_CONFIG,
    standardized_output: str = DEFAULT_RESEARCH_STANDARDIZED_OUTPUT,
    research_output: str = DEFAULT_RESEARCH_READY_OUTPUT,
    label_output: str = DEFAULT_RESEARCH_LABEL_OUTPUT,
    overlays_output: str = DEFAULT_RESEARCH_OVERLAYS_OUTPUT,
    predictions_output: str = DEFAULT_RESEARCH_PREDICTIONS_OUTPUT,
    paper_output: str = DEFAULT_RESEARCH_PAPER_LEDGER_OUTPUT,
    report_output: str = DEFAULT_RESEARCH_REPORT_OUTPUT,
    skip_neural: bool = False,
) -> dict[str, Any]:
    config = load_research_config(config_path)
    standardized, source_info = load_research_source(input_path_like, config)
    standardized.to_csv(ensure_parent_dir(standardized_output), index=False)

    research_ready = build_research_ready_frame(standardized, feature_config="python_training/config/features.yaml", config=config)
    research_ready.to_csv(ensure_parent_dir(research_output), index=False)

    labels = create_research_labels(research_ready, config)
    labels.to_csv(ensure_parent_dir(label_output), index=False)

    overlays = build_overlay_objects(labels)
    with ensure_parent_dir(overlays_output).open("w", encoding="utf-8") as handle:
        json.dump(overlays, handle, indent=2)

    feature_columns = select_research_feature_columns(labels)
    baseline_predictions, baseline_report = train_walk_forward_baseline(labels, feature_columns, config)
    baseline_predictions = pd.concat(
        [
            labels.reset_index(drop=True),
            baseline_predictions.drop(
                columns=[
                    "time",
                    "close",
                    "volume",
                    "session_name",
                    "regime",
                    "confluence_tags",
                    "smc_quality_score",
                    "entry_zone_present",
                    "h4_bias",
                    "atr_14",
                ]
            ),
        ],
        axis=1,
    )

    neural_report = maybe_run_neural_model(labels, feature_columns, config, skip_neural=skip_neural)
    ensemble_predictions, ensemble_report = blend_predictions(baseline_predictions, neural_report, config)
    ensemble_predictions = calculate_signal_levels(ensemble_predictions, config)
    ensemble_predictions, paper_ledger, paper_trading = build_paper_trading_ledger(ensemble_predictions, config)
    ensemble_predictions.to_csv(ensure_parent_dir(predictions_output), index=False)
    paper_ledger.to_csv(ensure_parent_dir(paper_output), index=False)

    latest_row = ensemble_predictions.iloc[-1]
    latest_signal = {
        "time": str(latest_row["time"]),
        "signal": str(latest_row["recommended_trade"]),
        "signal_5m": int(latest_row["signal_5m"]),
        "signal_15m": int(latest_row["signal_15m"]),
        "signal_60m": int(latest_row["signal_60m"]),
        "session_name": str(latest_row["session_name"]),
        "regime": str(latest_row["regime"]),
        "setup_score": float(latest_row["setup_score"]),
        "expected_value": float(latest_row["expected_value"]),
        "prob_short_5m": float(latest_row["prob_short_5m"]),
        "prob_hold_5m": float(latest_row["prob_hold_5m"]),
        "prob_long_5m": float(latest_row["prob_long_5m"]),
        "prob_short_15m": float(latest_row["prob_short_15m"]),
        "prob_hold_15m": float(latest_row["prob_hold_15m"]),
        "prob_long_15m": float(latest_row["prob_long_15m"]),
        "prob_short_60m": float(latest_row["prob_short_60m"]),
        "prob_hold_60m": float(latest_row["prob_hold_60m"]),
        "prob_long_60m": float(latest_row["prob_long_60m"]),
        "entry_price": float(latest_row["entry_price"]),
        "stop_loss": float(latest_row["stop_loss"]),
        "tp1": float(latest_row["tp1"]),
        "tp2": float(latest_row["tp2"]),
        "reason_blocked": str(latest_row["reason_blocked"]),
        "gate_status": str(latest_row["gate_status"]),
        "paper_status": str(latest_row["paper_status"]),
        "paper_reason_blocked": str(latest_row["paper_reason_blocked"]),
        "confluence_tags": str(latest_row["confluence_tags"]),
    }

    session_profile = build_session_profile(labels)

    output_paths = {
        "standardized_output": display_path(standardized_output),
        "research_output": display_path(research_output),
        "label_output": display_path(label_output),
        "overlays_output": display_path(overlays_output),
        "predictions_output": display_path(predictions_output),
        "paper_output": display_path(paper_output),
        "report_output": display_path(report_output),
    }
    dataset_summary = {
        "standardized_rows": int(len(standardized)),
        "research_rows": int(len(research_ready)),
        "label_rows": int(len(labels)),
        "prediction_rows": int(len(ensemble_predictions)),
        "paper_trade_rows": int(len(paper_ledger)),
        "sessions": labels["session_name"].value_counts().to_dict(),
        "regimes": labels["regime"].value_counts().to_dict(),
        "recommended_trade_counts": ensemble_predictions["recommended_trade"].value_counts().to_dict(),
    }
    report = build_report(
        source_info=source_info,
        dataset_summary=dataset_summary,
        feature_columns=feature_columns,
        baseline_report=baseline_report,
        neural_report=neural_report,
        ensemble_report=ensemble_report,
        latest_signal=latest_signal,
        session_profile=session_profile,
        overlays=overlays,
        paper_trading=paper_trading,
        output_paths=output_paths,
    )
    with ensure_parent_dir(report_output).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def print_research_summary(report: dict[str, Any]) -> None:
    latest = report["latest_signal"]
    print("Research cockpit artifacts generated:")
    print(f"  signal time      : {latest['time']}")
    print(f"  recommended      : {latest['signal']}")
    print(f"  session/regime   : {latest['session_name']} / {latest['regime']}")
    print(f"  setup score      : {latest['setup_score']:.2%}")
    print(f"  expected value   : {latest['expected_value']:.4f}")
    print(f"  paper status     : {latest['paper_status']}")
    if latest.get("reason_blocked"):
        print(f"  blocked by       : {latest['reason_blocked']}")
    print()
    print("Key outputs:")
    for key, value in report["outputs"].items():
        print(f"  {key}: {value}")
    print()
    print("Neural model status:")
    print(f"  {report['neural'].get('status', 'unknown')}")


def main() -> None:
    args = parse_args()
    config = load_research_config(args.config)
    input_path_like = args.input or config["data"]["default_input"]

    print("=" * 70)
    print("XAUUSD RESEARCH COCKPIT PIPELINE")
    print("=" * 70)
    print(f"Input      : {display_path(input_path_like)}")
    print(f"Config     : {display_path(args.config)}")
    print(f"Skip neural: {args.skip_neural}")
    print()

    report = run_research_pipeline(
        input_path_like=input_path_like,
        config_path=args.config,
        standardized_output=args.standardized_output,
        research_output=args.research_output,
        label_output=args.label_output,
        overlays_output=args.overlays_output,
        predictions_output=args.predictions_output,
        paper_output=args.paper_output,
        report_output=args.report_output,
        skip_neural=args.skip_neural,
    )
    print_research_summary(report)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
