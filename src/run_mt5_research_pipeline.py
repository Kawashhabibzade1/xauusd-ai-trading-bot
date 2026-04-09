"""
Build a local MT5-backed paper-trading report for the Streamlit research cockpit.

This adapter keeps the UI on a single MT5-local source of truth while avoiding the
much heavier full research retraining loop on every new MT5 bar.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from filter_overlap import annotate_hunt_windows
from mt5_client import (
    fetch_mt5_account_context,
    load_mt5_account_snapshot,
    normalize_mt5_symbol,
    resolve_login,
    resolve_password,
    resolve_server,
    resolve_terminal_path,
    sync_mt5_file_artifacts,
)
from pipeline_contract import (
    DEFAULT_MT5_LIVE_FEATURE_OUTPUT,
    DEFAULT_MT5_LIVE_MT5_VALIDATION_OUTPUT,
    DEFAULT_MT5_LIVE_RAW_INPUT,
    DEFAULT_MT5_LIVE_REPORT_PATH,
    DEFAULT_MT5_RESEARCH_CONFIG,
    DEFAULT_MT5_RESEARCH_FEEDBACK_OUTPUT,
    DEFAULT_MT5_RESEARCH_FINAL_LEDGER_OUTPUT,
    DEFAULT_MT5_RESEARCH_LEARNING_STATUS_OUTPUT,
    DEFAULT_MT5_RESEARCH_MANUAL_OVERRIDE_OUTPUT,
    DEFAULT_MT5_RESEARCH_OVERLAYS_OUTPUT,
    DEFAULT_MT5_RESEARCH_PAPER_LEDGER_OUTPUT,
    DEFAULT_MT5_RESEARCH_PREDICTIONS_OUTPUT,
    DEFAULT_MT5_RESEARCH_REPORT_OUTPUT,
    DEFAULT_MT5_RESEARCH_SNAPSHOT_DIR,
    DEFAULT_MT5_RESEARCH_TRADE_HISTORY_OUTPUT,
    DEFAULT_MT5_TRADE_DIRECTIVE_OUTPUT,
    display_path,
    ensure_parent_dir,
    json_dump,
    resolve_repo_path,
)
from research_pipeline import build_overlay_objects
from run_live_mt5_pipeline import run_live_pipeline


SIGNAL_NAME_MAP = {-1: "SHORT", 0: "HOLD", 1: "LONG"}
SETUP_NAME_MAP = {-1: "SHORT_SETUP", 0: "NO_TRADE", 1: "LONG_SETUP"}
OVERRIDEABLE_RISK_BLOCKERS = {"DAILY_TRADE_LIMIT", "DAILY_LOSS_LIMIT"}
MT5_FILES_ROOT = Path.home() / "Library" / "Application Support" / "net.metaquotes.wine.metatrader5" / "drive_c" / "Program Files" / "MetaTrader 5" / "MQL5" / "Files"
MT5_EXPORTER_LIVE_PATH = MT5_FILES_ROOT / "xauusd_mt5_live.csv"
MT5_BOT_TRADE_LOG_PATH = MT5_FILES_ROOT / "logs" / "xauusd_ai_trades.csv"
MT5_ACCOUNT_SNAPSHOT_PATH = MT5_FILES_ROOT / "config" / "mt5_account_snapshot.csv"


def build_mt5_runtime_health() -> dict[str, Any]:
    now_ts = time.time()

    def inspect(path: Path, stale_after_seconds: int) -> dict[str, Any]:
        if not path.exists():
            return {
                "path": str(path),
                "exists": False,
                "updated_at": "",
                "age_seconds": None,
                "stale": True,
                "status": "missing",
            }
        mtime = path.stat().st_mtime
        age_seconds = max(0, int(now_ts - mtime))
        return {
            "path": str(path),
            "exists": True,
            "updated_at": pd.Timestamp.fromtimestamp(mtime, tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "age_seconds": age_seconds,
            "stale": age_seconds > stale_after_seconds,
            "status": "stale" if age_seconds > stale_after_seconds else "live",
        }

    exporter = inspect(MT5_EXPORTER_LIVE_PATH, stale_after_seconds=30)
    trade_log = inspect(MT5_BOT_TRADE_LOG_PATH, stale_after_seconds=180)
    account_snapshot = inspect(MT5_ACCOUNT_SNAPSHOT_PATH, stale_after_seconds=30)
    return {
        "exporter": exporter,
        "bot_trade_log": trade_log,
        "account_snapshot": account_snapshot,
        "summary": {
            "exporter_live": exporter["status"] == "live",
            "bot_log_live": trade_log["status"] == "live",
            "account_snapshot_live": account_snapshot["status"] == "live",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD", help="MT5 symbol label for reporting.")
    parser.add_argument("--timeframe", default="M1", help="MT5 timeframe label for reporting.")
    parser.add_argument("--count", type=int, default=5000, help="Number of recent MT5 bars to request when bridge mode is used.")
    parser.add_argument(
        "--source-mode",
        choices=["auto", "bridge", "export"],
        default="auto",
        help="Use direct MT5 bridge, exporter CSV, or auto-try bridge then exporter CSV.",
    )
    parser.add_argument("--terminal-env", default="MT5_TERMINAL_PATH", help="Environment variable with the MT5 terminal path.")
    parser.add_argument("--login-env", default="MT5_LOGIN", help="Optional environment variable with the MT5 login.")
    parser.add_argument("--password-env", default="MT5_PASSWORD", help="Optional environment variable with the MT5 password.")
    parser.add_argument("--server-env", default="MT5_SERVER", help="Optional environment variable with the MT5 server.")
    parser.add_argument("--export-input", default="", help="Optional MT5 exporter CSV path.")
    parser.add_argument("--prefer-real-volume", action="store_true", help="Use real_volume when available; otherwise tick_volume is used.")
    parser.add_argument("--config", default=DEFAULT_MT5_RESEARCH_CONFIG, help="Local MT5 paper-trading YAML config.")
    parser.add_argument("--predictions-output", default=DEFAULT_MT5_RESEARCH_PREDICTIONS_OUTPUT, help="MT5 live research-like prediction CSV output.")
    parser.add_argument("--overlays-output", default=DEFAULT_MT5_RESEARCH_OVERLAYS_OUTPUT, help="MT5 live overlay JSON output.")
    parser.add_argument("--paper-output", default=DEFAULT_MT5_RESEARCH_PAPER_LEDGER_OUTPUT, help="MT5 live paper ledger CSV output.")
    parser.add_argument("--trade-history-output", default=DEFAULT_MT5_RESEARCH_TRADE_HISTORY_OUTPUT, help="Append-only MT5 paper trade history CSV output.")
    parser.add_argument("--final-ledger-output", default=DEFAULT_MT5_RESEARCH_FINAL_LEDGER_OUTPUT, help="Frozen final MT5 paper ledger CSV output.")
    parser.add_argument("--trade-directive-output", default=DEFAULT_MT5_TRADE_DIRECTIVE_OUTPUT, help="MT5 demo trade directive CSV output.")
    parser.add_argument("--snapshot-dir", default=DEFAULT_MT5_RESEARCH_SNAPSHOT_DIR, help="Immutable MT5 research snapshot directory.")
    parser.add_argument("--feedback-output", default=DEFAULT_MT5_RESEARCH_FEEDBACK_OUTPUT, help="Learning feedback CSV output.")
    parser.add_argument("--learning-status-output", default=DEFAULT_MT5_RESEARCH_LEARNING_STATUS_OUTPUT, help="Learning readiness JSON output.")
    parser.add_argument("--manual-override-output", default=DEFAULT_MT5_RESEARCH_MANUAL_OVERRIDE_OUTPUT, help="One-time paper-only override JSON state path.")
    parser.add_argument("--report-output", default=DEFAULT_MT5_RESEARCH_REPORT_OUTPUT, help="MT5 live cockpit report JSON output.")
    parser.add_argument("--skip-onnx-runtime-check", action="store_true", help="Skip optional onnxruntime verification.")
    parser.add_argument("--skip-confidence-analysis", action="store_true", help="Skip optional confidence analysis in the fast live path.")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip the approximate backtest in the fast live path.")
    return parser.parse_args(argv)


def load_config(path_like: str) -> dict[str, Any]:
    with resolve_repo_path(path_like).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_live_account_context(
    *,
    symbol: str,
    terminal_env: str,
    login_env: str,
    password_env: str,
    server_env: str,
) -> dict[str, Any] | None:
    try:
        context = fetch_mt5_account_context(
            symbol=symbol,
            terminal_path=resolve_terminal_path(terminal_env),
            login=resolve_login(login_env),
            password=resolve_password(password_env),
            server=resolve_server(server_env),
        )
    except Exception:
        context = None

    if context is None:
        try:
            context = load_mt5_account_snapshot()
        except Exception:
            return None

    if float(context.get("balance", 0.0) or 0.0) <= 0.0:
        return None
    return context


def drop_latest_live_bar(
    frame: pd.DataFrame,
    *,
    time_column: str = "time",
) -> tuple[pd.DataFrame, str]:
    if frame.empty or time_column not in frame.columns:
        return frame.copy(), ""

    working = frame.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working = working.dropna(subset=[time_column]).sort_values(time_column).drop_duplicates(subset=time_column).reset_index(drop=True)
    if len(working) <= 1:
        return working, ""

    dropped_time = working[time_column].iloc[-1]
    trimmed = working.iloc[:-1].copy().reset_index(drop=True)
    return trimmed, (str(dropped_time) if pd.notna(dropped_time) else "")


def write_csv_atomically(frame: pd.DataFrame, output_path: str | Path, **kwargs: Any) -> Path:
    output_file = ensure_parent_dir(output_path)
    temp_output = output_file.with_name(f".{output_file.name}.tmp-{os.getpid()}-{pd.Timestamp.utcnow().value}")
    try:
        frame.to_csv(temp_output, **kwargs)
        os.replace(temp_output, output_file)
    finally:
        if temp_output.exists():
            temp_output.unlink()
    return output_file


def session_name_from_hour(hour_value: int) -> str:
    if 13 <= hour_value <= 16:
        return "Overlap"
    if 7 <= hour_value <= 12:
        return "London"
    if 13 <= hour_value <= 20:
        return "New York"
    if 0 <= hour_value <= 7:
        return "Asia"
    return "Off Hours"


def directional_signal(short_prob: float, hold_prob: float, long_prob: float, threshold: float) -> int:
    values = {"SHORT": float(short_prob), "HOLD": float(hold_prob), "LONG": float(long_prob)}
    label = max(values, key=values.get)
    if values[label] < threshold or label == "HOLD":
        return 0
    return 1 if label == "LONG" else -1


def join_reasons(*parts: str) -> str:
    return "|".join(part for part in parts if part)


def split_reasons(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part for part in text.split("|") if part]


def load_manual_override(path_like: str) -> dict[str, Any]:
    path = resolve_repo_path(path_like)
    payload: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    used_signal_times = [
        str(value)
        for value in payload.get("used_signal_times", [])
        if str(value).strip()
    ]
    allowed_risk_blockers = [
        str(value)
        for value in payload.get("allowed_risk_blockers", sorted(OVERRIDEABLE_RISK_BLOCKERS))
        if str(value).strip()
    ]
    remaining_credits = max(0, int(payload.get("remaining_credits", 0)))
    initial_credits = max(remaining_credits, int(payload.get("initial_credits", remaining_credits)))
    return {
        "enabled": bool(payload.get("enabled", remaining_credits > 0)),
        "scope": str(payload.get("scope", "paper_only")),
        "note": str(payload.get("note", "")),
        "initial_credits": initial_credits,
        "remaining_credits": remaining_credits,
        "used_signal_times": used_signal_times,
        "allowed_risk_blockers": allowed_risk_blockers,
        "start_after": str(payload.get("start_after", "")).strip(),
        "last_used_at": str(payload.get("last_used_at", "")).strip(),
    }


def maybe_apply_manual_override(
    signal_time: pd.Timestamp,
    risk_blockers: list[str],
    manual_override: dict[str, Any],
) -> tuple[bool, str]:
    if not risk_blockers or not manual_override.get("enabled", False):
        return False, ""

    allowed = set(str(value) for value in manual_override.get("allowed_risk_blockers", []))
    if not set(risk_blockers).issubset(allowed):
        return False, ""

    start_after_raw = str(manual_override.get("start_after", "")).strip()
    if start_after_raw:
        start_after = pd.to_datetime(start_after_raw, errors="coerce")
        if pd.notna(start_after) and signal_time <= start_after:
            return False, ""

    signal_key = str(signal_time)
    used_signal_times = set(str(value) for value in manual_override.get("used_signal_times", []))
    if signal_key in used_signal_times:
        return True, "|".join(risk_blockers)

    remaining_credits = int(manual_override.get("remaining_credits", 0))
    if remaining_credits <= 0:
        return False, ""

    manual_override["remaining_credits"] = remaining_credits - 1
    manual_override.setdefault("used_signal_times", []).append(signal_key)
    manual_override["last_used_at"] = signal_key
    return True, "|".join(risk_blockers)


def _normalize_triplet(short_prob: np.ndarray, hold_prob: np.ndarray, long_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.vstack([short_prob, hold_prob, long_prob]).astype(float)
    stacked = np.clip(stacked, 0.0, None)
    totals = stacked.sum(axis=0)
    totals[totals == 0.0] = 1.0
    normalized = stacked / totals
    return normalized[0], normalized[1], normalized[2]


def build_proxy_probabilities(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    timing_score = np.tanh(frame["price_velocity"].fillna(0.0).to_numpy(dtype=float) / 2.0)
    timing_score += 0.35 * frame["bar_direction"].fillna(0.0).to_numpy(dtype=float)
    timing_score += 0.10 * (frame["entry_zone_present"].fillna(0.0).to_numpy(dtype=float) * 2.0 - 1.0)
    timing_confidence = np.clip(0.45 + 0.20 * np.abs(timing_score), 0.40, 0.85)
    prob_long_5 = np.where(timing_score > 0, timing_confidence, 0.15)
    prob_short_5 = np.where(timing_score < 0, timing_confidence, 0.15)
    prob_hold_5 = 1.0 - np.maximum(prob_long_5, prob_short_5)
    prob_short_5, prob_hold_5, prob_long_5 = _normalize_triplet(prob_short_5, prob_hold_5, prob_long_5)

    macro_score = frame["h4_bias"].fillna(0.0).to_numpy(dtype=float) * 0.55
    macro_score += np.tanh(frame["returns_15m"].fillna(0.0).to_numpy(dtype=float) * 300.0) * 0.25
    macro_score += np.tanh(frame["momentum"].fillna(0.0).to_numpy(dtype=float) / 12.0) * 0.20
    macro_confidence = np.clip(0.46 + 0.24 * np.abs(macro_score), 0.42, 0.88)
    prob_long_60 = np.where(macro_score > 0, macro_confidence, 0.14)
    prob_short_60 = np.where(macro_score < 0, macro_confidence, 0.14)
    prob_hold_60 = 1.0 - np.maximum(prob_long_60, prob_short_60)
    prob_short_60, prob_hold_60, prob_long_60 = _normalize_triplet(prob_short_60, prob_hold_60, prob_long_60)

    probs_5m = pd.DataFrame(
        {
            "prob_short_5m": prob_short_5,
            "prob_hold_5m": prob_hold_5,
            "prob_long_5m": prob_long_5,
        }
    )
    probs_60m = pd.DataFrame(
        {
            "prob_short_60m": prob_short_60,
            "prob_hold_60m": prob_hold_60,
            "prob_long_60m": prob_long_60,
        }
    )
    return probs_5m, probs_60m


def build_confluence_tags(row: pd.Series) -> str:
    tags: list[str] = []
    if float(row.get("entry_zone_present", 0.0)) >= 1.0:
        tags.append("ENTRY_ZONE")
    if float(row.get("bullish_ob", 0.0)) >= 1.0:
        tags.append("BULLISH_OB")
    if float(row.get("bearish_ob", 0.0)) >= 1.0:
        tags.append("BEARISH_OB")
    if float(row.get("fvg_up", 0.0)) > 0.0:
        tags.append("BULLISH_FVG")
    if float(row.get("fvg_down", 0.0)) > 0.0:
        tags.append("BEARISH_FVG")
    if float(row.get("liquidity_sweep_high", 0.0)) >= 1.0:
        tags.append("BUY_STOP_SWEEP")
    if float(row.get("liquidity_sweep_low", 0.0)) >= 1.0:
        tags.append("SELL_STOP_SWEEP")
    if float(row.get("h4_bias", 0.0)) > 0.0:
        tags.append("HIGHER_TF_BULLISH")
    elif float(row.get("h4_bias", 0.0)) < 0.0:
        tags.append("HIGHER_TF_BEARISH")
    return "|".join(tags) if tags else "NONE"


def derive_regime(row: pd.Series) -> str:
    volatility = float(row.get("volatility_regime", 0.0))
    h4_bias = float(row.get("h4_bias", 0.0))
    momentum = abs(float(row.get("momentum", 0.0)))
    if volatility >= 1.25:
        return "Expansion"
    if abs(h4_bias) >= 1 and momentum >= 8.0:
        return "Trend"
    return "Range"


def build_prediction_frame(
    validation_df: pd.DataFrame,
    config: dict[str, Any],
    source_timezone: str = "UTC",
) -> pd.DataFrame:
    trading_cfg = config["trading"]
    labels_cfg = config["labels"]
    ensemble_cfg = config["ensemble"]
    allowed = set(trading_cfg["allowed_sessions"])
    hunt_timezone = str(trading_cfg.get("hunt_timezone", "UTC"))
    hunt_windows = trading_cfg.get("hunt_windows")

    frame = validation_df.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    if "hour" not in frame.columns:
        frame["hour"] = frame["time"].dt.hour
    frame["market_session_name"] = frame["hour"].fillna(0).astype(int).map(session_name_from_hour)
    if hunt_windows:
        window_context = annotate_hunt_windows(
            frame.loc[:, ["time"]],
            timezone_name=hunt_timezone,
            windows=hunt_windows,
            source_timezone_name=source_timezone,
        )
        frame["trade_window_name"] = window_context["hunt_window_name"].astype(str)
        frame["trade_window_allowed"] = window_context["hunt_window_allowed"].astype(int)
        frame["trade_window_max_trades"] = window_context["hunt_window_trade_limit"].astype(int)
        frame["session_name"] = np.where(frame["trade_window_name"].ne(""), frame["trade_window_name"], frame["market_session_name"])
        frame["session_allowed"] = frame["trade_window_allowed"]
    else:
        frame["trade_window_name"] = ""
        frame["trade_window_allowed"] = frame["market_session_name"].isin(allowed).astype(int)
        frame["trade_window_max_trades"] = 0
        frame["session_name"] = frame["market_session_name"]
        frame["session_allowed"] = frame["trade_window_allowed"]
    frame["regime"] = frame.apply(derive_regime, axis=1)
    frame["confluence_tags"] = frame.apply(build_confluence_tags, axis=1)

    probs_5m, probs_60m = build_proxy_probabilities(frame)
    frame = pd.concat([frame, probs_5m, probs_60m], axis=1)
    frame["prob_short_15m"] = frame["prob_short"].astype(float)
    frame["prob_hold_15m"] = frame["prob_hold"].astype(float)
    frame["prob_long_15m"] = frame["prob_long"].astype(float)

    threshold = float(ensemble_cfg["confidence_threshold"])
    frame["signal_5m"] = frame.apply(
        lambda row: directional_signal(row["prob_short_5m"], row["prob_hold_5m"], row["prob_long_5m"], threshold),
        axis=1,
    )
    frame["signal_15m"] = frame.apply(
        lambda row: directional_signal(row["prob_short_15m"], row["prob_hold_15m"], row["prob_long_15m"], threshold),
        axis=1,
    )
    frame["signal_60m"] = frame.apply(
        lambda row: directional_signal(row["prob_short_60m"], row["prob_hold_60m"], row["prob_long_60m"], threshold),
        axis=1,
    )

    frame["setup_candidate"] = frame["signal_15m"].map(SETUP_NAME_MAP)
    frame["signal_class"] = np.where(
        (frame["signal_15m"] != 0)
        & (frame["signal_60m"] == frame["signal_15m"])
        & (frame["signal_5m"] != -frame["signal_15m"]),
        frame["signal_15m"],
        0,
    )
    frame["signal_name"] = frame["signal_class"].map(SIGNAL_NAME_MAP)

    long_stop_distance = np.maximum(frame["atr_14"].to_numpy(dtype=float), frame["swing_low_dist"].clip(lower=1e-6).to_numpy(dtype=float))
    short_stop_distance = np.maximum(frame["atr_14"].to_numpy(dtype=float), frame["swing_high_dist"].clip(lower=1e-6).to_numpy(dtype=float))
    stop_distance = np.where(frame["signal_class"].to_numpy(dtype=int) >= 0, long_stop_distance, short_stop_distance)
    frame["entry_price"] = frame["close"].astype(float)
    frame["planned_stop_distance"] = stop_distance
    frame["stop_loss"] = np.where(frame["signal_class"] >= 0, frame["entry_price"] - stop_distance, frame["entry_price"] + stop_distance)
    frame["tp1"] = np.where(
        frame["signal_class"] >= 0,
        frame["entry_price"] + stop_distance * float(labels_cfg["tp1_rr"]),
        frame["entry_price"] - stop_distance * float(labels_cfg["tp1_rr"]),
    )
    frame["tp2"] = np.where(
        frame["signal_class"] >= 0,
        frame["entry_price"] + stop_distance * float(labels_cfg["tp2_rr"]),
        frame["entry_price"] - stop_distance * float(labels_cfg["tp2_rr"]),
    )

    frame["entry_quality"] = (
        (frame["smc_quality_score"].fillna(0.0) / 4.0) * 0.70
        + frame["entry_zone_present"].fillna(0.0) * 0.20
        + frame["inducement_taken"].fillna(0.0) * 0.10
    ).clip(0.0, 1.0)
    frame["directional_confidence"] = np.where(
        frame["signal_class"] > 0,
        frame["prob_long_15m"],
        np.where(frame["signal_class"] < 0, frame["prob_short_15m"], np.maximum(frame["prob_long_15m"], frame["prob_short_15m"])),
    )
    adverse_probability = np.where(
        frame["signal_class"] > 0,
        frame["prob_short_15m"] + frame["prob_hold_15m"] * 0.50,
        np.where(frame["signal_class"] < 0, frame["prob_long_15m"] + frame["prob_hold_15m"] * 0.50, 1.0),
    )
    frame["expected_value"] = np.where(
        frame["signal_class"] != 0,
        frame["directional_confidence"] * float(labels_cfg["tp1_rr"]) - adverse_probability,
        0.0,
    )

    session_score = np.where(
        frame["trade_window_allowed"].eq(1),
        1.0,
        np.where(
            frame["market_session_name"].eq("Overlap"),
            1.0,
            np.where(frame["market_session_name"].eq("London"), 0.80, np.where(frame["market_session_name"].eq("New York"), 0.72, 0.0)),
        ),
    )
    frame["setup_score"] = (
        frame["directional_confidence"] * float(ensemble_cfg["setup_score_weights"]["probability"])
        + frame["entry_quality"] * float(ensemble_cfg["setup_score_weights"]["confluence"])
        + session_score * float(ensemble_cfg["setup_score_weights"]["session"])
        + (np.maximum(frame["expected_value"], 0.0) / max(float(labels_cfg["tp1_rr"]), 1.0)) * float(ensemble_cfg["setup_score_weights"]["expected_value"])
    ).clip(0.0, 1.0)

    frame["buy_pressure_proxy"] = (np.maximum(frame["delta"].fillna(0.0), 0.0) / frame["volume"].replace(0, np.nan)).fillna(0.0)
    frame["sell_pressure_proxy"] = (np.maximum(-frame["delta"].fillna(0.0), 0.0) / frame["volume"].replace(0, np.nan)).fillna(0.0)
    frame["time_of_day_bias"] = np.where(
        frame["market_session_name"].eq("Overlap"),
        0.80,
        np.where(frame["market_session_name"].eq("London"), 0.55, np.where(frame["market_session_name"].eq("New York"), 0.45, -0.15)),
    )
    frame["session_dominance"] = frame["buy_pressure_proxy"] - frame["sell_pressure_proxy"]
    frame["impulse_strength"] = frame["price_velocity"].fillna(0.0).abs()

    blocked_label = "HUNT_WINDOW_BLOCKED" if hunt_windows else "SESSION_BLOCKED"
    no_session = np.where(frame["session_allowed"].eq(0), blocked_label, "")
    no_bias = np.where(frame["signal_15m"].eq(0), "MAIN_BIAS_HOLD", "")
    no_confirmation = np.where((frame["signal_15m"] != 0) & (frame["signal_60m"] != frame["signal_15m"]), "H60_NOT_CONFIRMED", "")
    bad_timing = np.where((frame["signal_15m"] != 0) & (frame["signal_5m"] == -frame["signal_15m"]), "M5_TIMING_OPPOSES", "")
    low_entry = np.where(frame["entry_quality"] < float(trading_cfg["min_entry_quality"]), "LOW_ENTRY_QUALITY", "")
    low_confidence = np.where(frame["directional_confidence"] < threshold, "LOW_CONFIDENCE", "")
    negative_ev = np.where(frame["expected_value"] <= 0.0, "NEGATIVE_EV", "")
    low_setup = np.where(frame["setup_score"] < float(trading_cfg["min_setup_score"]), "LOW_SETUP_SCORE", "")
    frame["reason_blocked"] = [
        join_reasons(*parts)
        for parts in zip(no_session, no_bias, no_confirmation, bad_timing, low_entry, low_confidence, negative_ev, low_setup)
    ]
    frame["recommended_trade"] = np.where(frame["reason_blocked"].eq(""), frame["signal_name"], "WATCH")
    frame["gate_status"] = np.where(frame["recommended_trade"].isin(["LONG", "SHORT"]), "READY", "BLOCKED")
    frame["paper_status"] = np.where(frame["recommended_trade"].isin(["LONG", "SHORT"]), "SIGNAL_READY", "BLOCKED")
    frame["paper_reason_blocked"] = np.where(frame["recommended_trade"].isin(["LONG", "SHORT"]), "", frame["reason_blocked"])
    frame["paper_trade_id"] = pd.Series([pd.NA] * len(frame), dtype="object")
    frame["manual_override_used"] = False
    frame["manual_override_reason"] = ""
    return frame.sort_values("time").reset_index(drop=True)


def calculate_max_drawdown(equity_curve: list[dict[str, Any]]) -> float:
    if not equity_curve:
        return 0.0
    peak = float(equity_curve[0]["equity"])
    max_dd = 0.0
    for point in equity_curve:
        equity = float(point["equity"])
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def estimate_position_volume_lots(risk_cash: float, stop_distance: float, contract_size: float) -> float:
    risk_cash = float(risk_cash)
    stop_distance = abs(float(stop_distance))
    contract_size = float(contract_size)
    if risk_cash <= 0.0 or stop_distance <= 1e-8 or contract_size <= 0.0:
        return float("nan")
    return float(risk_cash / (stop_distance * contract_size))


def simulate_paper_trades(
    predictions: pd.DataFrame,
    raw_frame: pd.DataFrame,
    config: dict[str, Any],
    manual_override: dict[str, Any] | None = None,
    source_timezone: str = "UTC",
    starting_equity_override: float | None = None,
    contract_size_override: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    paper_cfg = config["trading"]["paper"]
    hunt_timezone = str(config.get("trading", {}).get("hunt_timezone", "UTC"))
    labels_cfg = config["labels"]
    override_state = dict(manual_override or {})

    working = predictions.copy().sort_values("time").reset_index(drop=True)
    raw = raw_frame.copy().sort_values("time").reset_index(drop=True)
    raw["time"] = pd.to_datetime(raw["time"])

    equity = float(starting_equity_override if starting_equity_override and starting_equity_override > 0.0 else paper_cfg["starting_equity"])
    risk_per_trade = float(paper_cfg["risk_per_trade"])
    contract_size = float(contract_size_override if contract_size_override and contract_size_override > 0.0 else (paper_cfg.get("contract_size", 100.0) or 100.0))
    max_holding_bars = int(paper_cfg["max_holding_bars"])
    max_trades_per_day = int(paper_cfg["max_trades_per_day"])
    daily_loss_stop = int(paper_cfg["daily_loss_streak_stop"])

    daily_trade_count: dict[str, int] = defaultdict(int)
    daily_loss_streak: dict[str, int] = defaultdict(int)
    window_trade_count: dict[str, int] = defaultdict(int)
    next_available_time: pd.Timestamp | None = None
    ledger_rows: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    open_trade_summary: dict[str, Any] | None = None
    trade_id = 0
    starting_balance_snapshot = float(equity)
    working["paper_risk_cash"] = 0.0
    working["paper_volume_lots"] = 0.0
    working["paper_account_balance"] = float(starting_balance_snapshot)
    working["position_sizing_mode"] = "equity_risk"
    working["assumed_contract_size"] = float(contract_size)
    ledger_columns = [
        "trade_id",
        "signal_time",
        "entry_time",
        "exit_time",
        "direction",
        "session_name",
        "trade_window_name",
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
        "risk_cash",
        "volume_lots",
        "position_sizing_mode",
        "assumed_contract_size",
        "realized_r",
        "pnl_cash",
        "equity_before",
        "equity_after",
        "manual_override_used",
        "manual_override_reason",
    ]

    for index, row in working.iterrows():
        signal_time = pd.Timestamp(row["time"])
        if signal_time.tzinfo is None:
            local_signal_time = signal_time.tz_localize(source_timezone).tz_convert(hunt_timezone)
        else:
            local_signal_time = signal_time.tz_convert(hunt_timezone)
        signal_day = str(local_signal_time.date())
        trade_window_name = str(row.get("trade_window_name", "") or row.get("session_name", "") or "UNSPECIFIED")
        trade_window_limit = int(row.get("trade_window_max_trades", 0) or 0)
        trade_window_key = f"{signal_day}|{trade_window_name}"
        if row["recommended_trade"] not in {"LONG", "SHORT"}:
            continue

        entry_price = float(row["entry_price"])
        stop_price = float(row["stop_loss"])
        tp1 = float(row["tp1"])
        tp2 = float(row["tp2"])
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 1e-8:
            working.at[index, "paper_status"] = "BLOCKED_BY_RISK"
            working.at[index, "paper_reason_blocked"] = join_reasons(str(working.at[index, "paper_reason_blocked"]), "INVALID_STOP_DISTANCE")
            continue
        risk_cash = equity * risk_per_trade
        volume_lots = estimate_position_volume_lots(risk_cash, stop_distance, contract_size)
        working.at[index, "paper_account_balance"] = float(equity)
        working.at[index, "paper_risk_cash"] = float(risk_cash)
        working.at[index, "paper_volume_lots"] = float(volume_lots) if pd.notna(volume_lots) else 0.0
        working.at[index, "position_sizing_mode"] = "equity_risk"
        working.at[index, "assumed_contract_size"] = float(contract_size)

        risk_blockers: list[str] = []
        if next_available_time is not None and signal_time <= next_available_time:
            risk_blockers.append("OPEN_TRADE_ACTIVE")
        if daily_trade_count[signal_day] >= max_trades_per_day:
            risk_blockers.append("DAILY_TRADE_LIMIT")
        if daily_loss_streak[signal_day] >= daily_loss_stop:
            risk_blockers.append("DAILY_LOSS_LIMIT")
        if trade_window_limit > 0 and window_trade_count[trade_window_key] >= trade_window_limit:
            risk_blockers.append("WINDOW_TRADE_LIMIT")

        if risk_blockers:
            override_used, override_reason = maybe_apply_manual_override(signal_time, risk_blockers, override_state)
            if not override_used:
                working.at[index, "paper_status"] = "BLOCKED_BY_RISK"
                working.at[index, "paper_reason_blocked"] = join_reasons(str(working.at[index, "paper_reason_blocked"]), *risk_blockers)
                continue
            working.at[index, "manual_override_used"] = True
            working.at[index, "manual_override_reason"] = override_reason

        trade_id += 1

        future_bars = raw.loc[raw["time"] > signal_time].head(max_holding_bars).copy()
        if future_bars.empty:
            working.at[index, "paper_status"] = "SIGNAL_READY"
            open_trade_summary = {
                "trade_id": f"T{trade_id:04d}",
                "signal_time": str(signal_time),
                "direction": row["recommended_trade"],
                "trade_window_name": trade_window_name,
                "entry_price": entry_price,
                "stop_loss": stop_price,
                "tp1": tp1,
                "tp2": tp2,
                "setup_score": float(row["setup_score"]),
                "expected_value": float(row["expected_value"]),
                "risk_cash": float(risk_cash),
                "volume_lots": volume_lots,
                "position_sizing_mode": "equity_risk",
                "assumed_contract_size": float(contract_size),
                "manual_override_used": bool(working.at[index, "manual_override_used"]),
                "manual_override_reason": str(working.at[index, "manual_override_reason"]),
            }
            working.at[index, "paper_trade_id"] = open_trade_summary["trade_id"]
            break

        tp1_hit = False
        tp2_hit = False
        realized_r = 0.0
        exit_reason = None
        exit_price = np.nan
        exit_time = None

        for bar_index, bar in future_bars.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])

            if row["recommended_trade"] == "LONG":
                hit_stop = low <= stop_price
                hit_tp1 = high >= tp1
                hit_tp2 = high >= tp2
            else:
                hit_stop = high >= stop_price
                hit_tp1 = low <= tp1
                hit_tp2 = low <= tp2

            if not tp1_hit:
                if hit_stop:
                    realized_r = -1.0
                    exit_reason = "STOP_LOSS"
                    exit_price = stop_price
                    exit_time = str(bar["time"])
                    break
                if hit_tp1:
                    tp1_hit = True
                    realized_r += 0.75
                    stop_price = entry_price

            if tp1_hit:
                if hit_stop:
                    exit_reason = "BREAKEVEN_AFTER_TP1"
                    exit_price = entry_price
                    exit_time = str(bar["time"])
                    break
                if hit_tp2:
                    tp2_hit = True
                    realized_r += 1.25
                    exit_reason = "TP2"
                    exit_price = tp2
                    exit_time = str(bar["time"])
                    break

            is_last_bar = bar_index == future_bars.index[-1]
            if is_last_bar and exit_reason is None:
                close_price = float(bar["close"])
                if row["recommended_trade"] == "LONG":
                    move_r = (close_price - entry_price) / stop_distance
                else:
                    move_r = (entry_price - close_price) / stop_distance
                realized_r = realized_r + (0.50 * move_r if tp1_hit else move_r)
                exit_reason = "TIME_EXIT"
                exit_price = close_price
                exit_time = str(bar["time"])

        if exit_reason is None:
            working.at[index, "paper_status"] = "OPEN"
            working.at[index, "paper_trade_id"] = f"T{trade_id:04d}"
            daily_trade_count[signal_day] += 1
            if trade_window_limit > 0:
                window_trade_count[trade_window_key] += 1
            open_trade_summary = {
                "trade_id": f"T{trade_id:04d}",
                "signal_time": str(signal_time),
                "direction": row["recommended_trade"],
                "trade_window_name": trade_window_name,
                "entry_price": entry_price,
                "stop_loss": stop_price,
                "tp1": tp1,
                "tp2": tp2,
                "setup_score": float(row["setup_score"]),
                "expected_value": float(row["expected_value"]),
                "risk_cash": float(risk_cash),
                "volume_lots": volume_lots,
                "position_sizing_mode": "equity_risk",
                "assumed_contract_size": float(contract_size),
                "manual_override_used": bool(working.at[index, "manual_override_used"]),
                "manual_override_reason": str(working.at[index, "manual_override_reason"]),
            }
            next_available_time = future_bars.iloc[-1]["time"]
            continue

        working.at[index, "paper_status"] = "OPENED"
        working.at[index, "paper_trade_id"] = f"T{trade_id:04d}"
        pnl_cash = risk_cash * realized_r
        equity_after = equity + pnl_cash
        daily_trade_count[signal_day] += 1
        if trade_window_limit > 0:
            window_trade_count[trade_window_key] += 1
        if realized_r < 0:
            daily_loss_streak[signal_day] += 1
        else:
            daily_loss_streak[signal_day] = 0

        ledger_rows.append(
            {
                "trade_id": f"T{trade_id:04d}",
                "signal_time": str(signal_time),
                "entry_time": str(signal_time),
                "exit_time": exit_time,
                "direction": row["recommended_trade"],
                "session_name": row["session_name"],
                "trade_window_name": trade_window_name,
                "regime": row["regime"],
                "entry_price": entry_price,
                "stop_loss": float(row["stop_loss"]),
                "tp1": tp1,
                "tp2": tp2,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "tp1_hit": tp1_hit,
                "tp2_hit": tp2_hit,
                "setup_score": float(row["setup_score"]),
                "expected_value": float(row["expected_value"]),
                "entry_quality": float(row["entry_quality"]),
                "risk_cash": float(risk_cash),
                "volume_lots": volume_lots,
                "position_sizing_mode": "equity_risk",
                "assumed_contract_size": float(contract_size),
                "realized_r": float(realized_r),
                "pnl_cash": float(pnl_cash),
                "equity_before": float(equity),
                "equity_after": float(equity_after),
                "manual_override_used": bool(working.at[index, "manual_override_used"]),
                "manual_override_reason": str(working.at[index, "manual_override_reason"]),
            }
        )
        equity_curve.append({"time": exit_time, "equity": float(equity_after)})
        equity = equity_after
        next_available_time = pd.Timestamp(exit_time) if exit_time else None

    ledger = pd.DataFrame(ledger_rows, columns=ledger_columns)
    if ledger.empty:
        summary = {
            "starting_equity": float(paper_cfg["starting_equity"]),
            "ending_equity": float(paper_cfg["starting_equity"]),
            "net_pnl_cash": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "flat_count": 0,
            "win_rate": 0.0,
            "precision": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "max_drawdown": 0.0,
            "risk_per_trade": risk_per_trade,
            "position_sizing_mode": "equity_risk",
            "assumed_contract_size": float(contract_size),
            "equity_curve": [],
            "open_trade": open_trade_summary,
        }
        return working, ledger, summary, override_state

    positive_r = ledger.loc[ledger["realized_r"] > 0, "realized_r"].sum()
    negative_r = ledger.loc[ledger["realized_r"] < 0, "realized_r"].sum()
    profit_factor = float(positive_r / abs(negative_r)) if negative_r < 0 else 0.0
    starting_equity = float(paper_cfg["starting_equity"])
    ending_equity = float(ledger["equity_after"].iloc[-1])
    win_count = int((ledger["realized_r"] > 0).sum())
    loss_count = int((ledger["realized_r"] < 0).sum())
    flat_count = int((ledger["realized_r"] == 0).sum())
    summary = {
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "net_pnl_cash": ending_equity - starting_equity,
        "trade_count": int(len(ledger)),
        "win_count": win_count,
        "loss_count": loss_count,
        "flat_count": flat_count,
        "win_rate": float((ledger["realized_r"] > 0).mean()),
        "precision": float((ledger["realized_r"] > 0).mean()),
        "profit_factor": profit_factor,
        "expectancy_r": float(ledger["realized_r"].mean()),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "risk_per_trade": risk_per_trade,
        "position_sizing_mode": "equity_risk",
        "assumed_contract_size": float(contract_size),
        "equity_curve": equity_curve,
        "open_trade": open_trade_summary,
    }
    return working, ledger, summary, override_state


def build_session_profile(predictions: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for session_name, subset in predictions.groupby("session_name"):
        profile[str(session_name)] = {
            "bars": int(len(subset)),
            "avg_session_dominance": float(subset["session_dominance"].mean()),
            "avg_impulse_strength": float(subset["impulse_strength"].mean()),
        }
    return profile


def build_learning_feedback(predictions: pd.DataFrame, paper_ledger: pd.DataFrame) -> pd.DataFrame:
    feedback_columns = [
        "time",
        "recommended_trade",
        "gate_status",
        "paper_status",
        "paper_reason_blocked",
        "paper_trade_id",
        "session_name",
        "regime",
        "setup_candidate",
        "setup_score",
        "expected_value",
        "directional_confidence",
        "entry_quality",
        "signal_5m",
        "signal_15m",
        "signal_60m",
        "prob_short_15m",
        "prob_hold_15m",
        "prob_long_15m",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "confluence_tags",
        "buy_pressure_proxy",
        "sell_pressure_proxy",
        "time_of_day_bias",
        "session_dominance",
        "impulse_strength",
    ]
    feedback = predictions.loc[:, [column for column in feedback_columns if column in predictions.columns]].copy()
    feedback = feedback.rename(columns={"time": "signal_time"})
    feedback["signal_time"] = feedback["signal_time"].astype(str)
    feedback["actionable_signal"] = feedback["recommended_trade"].isin(["LONG", "SHORT"])
    feedback["feedback_kind"] = np.where(
        feedback["paper_status"].eq("OPENED"),
        "closed_trade",
        np.where(
            feedback["paper_status"].eq("OPEN"),
            "open_trade",
            np.where(feedback["actionable_signal"], "blocked_trade", "watch_only"),
        ),
    )

    if paper_ledger.empty:
        feedback["has_trade_outcome"] = False
        feedback["trade_outcome"] = "PENDING"
        feedback["realized_r"] = np.nan
        feedback["pnl_cash"] = np.nan
        feedback["exit_reason"] = ""
        feedback["tp1_hit"] = False
        feedback["tp2_hit"] = False
        feedback["eligible_for_batch_learning"] = False
        return feedback

    ledger = paper_ledger.copy()
    join_columns = [
        "trade_id",
        "signal_time",
        "direction",
        "exit_reason",
        "tp1_hit",
        "tp2_hit",
        "realized_r",
        "pnl_cash",
    ]
    ledger = ledger.loc[:, [column for column in join_columns if column in ledger.columns]].copy()
    ledger["signal_time"] = ledger["signal_time"].astype(str)
    feedback = feedback.merge(ledger, how="left", left_on=["paper_trade_id", "signal_time"], right_on=["trade_id", "signal_time"])
    feedback["has_trade_outcome"] = feedback["realized_r"].notna()
    feedback["trade_outcome"] = np.where(
        feedback["realized_r"].isna(),
        "PENDING",
        np.where(
            feedback["realized_r"] > 0,
            "WIN",
            np.where(feedback["realized_r"] < 0, "LOSS", "FLAT"),
        ),
    )
    feedback["eligible_for_batch_learning"] = feedback["has_trade_outcome"] & feedback["actionable_signal"]
    return feedback


def _share_from_counts(values: pd.Series) -> tuple[float, dict[str, int], str]:
    if values.empty:
        return 0.0, {}, ""
    counts = values.astype(str).value_counts()
    dominant_label = str(counts.index[0])
    dominant_share = float(counts.iloc[0] / counts.sum())
    return dominant_share, {str(key): int(val) for key, val in counts.items()}, dominant_label


def build_learning_status(feedback: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    learning_cfg = config["learning"]
    closed = feedback.loc[feedback["eligible_for_batch_learning"].fillna(False)].copy()
    actionable = feedback.loc[feedback["actionable_signal"].fillna(False)].copy()

    if not closed.empty:
        closed["signal_time"] = pd.to_datetime(closed["signal_time"], errors="coerce")
        closed["signal_day"] = closed["signal_time"].dt.date.astype(str)
    else:
        closed["signal_day"] = pd.Series(dtype="object")

    closed_trades = int(len(closed))
    trading_days = int(closed["signal_day"].nunique()) if "signal_day" in closed.columns else 0
    long_trades = int((closed["recommended_trade"] == "LONG").sum()) if closed_trades else 0
    short_trades = int((closed["recommended_trade"] == "SHORT").sum()) if closed_trades else 0
    dominant_session_share, session_counts, dominant_session = _share_from_counts(closed["session_name"]) if closed_trades else (0.0, {}, "")
    dominant_direction_share, direction_counts, dominant_direction = _share_from_counts(closed["recommended_trade"]) if closed_trades else (0.0, {}, "")

    recent_window = min(int(learning_cfg["recent_trade_window"]), closed_trades) if closed_trades else 0
    recent_trade_share = float(recent_window / closed_trades) if closed_trades else 0.0

    checks = [
        {
            "check": "closed_trades",
            "required": str(int(learning_cfg["batch_min_closed_trades"])),
            "current": closed_trades,
            "passed": closed_trades >= int(learning_cfg["batch_min_closed_trades"]),
        },
        {
            "check": "trading_days",
            "required": str(int(learning_cfg["batch_min_days"])),
            "current": trading_days,
            "passed": trading_days >= int(learning_cfg["batch_min_days"]),
        },
        {
            "check": "long_trades",
            "required": str(int(learning_cfg["batch_min_long_trades"])),
            "current": long_trades,
            "passed": long_trades >= int(learning_cfg["batch_min_long_trades"]),
        },
        {
            "check": "short_trades",
            "required": str(int(learning_cfg["batch_min_short_trades"])),
            "current": short_trades,
            "passed": short_trades >= int(learning_cfg["batch_min_short_trades"]),
        },
        {
            "check": "max_single_session_share",
            "required": f"<= {float(learning_cfg['max_single_session_share']):.2f}",
            "current": dominant_session_share,
            "passed": dominant_session_share <= float(learning_cfg["max_single_session_share"]),
        },
        {
            "check": "max_direction_share",
            "required": f"<= {float(learning_cfg['max_direction_share']):.2f}",
            "current": dominant_direction_share,
            "passed": dominant_direction_share <= float(learning_cfg["max_direction_share"]),
        },
        {
            "check": "max_recent_trade_share",
            "required": f"<= {float(learning_cfg['max_recent_trade_share']):.2f}",
            "current": recent_trade_share,
            "passed": recent_trade_share <= float(learning_cfg["max_recent_trade_share"]),
        },
    ]
    reasons = [str(item["check"]) for item in checks if not item["passed"]]
    return {
        "closed_trades": closed_trades,
        "actionable_signals": int(len(actionable)),
        "watch_only_rows": int((feedback["feedback_kind"] == "watch_only").sum()),
        "trading_days": trading_days,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "dominant_session": dominant_session,
        "dominant_session_share": dominant_session_share,
        "session_counts": session_counts,
        "dominant_direction": dominant_direction,
        "dominant_direction_share": dominant_direction_share,
        "direction_counts": direction_counts,
        "recent_trade_window": recent_window,
        "recent_trade_share": recent_trade_share,
        "mean_realized_r": float(closed["realized_r"].mean()) if closed_trades else 0.0,
        "retrain_ready": not reasons,
        "retrain_blockers": reasons,
        "checks": checks,
        "recommended_action": "batch_retraining_allowed" if not reasons else "collect_more_diverse_feedback",
    }


def build_trade_directive_frame(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "epoch_utc",
        "time_utc",
        "recommended_trade",
        "gate_status",
        "paper_status",
        "paper_reason_blocked",
        "directional_confidence",
        "setup_score",
        "expected_value",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "session_name",
        "manual_override_used",
        "execution_scope",
        "learning_source",
        "streamlit_scope",
        "volume_lots",
        "risk_cash",
        "position_sizing_mode",
        "account_balance_snapshot",
        "assumed_contract_size",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    trading_cfg = config.get("trading", {})
    threshold = float(config.get("ensemble", {}).get("confidence_threshold", 0.46))
    ignored_blockers = {"H60_NOT_CONFIRMED", "NEGATIVE_EV"}
    latest_prediction_time = pd.to_datetime(predictions["time"], errors="coerce", utc=True).max()
    directive_max_age_bars = int(trading_cfg.get("directive_max_age_bars", 2) or 2)
    directive_bar_seconds = 60
    now_utc = pd.Timestamp.now(tz="UTC")
    directive_reference_time = latest_prediction_time if pd.notna(latest_prediction_time) and latest_prediction_time >= now_utc else now_utc

    def build_directive_row(latest: pd.Series) -> dict[str, Any]:
        signal_15m = int(latest.get("signal_15m", 0) or 0)
        execution_reasons = [
            reason
            for reason in split_reasons(str(latest.get("reason_blocked", "")))
            if reason not in ignored_blockers
        ]
        execution_trade = str(latest.get("recommended_trade", "")).strip().upper()
        execution_gate_status = str(latest.get("gate_status", "")).strip().upper()
        execution_paper_status = str(latest.get("paper_status", "")).strip().upper()
        execution_reason_blocked = str(latest.get("paper_reason_blocked", "")).strip()

        session_allowed = int(latest.get("session_allowed", 0) or 0) == 1
        entry_quality_ok = float(latest.get("entry_quality", 0.0) or 0.0) >= float(trading_cfg.get("min_entry_quality", 0.0) or 0.0)
        setup_score_ok = float(latest.get("setup_score", 0.0) or 0.0) >= float(trading_cfg.get("min_setup_score", 0.0) or 0.0)
        confidence_ok = float(latest.get("directional_confidence", 0.0) or 0.0) >= threshold
        if signal_15m != 0 and session_allowed and entry_quality_ok and setup_score_ok and confidence_ok and not execution_reasons:
            execution_trade = "LONG" if signal_15m > 0 else "SHORT"
            execution_gate_status = "READY"
            execution_paper_status = "SIGNAL_READY"
            execution_reason_blocked = ""

        account_balance_snapshot = float(latest.get("paper_account_balance", 0.0) or 0.0)
        assumed_contract_size = float(latest.get("assumed_contract_size", 0.0) or 0.0)
        if assumed_contract_size <= 0.0:
            assumed_contract_size = float(config.get("trading", {}).get("paper", {}).get("contract_size", 100.0) or 100.0)
        risk_cash = float(latest.get("paper_risk_cash", 0.0) or 0.0)
        if risk_cash <= 0.0 and account_balance_snapshot > 0.0:
            risk_cash = account_balance_snapshot * float(config.get("trading", {}).get("paper", {}).get("risk_per_trade", 0.0) or 0.0)

        entry_price = float(latest.get("entry_price", 0.0) or 0.0)
        stop_loss = float(latest.get("stop_loss", 0.0) or 0.0)
        tp1 = float(latest.get("tp1", 0.0) or 0.0)
        tp2 = float(latest.get("tp2", 0.0) or 0.0)
        stop_distance = abs(entry_price - stop_loss)
        tp1_distance = abs(tp1 - entry_price)
        tp2_distance = abs(tp2 - entry_price)
        if execution_trade == "LONG":
            stop_loss = entry_price - stop_distance
            tp1 = entry_price + tp1_distance
            tp2 = entry_price + tp2_distance
        elif execution_trade == "SHORT":
            stop_loss = entry_price + stop_distance
            tp1 = entry_price - tp1_distance
            tp2 = entry_price - tp2_distance

        volume_lots = float(latest.get("paper_volume_lots", 0.0) or 0.0)
        if volume_lots <= 0.0 and execution_gate_status == "READY" and execution_paper_status == "SIGNAL_READY":
            volume_lots = estimate_position_volume_lots(risk_cash, stop_distance, assumed_contract_size)

        signal_time = pd.to_datetime(latest.get("time"), errors="coerce", utc=True)
        if pd.notna(signal_time) and pd.notna(directive_reference_time):
            age_seconds = max(0.0, float((directive_reference_time - signal_time).total_seconds()))
            if age_seconds > directive_max_age_bars * directive_bar_seconds:
                execution_trade = "WATCH"
                execution_gate_status = "BLOCKED"
                execution_paper_status = "BLOCKED"
                execution_reason_blocked = join_reasons(execution_reason_blocked, "STALE_DIRECTIVE")
                volume_lots = 0.0
                risk_cash = 0.0

        epoch_utc = int(signal_time.timestamp()) if pd.notna(signal_time) else 0
        return {
            "epoch_utc": epoch_utc,
            "time_utc": str(latest.get("time", "")),
            "recommended_trade": execution_trade,
            "gate_status": execution_gate_status,
            "paper_status": execution_paper_status,
            "paper_reason_blocked": execution_reason_blocked,
            "directional_confidence": float(latest.get("directional_confidence", 0.0)),
            "setup_score": float(latest.get("setup_score", 0.0)),
            "expected_value": float(latest.get("expected_value", 0.0)),
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "session_name": str(latest.get("session_name", "")),
            "manual_override_used": bool(latest.get("manual_override_used", False)),
            "execution_scope": "mirror_to_mt5_demo",
            "learning_source": "paper_signal_with_mt5_execution",
            "streamlit_scope": "paper_research_only",
            "volume_lots": float(volume_lots or 0.0),
            "risk_cash": float(risk_cash or 0.0),
            "position_sizing_mode": str(latest.get("position_sizing_mode", "equity_risk")),
            "account_balance_snapshot": float(account_balance_snapshot or 0.0),
            "assumed_contract_size": float(assumed_contract_size or 0.0),
        }

    evaluated_rows = [build_directive_row(predictions.iloc[index]) for index in range(len(predictions))]
    selected_row = next(
        (
            row
            for row in reversed(evaluated_rows)
            if row["recommended_trade"] in {"LONG", "SHORT"}
            and row["gate_status"] == "READY"
            and row["paper_status"] == "SIGNAL_READY"
        ),
        evaluated_rows[-1],
    )
    return pd.DataFrame(
        [
            selected_row
        ],
        columns=columns,
    )


def build_history_trade_key(row: pd.Series) -> str:
    signal_time = str(row.get("signal_time", "")).strip()
    direction = str(row.get("direction", "")).strip().upper()
    trade_window_name = str(row.get("trade_window_name", "") or row.get("session_name", "")).strip()
    return "|".join([signal_time, direction, trade_window_name])


def build_final_trade_key(row: pd.Series) -> str:
    history_trade_key = str(row.get("history_trade_key", "")).strip()
    if history_trade_key:
        return history_trade_key
    signal_time = str(row.get("signal_time", "")).strip()
    direction = str(row.get("direction", "")).strip().upper()
    trade_window_name = str(row.get("trade_window_name", "") or row.get("session_name", "")).strip()
    if signal_time and direction and trade_window_name:
        return "|".join([signal_time, direction, trade_window_name])
    exit_time = str(row.get("exit_time", "")).strip()
    if not exit_time or not direction or not trade_window_name:
        return ""
    return "|".join([exit_time, direction, trade_window_name])


def infer_trade_signal_timestamp(row: pd.Series) -> pd.Timestamp:
    for column in ("signal_time", "entry_time"):
        parsed = pd.to_datetime(row.get(column), errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed
    for column in ("history_trade_key", "final_trade_key"):
        key_text = str(row.get(column, "")).strip()
        if not key_text:
            continue
        candidate = key_text.split("|", 1)[0].strip()
        parsed = pd.to_datetime(candidate, errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed
    return pd.NaT


def backfill_trade_key_times(
    frame: pd.DataFrame,
    *,
    key_column: str,
    signal_column: str = "signal_time",
    entry_column: str = "entry_time",
) -> pd.DataFrame:
    if frame.empty or key_column not in frame.columns:
        return frame

    working = frame.copy()
    inferred_times = pd.to_datetime(
        working[key_column].astype(str).str.split("|").str[0],
        errors="coerce",
        utc=True,
    )
    for column in (signal_column, entry_column):
        if column not in working.columns:
            continue
        parsed = pd.to_datetime(working[column], errors="coerce", utc=True)
        missing = parsed.isna()
        if missing.any():
            working.loc[missing, column] = inferred_times.loc[missing]
    return working


def drop_malformed_closed_trade_rows(
    frame: pd.DataFrame,
    *,
    key_column: str,
    exit_column: str = "exit_time",
) -> pd.DataFrame:
    if frame.empty:
        return frame

    working = frame.copy()
    if key_column in working.columns:
        working = working.loc[working[key_column].astype(str).str.len() > 0].copy()
    if exit_column in working.columns:
        exit_times = pd.to_datetime(working[exit_column], errors="coerce", utc=True)
        working = working.loc[exit_times.notna()].copy()
    return working


def sanitize_trade_history_frame(
    frame: pd.DataFrame,
    *,
    key_column: str = "history_trade_key",
) -> pd.DataFrame:
    if frame.empty:
        return frame

    working = frame.copy()
    for column in ("snapshot_created_at", "source_start", "source_end", "signal_time", "entry_time", "exit_time"):
        if column in working.columns:
            working[column] = pd.to_datetime(working[column], errors="coerce", utc=True)

    working = backfill_trade_key_times(working, key_column=key_column)
    working = drop_malformed_closed_trade_rows(working, key_column=key_column)

    sort_columns = [column for column in ("snapshot_created_at", "source_end", "signal_time", "exit_time") if column in working.columns]
    if sort_columns:
        working = working.sort_values(by=sort_columns, ascending=True).copy()
    if key_column in working.columns:
        working = working.drop_duplicates(subset=[key_column], keep="last").copy()

    for column in ("snapshot_created_at", "source_start", "source_end", "signal_time", "entry_time", "exit_time"):
        if column in working.columns:
            working[column] = working[column].astype(str).replace("NaT", "")

    return working


def resolve_trade_curve_time(row: pd.Series) -> str:
    for column in ("exit_time", "signal_time", "entry_time"):
        parsed = pd.to_datetime(row.get(column), errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed.isoformat()
    inferred = infer_trade_signal_timestamp(row)
    if pd.notna(inferred):
        return inferred.isoformat()
    return ""


def build_trade_summary_from_ledger(
    ledger: pd.DataFrame,
    starting_equity: float,
    open_trade_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    starting_equity = float(starting_equity)
    summary_risk_per_trade = 0.0
    summary_position_sizing_mode = ""
    summary_contract_size = 0.0
    if open_trade_summary:
        summary_position_sizing_mode = str(open_trade_summary.get("position_sizing_mode", "") or "")
        summary_contract_size = float(open_trade_summary.get("assumed_contract_size", 0.0) or 0.0)
        open_trade_risk_cash = float(open_trade_summary.get("risk_cash", 0.0) or 0.0)
        if starting_equity > 0.0 and open_trade_risk_cash > 0.0:
            summary_risk_per_trade = float(open_trade_risk_cash / starting_equity)
    if ledger.empty:
        return {
            "starting_equity": starting_equity,
            "ending_equity": starting_equity,
            "net_pnl_cash": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "flat_count": 0,
            "win_rate": 0.0,
            "precision": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "max_drawdown": 0.0,
            "risk_per_trade": summary_risk_per_trade,
            "position_sizing_mode": summary_position_sizing_mode,
            "assumed_contract_size": summary_contract_size,
            "equity_curve": [],
            "open_trade": open_trade_summary,
        }

    positive_r = ledger.loc[ledger["realized_r"] > 0, "realized_r"].sum()
    negative_r = ledger.loc[ledger["realized_r"] < 0, "realized_r"].sum()
    profit_factor = float(positive_r / abs(negative_r)) if negative_r < 0 else 0.0
    ending_equity = float(ledger["equity_after"].iloc[-1])
    win_count = int((ledger["realized_r"] > 0).sum())
    loss_count = int((ledger["realized_r"] < 0).sum())
    flat_count = int((ledger["realized_r"] == 0).sum())
    if "risk_cash" in ledger.columns and "equity_before" in ledger.columns:
        equity_before = pd.to_numeric(ledger["equity_before"], errors="coerce")
        risk_cash = pd.to_numeric(ledger["risk_cash"], errors="coerce")
        valid = equity_before.gt(0) & risk_cash.notna()
        if valid.any():
            summary_risk_per_trade = float((risk_cash[valid] / equity_before[valid]).iloc[-1])
    if "position_sizing_mode" in ledger.columns:
        sizing_series = ledger["position_sizing_mode"].dropna().astype(str)
        if not sizing_series.empty:
            summary_position_sizing_mode = str(sizing_series.iloc[-1])
    if "assumed_contract_size" in ledger.columns:
        contract_series = pd.to_numeric(ledger["assumed_contract_size"], errors="coerce").dropna()
        if not contract_series.empty:
            summary_contract_size = float(contract_series.iloc[-1])
    equity_curve = []
    for _, row in ledger.iterrows():
        point_time = resolve_trade_curve_time(row)
        if not point_time:
            continue
        equity_curve.append({"time": point_time, "equity": float(row.get("equity_after", 0.0))})
    return {
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "net_pnl_cash": ending_equity - starting_equity,
        "trade_count": int(len(ledger)),
        "win_count": win_count,
        "loss_count": loss_count,
        "flat_count": flat_count,
        "win_rate": float((ledger["realized_r"] > 0).mean()),
        "precision": float((ledger["realized_r"] > 0).mean()),
        "profit_factor": profit_factor,
        "expectancy_r": float(ledger["realized_r"].mean()),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "risk_per_trade": summary_risk_per_trade,
        "position_sizing_mode": summary_position_sizing_mode,
        "assumed_contract_size": summary_contract_size,
        "equity_curve": equity_curve,
        "open_trade": open_trade_summary,
    }


def materialize_final_trade_ledger(
    trade_history: pd.DataFrame,
    final_ledger_output: str,
    starting_equity: float,
    open_trade_summary: dict[str, Any] | None = None,
    risk_per_trade: float = 0.0,
    position_sizing_mode: str = "equity_risk",
    assumed_contract_size: float = 100.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_columns = [
        "final_trade_key",
        "snapshot_run_id",
        "snapshot_created_at",
        "source_end",
        "latest_signal_time_at_run",
        "latest_signal_at_run",
        "history_trade_key",
        "trade_id",
        "signal_time",
        "entry_time",
        "exit_time",
        "direction",
        "session_name",
        "trade_window_name",
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
        "risk_cash",
        "volume_lots",
        "position_sizing_mode",
        "assumed_contract_size",
        "realized_r",
        "pnl_cash",
        "equity_before",
        "equity_after",
        "manual_override_used",
        "manual_override_reason",
    ]
    final_path = ensure_parent_dir(final_ledger_output)
    if trade_history.empty:
        empty = pd.DataFrame(columns=base_columns)
        empty.to_csv(final_path, index=False)
        return empty, build_trade_summary_from_ledger(empty, starting_equity, open_trade_summary)

    frame = trade_history.copy()
    frame["final_trade_key"] = frame.apply(build_final_trade_key, axis=1)
    frame = frame.loc[frame["final_trade_key"].astype(str).str.len() > 0].copy()
    if frame.empty:
        empty = pd.DataFrame(columns=base_columns)
        empty.to_csv(final_path, index=False)
        return empty, build_trade_summary_from_ledger(empty, starting_equity, open_trade_summary)

    for column in ("snapshot_created_at", "source_end", "signal_time", "entry_time", "exit_time"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ("realized_r", "pnl_cash"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    frame = backfill_trade_key_times(frame, key_column="history_trade_key")
    frame = drop_malformed_closed_trade_rows(frame, key_column="final_trade_key")
    if frame.empty:
        empty = pd.DataFrame(columns=base_columns)
        empty.to_csv(final_path, index=False)
        return empty, build_trade_summary_from_ledger(empty, starting_equity, open_trade_summary)

    sort_columns = [column for column in ("snapshot_created_at", "source_end", "signal_time", "exit_time") if column in frame.columns]
    ordered = frame.sort_values(by=sort_columns, ascending=True).copy()
    final_ledger = ordered.drop_duplicates(subset=["final_trade_key"], keep="last").copy()
    final_sort_columns = [column for column in ("exit_time", "signal_time", "snapshot_created_at") if column in final_ledger.columns]
    if final_sort_columns:
        final_ledger = final_ledger.sort_values(by=final_sort_columns, ascending=True).copy()

    for index, row in final_ledger.iterrows():
        inferred_signal_time = infer_trade_signal_timestamp(row)
        if pd.isna(inferred_signal_time):
            continue
        signal_time = pd.to_datetime(row.get("signal_time"), errors="coerce", utc=True)
        entry_time = pd.to_datetime(row.get("entry_time"), errors="coerce", utc=True)
        if pd.isna(signal_time):
            final_ledger.at[index, "signal_time"] = inferred_signal_time
        if pd.isna(entry_time):
            final_ledger.at[index, "entry_time"] = inferred_signal_time
    if final_sort_columns:
        final_ledger = final_ledger.sort_values(by=final_sort_columns, ascending=True).copy()

    equity = float(starting_equity)
    risk_per_trade = float(risk_per_trade or 0.0)
    if "realized_r" not in final_ledger.columns:
        final_ledger["realized_r"] = 0.0

    final_ledger["position_sizing_mode"] = position_sizing_mode or "equity_risk"
    final_ledger["assumed_contract_size"] = float(assumed_contract_size or 100.0)

    for index, row in final_ledger.iterrows():
        realized_r_raw = pd.to_numeric(row.get("realized_r", 0.0), errors="coerce")
        realized_r = 0.0 if pd.isna(realized_r_raw) else float(realized_r_raw)
        entry_price_raw = pd.to_numeric(row.get("entry_price", np.nan), errors="coerce")
        stop_loss_raw = pd.to_numeric(row.get("stop_loss", np.nan), errors="coerce")
        stop_distance = abs(float(entry_price_raw) - float(stop_loss_raw)) if pd.notna(entry_price_raw) and pd.notna(stop_loss_raw) else np.nan

        equity_before = equity
        if position_sizing_mode == "equity_risk" and risk_per_trade > 0.0:
            risk_cash = equity_before * risk_per_trade
            pnl_cash = risk_cash * realized_r
        else:
            pnl_cash_raw = pd.to_numeric(row.get("pnl_cash", 0.0), errors="coerce")
            pnl_cash = 0.0 if pd.isna(pnl_cash_raw) else float(pnl_cash_raw)
            risk_cash = abs(pnl_cash / realized_r) if abs(realized_r) > 1e-8 else np.nan

        equity_after = equity_before + pnl_cash
        final_ledger.at[index, "risk_cash"] = float(risk_cash) if pd.notna(risk_cash) else np.nan
        final_ledger.at[index, "pnl_cash"] = float(pnl_cash)
        final_ledger.at[index, "equity_before"] = float(equity_before)
        final_ledger.at[index, "equity_after"] = float(equity_after)
        final_ledger.at[index, "volume_lots"] = estimate_position_volume_lots(risk_cash, stop_distance, assumed_contract_size)
        equity = equity_after

    ordered_output_columns = [column for column in base_columns if column in final_ledger.columns]
    ordered_output_columns.extend(column for column in final_ledger.columns if column not in ordered_output_columns)
    final_ledger = final_ledger.loc[:, ordered_output_columns].copy()

    for column in ("snapshot_created_at", "source_end", "signal_time", "entry_time", "exit_time"):
        if column in final_ledger.columns:
            final_ledger[column] = final_ledger[column].astype(str).replace("NaT", "")

    final_ledger.to_csv(final_path, index=False)
    return final_ledger, build_trade_summary_from_ledger(final_ledger, starting_equity, open_trade_summary)


def append_trade_history(
    paper_ledger: pd.DataFrame,
    trade_history_output: str,
    snapshot_run_id: str,
    snapshot_created_at: str,
    report: dict[str, Any],
) -> pd.DataFrame:
    history_path = resolve_repo_path(trade_history_output)
    existing = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    if not existing.empty and "history_trade_key" not in existing.columns:
        existing["history_trade_key"] = existing.apply(build_history_trade_key, axis=1)
    prediction_window = report.get("prediction_window", {}) or {}
    rebuild_window_start = pd.to_datetime(prediction_window.get("start"), errors="coerce", utc=True)
    rebuild_window_end = pd.to_datetime(prediction_window.get("end"), errors="coerce", utc=True)
    if (pd.isna(rebuild_window_start) or pd.isna(rebuild_window_end)) and not paper_ledger.empty and "signal_time" in paper_ledger.columns:
        signal_times = pd.to_datetime(paper_ledger["signal_time"], errors="coerce", utc=True)
        if signal_times.notna().any():
            rebuild_window_start = signal_times.min()
            rebuild_window_end = signal_times.max()
    source_meta = report.get("mt5_live_source", {}) or {}
    if not existing.empty and "signal_time" in existing.columns and pd.notna(rebuild_window_start) and pd.notna(rebuild_window_end):
        existing_signal_times = pd.to_datetime(existing["signal_time"], errors="coerce", utc=True)
        # Only the current prediction window is authoritative; older frozen trades must survive empty future runs.
        keep_existing = (
            existing_signal_times.isna()
            | existing_signal_times.lt(rebuild_window_start)
            | existing_signal_times.gt(rebuild_window_end)
        )
        existing = existing.loc[keep_existing].copy()

    if paper_ledger.empty:
        if history_path.exists():
            existing = sanitize_trade_history_frame(existing)
            existing.to_csv(ensure_parent_dir(trade_history_output), index=False)
            return existing
        empty_columns = [
            "history_trade_key",
            "snapshot_run_id",
            "snapshot_created_at",
            "source_start",
            "source_end",
            "latest_signal_time_at_run",
            "latest_signal_at_run",
            *paper_ledger.columns.tolist(),
        ]
        empty_history = pd.DataFrame(columns=empty_columns)
        empty_history.to_csv(ensure_parent_dir(trade_history_output), index=False)
        return empty_history

    history_rows = paper_ledger.copy()
    history_rows["history_trade_key"] = history_rows.apply(build_history_trade_key, axis=1)
    history_rows["snapshot_run_id"] = snapshot_run_id
    history_rows["snapshot_created_at"] = snapshot_created_at
    history_rows["source_start"] = str(source_meta.get("start", ""))
    history_rows["source_end"] = str(report.get("mt5_live_source", {}).get("end", ""))
    history_rows["latest_signal_time_at_run"] = str(report.get("latest_signal", {}).get("time", ""))
    history_rows["latest_signal_at_run"] = str(report.get("latest_signal", {}).get("recommended_trade", ""))

    ordered_columns = [
        "history_trade_key",
        "snapshot_run_id",
        "snapshot_created_at",
        "source_start",
        "source_end",
        "latest_signal_time_at_run",
        "latest_signal_at_run",
        *paper_ledger.columns.tolist(),
    ]
    history_rows = history_rows.loc[:, ordered_columns]

    if existing.empty:
        combined = history_rows.copy()
    else:
        existing = existing.reindex(columns=ordered_columns, fill_value="")
        combined = pd.concat([existing, history_rows], ignore_index=True)

    combined = sanitize_trade_history_frame(combined)

    combined.to_csv(ensure_parent_dir(trade_history_output), index=False)
    return combined


def write_run_snapshot(
    predictions: pd.DataFrame,
    overlays: dict[str, Any],
    paper_ledger: pd.DataFrame,
    final_trade_ledger: pd.DataFrame,
    feedback: pd.DataFrame,
    learning_status: dict[str, Any],
    report: dict[str, Any],
    snapshot_dir: str,
    snapshot_run_id: str,
    snapshot_created_at: str,
) -> dict[str, str]:
    snapshot_root = resolve_repo_path(snapshot_dir) / snapshot_run_id
    snapshot_root.mkdir(parents=True, exist_ok=True)

    predictions_path = snapshot_root / "predictions.csv"
    overlays_path = snapshot_root / "overlays.json"
    paper_ledger_path = snapshot_root / "paper_ledger.csv"
    final_trade_ledger_path = snapshot_root / "final_trade_ledger.csv"
    feedback_path = snapshot_root / "learning_feedback.csv"
    learning_status_path = snapshot_root / "learning_status.json"
    report_path = snapshot_root / "report.json"

    predictions_snapshot = predictions.copy()
    predictions_snapshot["time"] = predictions_snapshot["time"].astype(str)
    predictions_snapshot.to_csv(predictions_path, index=False)
    json_dump(overlays, overlays_path)
    paper_ledger.to_csv(paper_ledger_path, index=False)
    final_trade_ledger.to_csv(final_trade_ledger_path, index=False)
    feedback.to_csv(feedback_path, index=False)
    json_dump(learning_status, learning_status_path)
    json_dump(report, report_path)

    snapshot_meta = {
        "run_id": snapshot_run_id,
        "created_at_utc": snapshot_created_at,
        "latest_signal_time": str(report.get("latest_signal", {}).get("time", "")),
        "latest_signal": str(report.get("latest_signal", {}).get("recommended_trade", "")),
        "paper_trade_count": int(report.get("paper_trading", {}).get("trade_count", 0)),
        "ending_equity": float(report.get("paper_trading", {}).get("ending_equity", 0.0)),
        "files": {
            "predictions": display_path(predictions_path),
            "overlays": display_path(overlays_path),
            "paper_ledger": display_path(paper_ledger_path),
            "final_trade_ledger": display_path(final_trade_ledger_path),
            "learning_feedback": display_path(feedback_path),
            "learning_status": display_path(learning_status_path),
            "report": display_path(report_path),
        },
    }
    snapshot_meta_path = json_dump(snapshot_meta, snapshot_root / "snapshot_meta.json")
    return {
        "snapshot_root": display_path(snapshot_root),
        "snapshot_meta": display_path(snapshot_meta_path),
    }


def build_report(
    live_report: dict[str, Any],
    live_account_context: dict[str, Any] | None,
    mt5_runtime_health: dict[str, Any],
    predictions: pd.DataFrame,
    overlays: dict[str, Any],
    paper_ledger: pd.DataFrame,
    paper_summary: dict[str, Any],
    feedback: pd.DataFrame,
    learning_status: dict[str, Any],
    manual_override: dict[str, Any],
    predictions_output: str,
    overlays_output: str,
    paper_output: str,
    trade_directive_output: str,
    feedback_output: str,
    learning_status_output: str,
    report_output: str,
    trade_history_output: str,
    final_ledger_output: str,
    snapshot_dir: str,
    snapshot_run_id: str,
    snapshot_created_at: str,
) -> dict[str, Any]:
    latest = predictions.iloc[-1]
    prediction_times = pd.to_datetime(predictions["time"], errors="coerce") if "time" in predictions.columns else pd.Series(dtype="datetime64[ns]")
    prediction_window_start = ""
    prediction_window_end = ""
    if not prediction_times.empty and prediction_times.notna().any():
        prediction_window_start = str(prediction_times.min())
        prediction_window_end = str(prediction_times.max())
    confidence_report = live_report.get("confidence") or {}
    baseline_horizons = {
        "5": {
            "signal_rate": float((predictions["signal_5m"] != 0).mean()),
            "trade_precision": None,
            "trade_recall": None,
            "overall_log_loss": None,
            "overall_brier": None,
            "confidence_bins": [],
        },
        "15": {
            "signal_rate": float((predictions["signal_15m"] != 0).mean()),
            "trade_precision": float(confidence_report.get("selected_accuracy", 0.0)),
            "trade_recall": None,
            "overall_log_loss": None,
            "overall_brier": None,
            "confidence_bins": [],
        },
        "60": {
            "signal_rate": float((predictions["signal_60m"] != 0).mean()),
            "trade_precision": None,
            "trade_recall": None,
            "overall_log_loss": None,
            "overall_brier": None,
            "confidence_bins": [],
        },
    }

    report = {
        "mode": "mt5_live_paper",
        "run_metadata": {
            "run_id": snapshot_run_id,
            "created_at_utc": snapshot_created_at,
        },
        "notes": [
            "This cockpit is running on MT5 local live data from the desktop exporter on the same machine.",
            "5m and 60m layers are proxy confirmations derived from MT5 live feature state until the heavier multi-horizon MT5 research model is enabled.",
            "Paper trading is simulated locally from recent MT5 bars. The MT5 EA can mirror only READY directives from that same paper signal.",
            "Any MT5 broker execution is intentionally demo-only and stays separate from Streamlit metrics and learning feedback.",
            *list(live_report.get("notes", [])),
        ],
        "dataset_summary": {
            "standardized_rows": int(live_report.get("standardized_rows", 0)),
            "research_rows": int(len(predictions)),
            "label_rows": int(live_report.get("validation_rows", 0)),
            "prediction_rows": int(len(predictions)),
            "paper_trade_rows": int(len(paper_ledger)),
            "current_run_paper_trade_rows": int(len(paper_ledger)),
        },
        "mt5_live_source": {
            "provider": live_report.get("source_provider", "mt5_export"),
            "symbol": live_report.get("symbol", "XAUUSD"),
            "interval": live_report.get("interval", "M1"),
            "rows": live_report.get("source_rows", 0),
            "start": live_report.get("source_start"),
            "end": live_report.get("source_end"),
            "volume_mode": live_report.get("volume_mode", "volume"),
            "volume_note": live_report.get("volume_note", ""),
            "hunt_timezone": live_report.get("hunt_timezone", "UTC"),
            "hunt_windows": list(live_report.get("hunt_windows", [])),
            "raw_output": live_report.get("artifacts", {}).get("raw_output", display_path(DEFAULT_MT5_LIVE_RAW_INPUT)),
        },
        "mt5_account": {
            "connected": bool(live_account_context),
            "login": int((live_account_context or {}).get("login", 0) or 0),
            "server": str((live_account_context or {}).get("server", "") or ""),
            "currency": str((live_account_context or {}).get("currency", "") or ""),
            "balance": float((live_account_context or {}).get("balance", 0.0) or 0.0),
            "equity": float((live_account_context or {}).get("equity", 0.0) or 0.0),
            "leverage": int((live_account_context or {}).get("leverage", 0) or 0),
            "contract_size": float((live_account_context or {}).get("contract_size", 0.0) or 0.0),
        },
        "mt5_runtime_health": mt5_runtime_health,
        "prediction_window": {
            "start": prediction_window_start,
            "end": prediction_window_end,
            "rows": int(len(predictions)),
        },
        "latest_signal": {
            "time": str(latest["time"]),
            "signal": str(latest["recommended_trade"]),
            "recommended_trade": str(latest["recommended_trade"]),
            "gate_status": str(latest["gate_status"]),
            "session_name": str(latest["session_name"]),
            "setup_score": float(latest["setup_score"]),
            "expected_value": float(latest["expected_value"]),
            "entry_price": float(latest["entry_price"]),
            "stop_loss": float(latest["stop_loss"]),
            "tp1": float(latest["tp1"]),
            "tp2": float(latest["tp2"]),
            "reason_blocked": str(latest["reason_blocked"]),
            "paper_status": str(latest["paper_status"]),
            "paper_reason_blocked": str(latest["paper_reason_blocked"]),
            "regime": str(latest["regime"]),
            "confluence_tags": str(latest["confluence_tags"]),
            "signal_5m": int(latest["signal_5m"]),
            "signal_15m": int(latest["signal_15m"]),
            "signal_60m": int(latest["signal_60m"]),
        },
        "current_run_paper_trading": paper_summary,
        "paper_trading": paper_summary,
        "backtest": paper_summary,
        "learning": learning_status,
        "manual_override": {
            "enabled": bool(manual_override.get("enabled", False)),
            "scope": str(manual_override.get("scope", "paper_only")),
            "remaining_credits": int(manual_override.get("remaining_credits", 0)),
            "initial_credits": int(manual_override.get("initial_credits", 0)),
            "used_signal_times": [str(value) for value in manual_override.get("used_signal_times", [])],
            "start_after": str(manual_override.get("start_after", "")),
            "last_used_at": str(manual_override.get("last_used_at", "")),
            "allowed_risk_blockers": [str(value) for value in manual_override.get("allowed_risk_blockers", [])],
            "note": str(manual_override.get("note", "")),
        },
        "execution_policy": {
            "broker_execution_mode": "demo_only",
            "trade_setup_source": "paper_trade_directive",
            "learning_source": "paper_trading_only",
            "streamlit_scope": "paper_research_only",
            "ingests_mt5_broker_trades": False,
            "requires_trade_directive": True,
            "broker_mirror_trigger": "latest directive row with gate_status=READY and paper_status=SIGNAL_READY",
            "trade_directive_synced_targets": [],
            "trade_directive_sync_error": "",
        },
        "baseline": {
            "walk_forward_splits": "live_window",
            "horizons": baseline_horizons,
        },
        "neural": {
            "status": "disabled",
            "parameter_count": 0,
            "trainable_parameter_count": 0,
            "notes": ["The local MT5 paper-trading path is currently using the fast LightGBM live baseline only."],
        },
        "ensemble": {
            "status": "mt5_live_proxy_confirmations",
        },
        "session_profile": build_session_profile(predictions),
        "overlay_counts": {key: int(len(value)) for key, value in overlays.items()},
        "outputs": {
            "predictions_output": display_path(predictions_output),
            "overlays_output": display_path(overlays_output),
            "paper_output": display_path(paper_output),
            "trade_history_output": display_path(trade_history_output),
            "final_trade_ledger_output": display_path(final_ledger_output),
            "trade_directive_output": display_path(trade_directive_output),
            "feedback_output": display_path(feedback_output),
            "learning_status_output": display_path(learning_status_output),
            "report_output": display_path(report_output),
            "snapshot_root": display_path(resolve_repo_path(snapshot_dir) / snapshot_run_id),
        },
    }
    return report


def sync_paper_capital_to_mt5_account(report: dict[str, Any]) -> dict[str, Any]:
    mt5_account = report.get("mt5_account", {}) or {}
    if not mt5_account.get("connected"):
        return report

    balance = float(mt5_account.get("balance", 0.0) or 0.0)
    equity = float(mt5_account.get("equity", balance) or balance)
    if balance <= 0.0:
        return report

    for section_name in ("current_run_paper_trading", "paper_trading", "frozen_paper_trading", "backtest"):
        section = report.get(section_name)
        if not isinstance(section, dict):
            continue
        section["mt5_balance"] = balance
        section["mt5_equity"] = equity
        section["capital_source"] = "mt5_account_equity"
        section["display_starting_equity"] = balance
        section["display_ending_equity"] = equity
        section["display_net_pnl_cash"] = equity - balance
    return report


def write_outputs(
    predictions: pd.DataFrame,
    overlays: dict[str, Any],
    paper_ledger: pd.DataFrame,
    feedback: pd.DataFrame,
    learning_status: dict[str, Any],
    report: dict[str, Any],
    config_payload: dict[str, Any],
    predictions_output: str,
    overlays_output: str,
    paper_output: str,
    trade_history_output: str,
    final_ledger_output: str,
    trade_directive_output: str,
    feedback_output: str,
    learning_status_output: str,
    report_output: str,
    snapshot_dir: str,
    snapshot_run_id: str,
    snapshot_created_at: str,
) -> None:
    predictions_to_write = predictions.copy()
    predictions_to_write["time"] = predictions_to_write["time"].astype(str)
    predictions_output_path = ensure_parent_dir(predictions_output)
    predictions_to_write.to_csv(predictions_output_path, index=False)
    json_dump(overlays, overlays_output)
    paper_ledger.to_csv(ensure_parent_dir(paper_output), index=False)
    trade_history = append_trade_history(
        paper_ledger=paper_ledger,
        trade_history_output=trade_history_output,
        snapshot_run_id=snapshot_run_id,
        snapshot_created_at=snapshot_created_at,
        report=report,
    )
    current_run_summary = report.get("current_run_paper_trading", report.get("paper_trading", {}))
    final_trade_ledger, final_summary = materialize_final_trade_ledger(
        trade_history=trade_history,
        final_ledger_output=final_ledger_output,
        starting_equity=float(current_run_summary.get("starting_equity", 0.0)),
        open_trade_summary=current_run_summary.get("open_trade"),
        risk_per_trade=float(current_run_summary.get("risk_per_trade", 0.0)),
        position_sizing_mode=str(current_run_summary.get("position_sizing_mode", "") or "equity_risk"),
        assumed_contract_size=float(current_run_summary.get("assumed_contract_size", 100.0) or 100.0),
    )
    report["paper_trading"] = final_summary
    report["backtest"] = current_run_summary
    report["frozen_paper_trading"] = final_summary
    report.setdefault("dataset_summary", {})["saved_history_rows"] = int(len(trade_history))
    report["dataset_summary"]["paper_trade_rows"] = int(len(final_trade_ledger))
    report["dataset_summary"]["current_run_paper_trade_rows"] = int(len(paper_ledger))
    directive_source = pd.read_csv(predictions_output_path)
    trade_directive = build_trade_directive_frame(directive_source, config_payload)
    write_csv_atomically(trade_directive, trade_directive_output, index=False)
    synced_directive_targets: list[str] = []
    sync_error = ""
    try:
        synced_directive_targets = [item["target"] for item in sync_mt5_file_artifacts([trade_directive_output])]
    except Exception as exc:
        sync_error = str(exc)
    report.setdefault("outputs", {})["trade_directive_output"] = display_path(trade_directive_output)
    report.setdefault("execution_policy", {})["trade_directive_synced_targets"] = synced_directive_targets
    report.setdefault("execution_policy", {})["trade_directive_sync_error"] = sync_error
    sync_paper_capital_to_mt5_account(report)
    feedback.to_csv(ensure_parent_dir(feedback_output), index=False)
    json_dump(learning_status, learning_status_output)
    json_dump(report, report_output)
    write_run_snapshot(
        predictions=predictions,
        overlays=overlays,
        paper_ledger=paper_ledger,
        final_trade_ledger=final_trade_ledger,
        feedback=feedback,
        learning_status=learning_status,
        report=report,
        snapshot_dir=snapshot_dir,
        snapshot_run_id=snapshot_run_id,
        snapshot_created_at=snapshot_created_at,
    )


def run_mt5_research_pipeline(
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    count: int = 5000,
    source_mode: str = "auto",
    terminal_env: str = "MT5_TERMINAL_PATH",
    login_env: str = "MT5_LOGIN",
    password_env: str = "MT5_PASSWORD",
    server_env: str = "MT5_SERVER",
    export_input: str = "",
    prefer_real_volume: bool = False,
    config: str = DEFAULT_MT5_RESEARCH_CONFIG,
    predictions_output: str = DEFAULT_MT5_RESEARCH_PREDICTIONS_OUTPUT,
    overlays_output: str = DEFAULT_MT5_RESEARCH_OVERLAYS_OUTPUT,
    paper_output: str = DEFAULT_MT5_RESEARCH_PAPER_LEDGER_OUTPUT,
    trade_history_output: str = DEFAULT_MT5_RESEARCH_TRADE_HISTORY_OUTPUT,
    final_ledger_output: str = DEFAULT_MT5_RESEARCH_FINAL_LEDGER_OUTPUT,
    trade_directive_output: str = DEFAULT_MT5_TRADE_DIRECTIVE_OUTPUT,
    snapshot_dir: str = DEFAULT_MT5_RESEARCH_SNAPSHOT_DIR,
    feedback_output: str = DEFAULT_MT5_RESEARCH_FEEDBACK_OUTPUT,
    learning_status_output: str = DEFAULT_MT5_RESEARCH_LEARNING_STATUS_OUTPUT,
    manual_override_output: str = DEFAULT_MT5_RESEARCH_MANUAL_OVERRIDE_OUTPUT,
    report_output: str = DEFAULT_MT5_RESEARCH_REPORT_OUTPUT,
    skip_onnx_runtime_check: bool = False,
    skip_confidence_analysis: bool = False,
    skip_backtest: bool = False,
) -> dict[str, Any]:
    config_payload = load_config(config)
    live_account_context = load_live_account_context(
        symbol=symbol,
        terminal_env=terminal_env,
        login_env=login_env,
        password_env=password_env,
        server_env=server_env,
    )
    mt5_runtime_health = build_mt5_runtime_health()
    snapshot_timestamp = pd.Timestamp.now(tz="UTC")
    snapshot_created_at = snapshot_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot_run_id = snapshot_timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    live_report = run_live_pipeline(
        symbol=symbol,
        timeframe=timeframe,
        count=count,
        source_mode=source_mode,
        terminal_env=terminal_env,
        login_env=login_env,
        password_env=password_env,
        server_env=server_env,
        export_input=export_input,
        prefer_real_volume=prefer_real_volume,
        report_output=DEFAULT_MT5_LIVE_REPORT_PATH,
        hunt_timezone=str(config_payload.get("trading", {}).get("hunt_timezone", "UTC")),
        hunt_windows=config_payload.get("trading", {}).get("hunt_windows"),
        skip_onnx_runtime_check=skip_onnx_runtime_check,
        skip_confidence_analysis=skip_confidence_analysis,
        skip_backtest=skip_backtest,
    )
    source_timezone = str(live_report.get("source_timezone", "UTC") or "UTC")

    validation_df = pd.read_csv(resolve_repo_path(DEFAULT_MT5_LIVE_MT5_VALIDATION_OUTPUT))
    feature_df = pd.read_csv(resolve_repo_path(DEFAULT_MT5_LIVE_FEATURE_OUTPUT))
    validation_df, excluded_live_bar_time = drop_latest_live_bar(validation_df)
    feature_df, _ = drop_latest_live_bar(feature_df)
    base_columns = ["time", "open", "high", "low", "close", "volume"]
    feature_subset = feature_df.loc[:, [column for column in base_columns if column in feature_df.columns]].copy()
    validation_df = validation_df.merge(feature_subset, on="time", how="left")
    raw_df = pd.read_csv(resolve_repo_path(DEFAULT_MT5_LIVE_RAW_INPUT))
    raw_df, _ = drop_latest_live_bar(raw_df)
    manual_override = load_manual_override(manual_override_output)
    predictions = build_prediction_frame(validation_df, config_payload, source_timezone=source_timezone)
    overlays = build_overlay_objects(predictions)
    predictions, paper_ledger, paper_summary, manual_override = simulate_paper_trades(
        predictions,
        raw_df,
        config_payload,
        manual_override,
        source_timezone=source_timezone,
        starting_equity_override=(live_account_context or {}).get("balance"),
        contract_size_override=(live_account_context or {}).get("contract_size"),
    )
    feedback = build_learning_feedback(predictions, paper_ledger)
    learning_status = build_learning_status(feedback, config_payload)
    json_dump(manual_override, manual_override_output)
    report = build_report(
        live_report=live_report,
        live_account_context=live_account_context,
        mt5_runtime_health=mt5_runtime_health,
        predictions=predictions,
        overlays=overlays,
        paper_ledger=paper_ledger,
        paper_summary=paper_summary,
        feedback=feedback,
        learning_status=learning_status,
        manual_override=manual_override,
        predictions_output=predictions_output,
        overlays_output=overlays_output,
        paper_output=paper_output,
        trade_directive_output=trade_directive_output,
        feedback_output=feedback_output,
        learning_status_output=learning_status_output,
        report_output=report_output,
        trade_history_output=trade_history_output,
        final_ledger_output=final_ledger_output,
        snapshot_dir=snapshot_dir,
        snapshot_run_id=snapshot_run_id,
        snapshot_created_at=snapshot_created_at,
    )
    if excluded_live_bar_time:
        report.setdefault("notes", []).append(
            "The latest still-forming MT5 M1 bar was excluded from paper trading and the MT5 trade directive "
            f"to keep broker execution aligned with closed candles ({excluded_live_bar_time} UTC)."
        )
    write_outputs(
        predictions,
        overlays,
        paper_ledger,
        feedback,
        learning_status,
        report,
        config_payload,
        predictions_output,
        overlays_output,
        paper_output,
        trade_history_output,
        final_ledger_output,
        trade_directive_output,
        feedback_output,
        learning_status_output,
        report_output,
        snapshot_dir,
        snapshot_run_id,
        snapshot_created_at,
    )
    return report


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return run_mt5_research_pipeline(
        symbol=args.symbol,
        timeframe=args.timeframe,
        count=args.count,
        source_mode=args.source_mode,
        terminal_env=args.terminal_env,
        login_env=args.login_env,
        password_env=args.password_env,
        server_env=args.server_env,
        export_input=args.export_input,
        prefer_real_volume=args.prefer_real_volume,
        config=args.config,
        predictions_output=args.predictions_output,
        overlays_output=args.overlays_output,
        paper_output=args.paper_output,
        trade_history_output=args.trade_history_output,
        final_ledger_output=args.final_ledger_output,
        trade_directive_output=args.trade_directive_output,
        snapshot_dir=args.snapshot_dir,
        feedback_output=args.feedback_output,
        learning_status_output=args.learning_status_output,
        manual_override_output=args.manual_override_output,
        report_output=args.report_output,
        skip_onnx_runtime_check=args.skip_onnx_runtime_check,
        skip_confidence_analysis=args.skip_confidence_analysis,
        skip_backtest=args.skip_backtest,
    )


def print_research_summary(report: dict[str, Any]) -> None:
    latest = report["latest_signal"]
    paper = report["paper_trading"]
    dataset = report["dataset_summary"]
    source = report["mt5_live_source"]
    override = report.get("manual_override", {})

    print()
    print("MT5 local cockpit report ready.")
    print(f"Source      : {source['provider']} {source['symbol']} {source['start']} -> {source['end']}")
    print(f"Predictions : {dataset['prediction_rows']}")
    print(f"Paper trades: {paper['trade_count']}")
    display_capital = float(paper.get("display_ending_equity", paper.get("ending_equity", 0.0)))
    display_pnl = float(paper.get("display_net_pnl_cash", paper.get("net_pnl_cash", 0.0)))
    print(f"Capital     : {display_capital:.2f} ({display_pnl:+.2f})")
    print(f"Win/Loss    : {int(paper.get('win_count', 0))}/{int(paper.get('loss_count', 0))}")
    print(f"Override    : {int(override.get('remaining_credits', 0))} credits remaining")
    print(f"Learning    : {'READY' if report['learning']['retrain_ready'] else 'NOT_READY'} ({', '.join(report['learning']['retrain_blockers']) or 'all checks passed'})")
    print(f"Latest      : {latest['signal']} @ {latest['time']}")
    print(f"Gate        : {latest['gate_status']} | {latest['reason_blocked'] or 'tradeable'}")
    print(f"Outputs     : {report['outputs']['report_output']}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print("=" * 70)
    print("XAUUSD MT5 RESEARCH PAPER-TRADING PIPELINE")
    print("=" * 70)
    print(f"Symbol     : {normalize_mt5_symbol(args.symbol)}")
    print(f"Timeframe  : {args.timeframe}")
    print(f"Count      : {args.count}")
    print(f"Source     : {args.source_mode}")
    print()

    report = run_from_args(args)
    print_research_summary(report)


if __name__ == "__main__":
    main()
