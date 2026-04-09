"""
Helpers for loading research cockpit artifacts and building Streamlit visuals.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from pipeline_contract import (
    DEFAULT_MT5_RESEARCH_OVERLAYS_OUTPUT,
    DEFAULT_MT5_RESEARCH_PAPER_LEDGER_OUTPUT,
    DEFAULT_MT5_RESEARCH_PREDICTIONS_OUTPUT,
    DEFAULT_MT5_RESEARCH_REPORT_OUTPUT,
    DEFAULT_MT5_RESEARCH_FINAL_LEDGER_OUTPUT,
    DEFAULT_MT5_RESEARCH_TRADE_HISTORY_OUTPUT,
    DEFAULT_RESEARCH_OVERLAYS_OUTPUT,
    DEFAULT_RESEARCH_PAPER_LEDGER_OUTPUT,
    DEFAULT_RESEARCH_PREDICTIONS_OUTPUT,
    DEFAULT_RESEARCH_REPORT_OUTPUT,
    display_path,
    resolve_repo_path,
)
from mt5_client import DEFAULT_MT5_FILES_DIR, fetch_recent_rates, load_exported_rates_csv, load_mt5_account_snapshot


SAFE_RESEARCH_OUTPUT_ROOTS = (
    resolve_repo_path("data/research"),
    resolve_repo_path("data/live/research_mt5"),
)
DEFAULT_MT5_BROKER_TRADE_LOG_PATH = DEFAULT_MT5_FILES_DIR / "logs" / "xauusd_ai_trades.csv"
DEFAULT_MT5_ACCOUNT_SNAPSHOT_PATH = DEFAULT_MT5_FILES_DIR / "config" / "mt5_account_snapshot.csv"


def _is_safe_research_output(path: Path) -> bool:
    resolved = path.resolve()
    for root in SAFE_RESEARCH_OUTPUT_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _resolve_safe_output_path(path_like: str | Path | None, fallback: str | Path) -> Path:
    candidate = resolve_repo_path(path_like or fallback)
    if _is_safe_research_output(candidate):
        return candidate
    return resolve_repo_path(fallback)


def _resolve_safe_snapshot_root(path_like: str | Path | None) -> Path | None:
    if not path_like:
        return None
    candidate = resolve_repo_path(path_like)
    if _is_safe_research_output(candidate):
        return candidate
    return None


def _load_json(path_like: str | Path) -> dict[str, Any]:
    with resolve_repo_path(path_like).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_research_bundle(report_path: str = DEFAULT_RESEARCH_REPORT_OUTPUT) -> dict[str, Any]:
    live_report = _load_json(report_path)
    live_outputs = live_report.get("outputs", {})
    snapshot_root = _resolve_safe_snapshot_root(live_outputs.get("snapshot_root", ""))
    report = live_report
    mode = str(report.get("mode", "")).strip().lower()
    is_mt5_live_paper = mode == "mt5_live_paper"
    default_predictions_path = DEFAULT_MT5_RESEARCH_PREDICTIONS_OUTPUT if is_mt5_live_paper else DEFAULT_RESEARCH_PREDICTIONS_OUTPUT
    default_overlays_path = DEFAULT_MT5_RESEARCH_OVERLAYS_OUTPUT if is_mt5_live_paper else DEFAULT_RESEARCH_OVERLAYS_OUTPUT
    default_paper_path = DEFAULT_MT5_RESEARCH_PAPER_LEDGER_OUTPUT if is_mt5_live_paper else DEFAULT_RESEARCH_PAPER_LEDGER_OUTPUT
    default_trade_history_path = DEFAULT_MT5_RESEARCH_TRADE_HISTORY_OUTPUT
    default_final_trade_ledger_path = DEFAULT_MT5_RESEARCH_FINAL_LEDGER_OUTPUT
    outputs = report.get("outputs", {})
    snapshot_root = _resolve_safe_snapshot_root(outputs.get("snapshot_root", "")) or snapshot_root

    def prefer_snapshot_artifact(filename: str, fallback: Path) -> Path:
        if snapshot_root is None:
            return fallback
        candidate = snapshot_root / filename
        if candidate.exists():
            return candidate
        return fallback

    predictions_path = prefer_snapshot_artifact(
        "predictions.csv",
        _resolve_safe_output_path(outputs.get("predictions_output", default_predictions_path), default_predictions_path),
    )
    overlays_path = prefer_snapshot_artifact(
        "overlays.json",
        _resolve_safe_output_path(outputs.get("overlays_output", default_overlays_path), default_overlays_path),
    )
    paper_path = prefer_snapshot_artifact(
        "paper_ledger.csv",
        _resolve_safe_output_path(outputs.get("paper_output", default_paper_path), default_paper_path),
    )
    trade_history_path = _resolve_safe_output_path(
        outputs.get("trade_history_output", default_trade_history_path),
        default_trade_history_path,
    )
    final_trade_ledger_path = prefer_snapshot_artifact(
        "final_trade_ledger.csv",
        _resolve_safe_output_path(outputs.get("final_trade_ledger_output", default_final_trade_ledger_path), default_final_trade_ledger_path),
    )

    predictions = pd.read_csv(predictions_path)
    if "time" in predictions.columns:
        predictions["time"] = pd.to_datetime(predictions["time"])
    if paper_path.exists() and paper_path.stat().st_size > 0:
        try:
            paper_ledger = pd.read_csv(paper_path)
        except EmptyDataError:
            paper_ledger = pd.DataFrame()
        for column in ("signal_time", "entry_time", "exit_time"):
            if column in paper_ledger.columns:
                paper_ledger[column] = pd.to_datetime(paper_ledger[column], errors="coerce")
    else:
        paper_ledger = pd.DataFrame()

    if trade_history_path.exists() and trade_history_path.stat().st_size > 0:
        try:
            trade_history = pd.read_csv(trade_history_path)
        except EmptyDataError:
            trade_history = pd.DataFrame()
        for column in ("signal_time", "entry_time", "exit_time", "snapshot_created_at"):
            if column in trade_history.columns:
                trade_history[column] = pd.to_datetime(trade_history[column], errors="coerce")
    else:
        trade_history = pd.DataFrame()

    if final_trade_ledger_path.exists() and final_trade_ledger_path.stat().st_size > 0:
        try:
            final_trade_ledger = pd.read_csv(final_trade_ledger_path)
        except EmptyDataError:
            final_trade_ledger = pd.DataFrame()
        for column in ("signal_time", "entry_time", "exit_time", "snapshot_created_at", "source_end"):
            if column in final_trade_ledger.columns:
                final_trade_ledger[column] = pd.to_datetime(final_trade_ledger[column], errors="coerce")
    else:
        final_trade_ledger = trade_history.copy()

    broker_trade_log_path = DEFAULT_MT5_BROKER_TRADE_LOG_PATH
    if broker_trade_log_path.exists() and broker_trade_log_path.stat().st_size > 0:
        try:
            broker_trade_log = pd.read_csv(broker_trade_log_path)
        except EmptyDataError:
            broker_trade_log = pd.DataFrame()
        if "time_utc" in broker_trade_log.columns:
            broker_trade_log["time_utc"] = pd.to_datetime(broker_trade_log["time_utc"], errors="coerce", utc=True)
    else:
        broker_trade_log = pd.DataFrame()

    if is_mt5_live_paper and DEFAULT_MT5_ACCOUNT_SNAPSHOT_PATH.exists():
        try:
            live_snapshot = load_mt5_account_snapshot()
        except Exception:
            live_snapshot = None
        if isinstance(live_snapshot, dict) and live_snapshot:
            mt5_account = dict(report.get("mt5_account", {}) or {})
            mt5_account.update(
                {
                    "connected": True,
                    "login": int(live_snapshot.get("login", 0) or 0),
                    "server": str(live_snapshot.get("server", "") or ""),
                    "currency": str(live_snapshot.get("currency", "") or ""),
                    "balance": float(live_snapshot.get("balance", 0.0) or 0.0),
                    "equity": float(live_snapshot.get("equity", 0.0) or 0.0),
                    "leverage": int(live_snapshot.get("leverage", 0) or 0),
                    "contract_size": float(live_snapshot.get("contract_size", 0.0) or 0.0),
                }
            )
            report["mt5_account"] = mt5_account

            balance = float(mt5_account.get("balance", 0.0) or 0.0)
            equity = float(mt5_account.get("equity", 0.0) or 0.0)
            for section_name in ("paper_trading", "current_run_paper_trading", "frozen_paper_trading"):
                section = report.get(section_name)
                if not isinstance(section, dict):
                    continue
                section["mt5_balance"] = balance
                section["mt5_equity"] = equity
                section["capital_source"] = "mt5_account_equity"
                section["display_starting_equity"] = balance
                section["display_ending_equity"] = equity
                section["display_net_pnl_cash"] = equity - balance

            runtime_health = dict(report.get("mt5_runtime_health", {}) or {})
            account_snapshot_status = dict(runtime_health.get("account_snapshot", {}) or {})
            account_snapshot_status["path"] = str(DEFAULT_MT5_ACCOUNT_SNAPSHOT_PATH)
            account_snapshot_status["exists"] = True
            account_snapshot_status["status"] = "live"
            account_snapshot_status["stale"] = False
            runtime_health["account_snapshot"] = account_snapshot_status
            summary = dict(runtime_health.get("summary", {}) or {})
            summary["account_snapshot_live"] = True
            runtime_health["summary"] = summary
            report["mt5_runtime_health"] = runtime_health

    overlays = _load_json(overlays_path)
    latest_signal = report.get("latest_signal", {})
    return {
        "report": report,
        "predictions": predictions,
        "paper_ledger": paper_ledger,
        "trade_history": trade_history,
        "final_trade_ledger": final_trade_ledger,
        "broker_trade_log": broker_trade_log,
        "overlays": overlays,
        "latest_signal": latest_signal,
        "paths": {
            "report": display_path(report_path),
            "predictions": display_path(predictions_path),
            "overlays": display_path(overlays_path),
            "paper": display_path(paper_path),
            "trade_history": display_path(trade_history_path),
            "final_trade_ledger": display_path(final_trade_ledger_path),
            "broker_trade_log": display_path(broker_trade_log_path),
        },
    }


def load_live_market_snapshot(
    provider: str = "auto",
    mt5_symbol: str = "XAUUSD",
    symbol: str = "XAU/USD",
    outputsize: int = 120,
) -> dict[str, Any]:
    provider_value = provider.strip().lower()
    payload: dict[str, Any] | None = None
    selected_provider = provider_value
    freshness_note = ""
    stale = False
    stale_seconds = 0

    def format_age(total_seconds: int) -> str:
        minutes, seconds = divmod(max(0, int(total_seconds)), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def try_mt5_bridge_payload() -> dict[str, Any] | None:
        try:
            return fetch_recent_rates(
                symbol=mt5_symbol or symbol or "XAUUSD",
                timeframe="M1",
                count=outputsize,
            )
        except Exception:
            return None

    def try_mt5_export_payload() -> dict[str, Any] | None:
        try:
            return load_exported_rates_csv(
                symbol=mt5_symbol or symbol or "XAUUSD",
                timeframe="M1",
            )
        except Exception:
            return None

    def try_mt5_payload() -> dict[str, Any] | None:
        return try_mt5_bridge_payload() or try_mt5_export_payload()

    if provider_value in {"mt5", "mt5 local"}:
        selected_provider = "mt5"
        payload = try_mt5_payload()
        if payload is None:
            return {
                "enabled": False,
                "error": (
                    "MT5 Local could not be reached. This provider works only on the same machine as a running "
                    "MetaTrader terminal, either through the direct bridge or through the MT5 live exporter CSV."
                ),
            }
    elif provider_value in {"mt5 export", "mt5 exporter", "mt5 exporter csv"}:
        selected_provider = "mt5_export"
        payload = try_mt5_export_payload()
        if payload is None:
            return {
                "enabled": False,
                "error": (
                    "MT5 exporter CSV could not be reached. Make sure the MT5 exporter is writing "
                    "`xauusd_mt5_live.csv` into the local MetaTrader `MQL5/Files` folder."
                ),
            }
    elif provider_value in {"oanda", "twelvedata"}:
        return {
            "enabled": False,
            "error": f"{provider} support is disabled in this project. Use MT5 Local or MT5 Exporter.",
        }
    elif provider_value == "auto":
        mt5_payload = try_mt5_payload()
        if mt5_payload is not None:
            selected_provider = "mt5"
            payload = mt5_payload
        else:
            return {
                "enabled": False,
                "error": "No local MT5 source is available. OANDA and Twelve Data support are disabled.",
            }
    elif not str(selected_provider).startswith("mt5"):
        return {
            "enabled": False,
            "error": f"Unsupported live provider: {provider}",
        }

    values = payload["values"]
    latest = values[-1]
    previous = values[-2] if len(values) > 1 else latest
    change = latest["close"] - previous["close"]
    change_pct = (change / previous["close"] * 100.0) if previous["close"] else 0.0

    frame = pd.DataFrame(values)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    resolved_provider = payload.get("meta", {}).get("provider", selected_provider)
    if resolved_provider == "mt5_export":
        source_path_text = payload.get("meta", {}).get("source_path")
        if source_path_text:
            source_path = Path(source_path_text)
            if source_path.exists():
                source_mtime = source_path.stat().st_mtime
                source_modified_at = datetime.fromtimestamp(source_mtime)
                stale_seconds = max(0, int(datetime.now().timestamp() - source_mtime))
                stale = stale_seconds > 120
                freshness_note = (
                    f"MT5 exporter file last updated {format_age(stale_seconds)} ago "
                    f"at {source_modified_at:%Y-%m-%d %H:%M:%S}."
                )

    return {
        "enabled": True,
        "provider": resolved_provider,
        "meta": payload.get("meta", {}),
        "frame": frame,
        "latest": latest,
        "change": change,
        "change_pct": change_pct,
        "has_volume": payload.get("has_volume", False),
        "volume_note": payload.get(
            "volume_note",
            "Source volume is available." if payload.get("has_volume", False) else "Source volume is not available.",
        ),
        "stale": stale,
        "stale_seconds": stale_seconds,
        "freshness_note": freshness_note,
        "display_symbol": payload.get("meta", {}).get("symbol", mt5_symbol if str(resolved_provider).startswith("mt5") else symbol),
    }


def load_mt5_timeframe_ribbon(mt5_symbol: str = "XAUUSD") -> dict[str, Any]:
    timeframe_rules = [
        ("M1", "1min", "%d %b %H:%M"),
        ("M5", "5min", "%d %b %H:%M"),
        ("M15", "15min", "%d %b %H:%M"),
        ("M30", "30min", "%d %b %H:%M"),
        ("H1", "1h", "%d %b %H:%M"),
        ("H4", "4h", "%d %b %H:%M"),
        ("D1", "1D", "%d %b"),
        ("W1", "W-FRI", "%d %b"),
        ("MN", "ME", "%b %Y"),
    ]

    try:
        payload = fetch_recent_rates(
            symbol=mt5_symbol or "XAUUSD",
            timeframe="M1",
            count=5000,
        )
        frame = payload["frame"].copy()
        source_note = "Direct MT5 live bars aggregated locally across multiple timeframes."
        provider = "mt5_bridge"
    except Exception:
        try:
            fallback = load_exported_rates_csv(symbol=mt5_symbol or "XAUUSD", timeframe="M1")
            frame = fallback["frame"].copy()
            source_note = "MT5 exporter M1 feed aggregated locally across multiple timeframes."
            provider = "mt5_export_resampled"
        except Exception as exc:
            return {
                "enabled": False,
                "error": f"Could not build MT5 timeframe ribbon: {exc}",
            }

    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time").set_index("time")
    items: list[dict[str, Any]] = []

    for label, rule, time_format in timeframe_rules:
        aggregated = (
            frame.resample(rule, label="right", closed="right")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )
        if aggregated.empty:
            continue
        latest = aggregated.iloc[-1]
        previous_close = float(aggregated.iloc[-2]["close"]) if len(aggregated) >= 2 else float(latest["open"])
        open_price = float(latest["open"])
        close_price = float(latest["close"])
        direction_move = close_price - previous_close
        direction = "UP" if direction_move > 0 else "DOWN" if direction_move < 0 else "FLAT"
        items.append(
            {
                "timeframe": label,
                "time": pd.Timestamp(latest["time"]).strftime(time_format),
                "open": open_price,
                "close": close_price,
                "move": direction_move,
                "bar_body_move": close_price - open_price,
                "previous_close": previous_close,
                "direction": direction,
                "color": "#2a8c69" if direction == "UP" else "#b9444b" if direction == "DOWN" else "#b49b57",
            }
        )

    return {
        "enabled": True,
        "provider": provider,
        "symbol": mt5_symbol or "XAUUSD",
        "bars": items,
        "note": source_note,
    }


def build_probability_figure(predictions: pd.DataFrame, bars: int = 180) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    subset = predictions.tail(bars).copy()
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("5m Probability", "15m Probability", "60m Probability"),
    )
    palette = {"short": "#d64550", "hold": "#b49b57", "long": "#2a8c69"}
    for row_idx, horizon in enumerate((5, 15, 60), start=1):
        for label in ("short", "hold", "long"):
            figure.add_trace(
                go.Scatter(
                    x=subset["time"],
                    y=subset[f"prob_{label}_{horizon}m"],
                    mode="lines",
                    name=f"{label.upper()} {horizon}m",
                    line={"color": palette[label], "width": 2},
                    showlegend=row_idx == 1,
                ),
                row=row_idx,
                col=1,
            )
        figure.update_yaxes(range=[0, 1], row=row_idx, col=1)
    figure.update_layout(
        height=820,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="#f4f0e8",
        plot_bgcolor="#fffaf0",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
    )
    return figure


def build_candlestick_figure(predictions: pd.DataFrame, overlays: dict[str, Any], bars: int = 220) -> Any:
    import plotly.graph_objects as go

    subset = predictions.tail(bars).copy()
    visible_times = set(subset["time"].astype(str))
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=subset["time"],
            open=subset["open"],
            high=subset["high"],
            low=subset["low"],
            close=subset["close"],
            name="XAUUSD",
            increasing_line_color="#1f7a5a",
            decreasing_line_color="#9a2f2f",
        )
    )

    for zone in overlays.get("fvg_zones", [])[-80:]:
        if zone["time"] not in visible_times:
            continue
        zone_time = pd.to_datetime(zone["time"])
        zone_end = zone_time + pd.Timedelta(minutes=3)
        fill = "rgba(38, 166, 154, 0.18)" if zone["direction"] == "bullish" else "rgba(214, 69, 80, 0.18)"
        figure.add_shape(
            type="rect",
            x0=zone_time,
            x1=zone_end,
            y0=zone["low"],
            y1=zone["high"],
            xref="x",
            yref="y",
            fillcolor=fill,
            line={"width": 6, "color": fill.replace("0.18", "0.50")},
        )

    for block in overlays.get("order_blocks", [])[-60:]:
        if block["time"] not in visible_times:
            continue
        block_time = pd.to_datetime(block["time"])
        block_end = block_time + pd.Timedelta(minutes=3)
        color = "rgba(30, 136, 229, 0.16)" if block["direction"] == "bullish" else "rgba(168, 82, 209, 0.14)"
        figure.add_shape(
            type="rect",
            x0=block_time,
            x1=block_end,
            y0=block["low"],
            y1=block["high"],
            xref="x",
            yref="y",
            fillcolor=color,
            line={"width": 5, "color": color.replace("0.16", "0.35").replace("0.14", "0.35")},
        )

    for event_name, color in (("bos_events", "#23395d"), ("choch_events", "#ff8c42")):
        events = [item for item in overlays.get(event_name, [])[-80:] if item["time"] in visible_times]
        if not events:
            continue
        figure.add_trace(
            go.Scatter(
                x=[pd.to_datetime(item["time"]) for item in events],
                y=[item.get("level", subset["close"].iloc[-1]) for item in events],
                mode="markers",
                marker={"color": color, "size": 9, "symbol": "diamond"},
                name=event_name.replace("_", " ").upper(),
            )
        )

    entries = [item for item in overlays.get("entry_markers", [])[-80:] if item["time"] in visible_times]
    if entries:
        figure.add_trace(
            go.Scatter(
                x=[pd.to_datetime(item["time"]) for item in entries],
                y=[item["price"] for item in entries],
                mode="markers",
                marker={"color": "#111111", "size": 8, "symbol": "x"},
                name="ENTRY",
            )
        )

    latest = subset.iloc[-1]
    figure.add_trace(
        go.Scatter(
            x=[latest["time"], latest["time"], latest["time"]],
            y=[latest["entry_price"], latest["stop_loss"], latest["tp1"]],
            mode="markers",
            marker={"size": 11, "color": ["#111111", "#d64550", "#2a8c69"]},
            name="Trade Levels",
        )
    )
    figure.update_layout(
        height=700,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        paper_bgcolor="#f4f0e8",
        plot_bgcolor="#fffaf0",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
    )
    return figure


def build_session_profile_figure(report: dict[str, Any]) -> Any:
    import plotly.graph_objects as go

    profile = report.get("session_profile", {})
    sessions = list(profile.keys())
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=sessions,
            y=[profile[name]["bars"] for name in sessions],
            name="Bars",
            marker_color="#d1a054",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sessions,
            y=[profile[name]["avg_session_dominance"] for name in sessions],
            name="Avg Session Dominance",
            mode="lines+markers",
            yaxis="y2",
            line={"color": "#2a8c69", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sessions,
            y=[profile[name]["avg_impulse_strength"] for name in sessions],
            name="Avg Impulse Strength",
            mode="lines+markers",
            yaxis="y2",
            line={"color": "#23395d", "width": 3},
        )
    )
    figure.update_layout(
        height=480,
        paper_bgcolor="#f4f0e8",
        plot_bgcolor="#fffaf0",
        yaxis={"title": "Bars"},
        yaxis2={"title": "Behavior Proxy", "overlaying": "y", "side": "right"},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    return figure


def build_hour_bias_figure(predictions: pd.DataFrame) -> Any:
    import plotly.express as px

    frame = predictions.copy()
    frame["hour"] = pd.to_datetime(frame["time"]).dt.hour
    grouped = (
        frame.groupby("hour", as_index=False)
        .agg(
            avg_buy_pressure=("buy_pressure_proxy", "mean"),
            avg_sell_pressure=("sell_pressure_proxy", "mean"),
            avg_bias=("time_of_day_bias", "mean"),
            avg_expected_value=("expected_value", "mean"),
        )
    )
    melted = grouped.melt(id_vars="hour", var_name="metric", value_name="value")
    figure = px.line(
        melted,
        x="hour",
        y="value",
        color="metric",
        markers=True,
        color_discrete_sequence=["#2a8c69", "#d64550", "#23395d", "#b49b57"],
    )
    figure.update_layout(
        height=460,
        paper_bgcolor="#f4f0e8",
        plot_bgcolor="#fffaf0",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    return figure


def build_equity_curve_figure(report: dict[str, Any]) -> Any:
    import plotly.graph_objects as go

    curve = (
        report.get("current_run_paper_trading", {}).get("equity_curve", [])
        or report.get("paper_trading", {}).get("equity_curve", [])
        or report.get("backtest", {}).get("equity_curve", [])
        or report.get("frozen_paper_trading", {}).get("equity_curve", [])
    )
    if not curve:
        figure = go.Figure()
        figure.update_layout(
            height=360,
            paper_bgcolor="#f4f0e8",
            plot_bgcolor="#fffaf0",
            annotations=[
                {
                    "text": "No actionable trades yet in the current report.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
        return figure

    frame = pd.DataFrame(curve)
    frame["time"] = pd.to_datetime(frame["time"])
    figure = go.Figure(
        data=[
            go.Scatter(
                x=frame["time"],
                y=frame["equity"],
                mode="lines",
                line={"width": 3, "color": "#23395d"},
                name="Equity",
            )
        ]
    )
    figure.update_layout(
        height=360,
        paper_bgcolor="#f4f0e8",
        plot_bgcolor="#fffaf0",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return figure


def build_calibration_figure(report: dict[str, Any], horizon: int = 15) -> Any:
    import plotly.graph_objects as go

    bins = report.get("baseline", {}).get("horizons", {}).get(str(horizon), {}).get("confidence_bins", [])
    frame = pd.DataFrame(bins)
    figure = go.Figure()
    if frame.empty:
        figure.update_layout(
            height=360,
            paper_bgcolor="#f4f0e8",
            plot_bgcolor="#fffaf0",
            annotations=[
                {
                    "text": "No calibration bins available for this horizon yet.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
        return figure

    figure.add_trace(
        go.Bar(
            x=frame["bin_start"].round(2).astype(str) + "-" + frame["bin_end"].round(2).astype(str),
            y=frame["avg_confidence"],
            name="Avg Confidence",
            marker_color="#d1a054",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["bin_start"].round(2).astype(str) + "-" + frame["bin_end"].round(2).astype(str),
            y=frame["accuracy"],
            name="Observed Accuracy",
            mode="lines+markers",
            line={"color": "#2a8c69", "width": 3},
        )
    )
    figure.update_layout(
        height=360,
        paper_bgcolor="#f4f0e8",
        plot_bgcolor="#fffaf0",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis={"range": [0, 1]},
    )
    return figure


def build_model_metric_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = report.get("baseline", {}).get("horizons", {})
    neural = report.get("neural", {}).get("horizons", {})

    for horizon in (5, 15, 60):
        baseline_metrics = baseline.get(str(horizon), {})
        neural_metrics = neural.get(str(horizon), {})
        rows.append(
            {
                "Horizon": f"{horizon}m",
                "Baseline Signal Rate": baseline_metrics.get("signal_rate"),
                "Baseline Trade Precision": baseline_metrics.get("trade_precision"),
                "Baseline Trade Recall": baseline_metrics.get("trade_recall"),
                "Baseline Log Loss": baseline_metrics.get("overall_log_loss"),
                "Baseline Brier": baseline_metrics.get("overall_brier"),
                "Neural Accuracy": neural_metrics.get("accuracy"),
                "Neural Log Loss": neural_metrics.get("log_loss"),
                "Neural Brier": neural_metrics.get("brier"),
            }
        )
    return pd.DataFrame(rows)


def build_learning_checks_table(report: dict[str, Any]) -> pd.DataFrame:
    learning = report.get("learning", {})
    checks = learning.get("checks", [])
    if not checks:
        return pd.DataFrame()
    frame = pd.DataFrame(checks)
    if "passed" in frame.columns:
        frame["passed"] = frame["passed"].map({True: "PASS", False: "BLOCK"})
    return frame


def build_walk_forward_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = report.get("baseline", {}).get("horizons", {})
    for horizon, payload in baseline.items():
        for fold in payload.get("fold_metrics", []):
            rows.append(
                {
                    "Horizon": f"{horizon}m",
                    "Fold": fold["fold"],
                    "Samples": fold["samples"],
                    "Accuracy": fold["accuracy"],
                    "Log Loss": fold["log_loss"],
                    "Brier": fold["brier"],
                }
            )
    return pd.DataFrame(rows)


def build_recent_signal_table(predictions: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    columns = [
        "time",
        "session_name",
        "regime",
        "setup_candidate",
        "signal_name",
        "recommended_trade",
        "gate_status",
        "setup_score",
        "expected_value",
        "prob_short_15m",
        "prob_hold_15m",
        "prob_long_15m",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "reason_blocked",
        "paper_status",
        "paper_reason_blocked",
        "confluence_tags",
    ]
    available_columns = [column for column in columns if column in predictions.columns]
    table = predictions.loc[:, available_columns].tail(limit).copy()
    table["time"] = table["time"].astype(str)
    return table.iloc[::-1].reset_index(drop=True)


def build_blocked_signal_table(predictions: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    reason_blocked = predictions["reason_blocked"] if "reason_blocked" in predictions.columns else pd.Series("", index=predictions.index)
    paper_reason = predictions["paper_reason_blocked"] if "paper_reason_blocked" in predictions.columns else pd.Series("", index=predictions.index)
    columns = [
        "time",
        "session_name",
        "setup_candidate",
        "recommended_trade",
        "reason_blocked",
        "paper_status",
        "paper_reason_blocked",
        "setup_score",
        "expected_value",
        "entry_quality",
        "confluence_tags",
    ]
    available_columns = [column for column in columns if column in predictions.columns]
    table = predictions.loc[
        reason_blocked.fillna("").ne("") | paper_reason.fillna("").ne(""),
        available_columns,
    ].tail(limit).copy()
    if table.empty:
        return table
    table["time"] = table["time"].astype(str)
    return table.iloc[::-1].reset_index(drop=True)


def build_paper_ledger_table(paper_ledger: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    if paper_ledger.empty:
        return paper_ledger
    columns = [
        "trade_id",
        "signal_time",
        "entry_time",
        "exit_time",
        "direction",
        "volume_lots",
        "risk_cash",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "exit_price",
        "session_name",
        "exit_reason",
        "realized_r",
        "pnl_cash",
        "equity_before",
        "equity_after",
        "setup_score",
        "expected_value",
    ]
    table = paper_ledger.loc[:, [column for column in columns if column in paper_ledger.columns]].tail(limit).copy()
    for column in ("signal_time", "entry_time", "exit_time"):
        if column in table.columns:
            table[column] = table[column].astype(str)
    for column in ("volume_lots",):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(3)
    for column in ("risk_cash", "entry_price", "stop_loss", "tp1", "tp2", "exit_price"):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
    table = table.rename(
        columns={
            "pnl_cash": "paper_pnl_cash",
            "equity_before": "paper_equity_before",
            "equity_after": "paper_equity_after",
        }
    )
    return table.iloc[::-1].reset_index(drop=True)


def _summarize_trade_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "trade_count": 0,
            "wins": 0,
            "losses": 0,
            "flats": 0,
            "net_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "win_rate": 0.0,
        }

    realized = pd.to_numeric(frame.get("realized_r", pd.Series(index=frame.index, dtype="float64")), errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(frame.get("pnl_cash", pd.Series(index=frame.index, dtype="float64")), errors="coerce").fillna(0.0)
    wins = int((realized > 0).sum())
    losses = int((realized < 0).sum())
    flats = int((realized == 0).sum())
    trade_count = int(len(frame))
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(pnl[pnl < 0].sum())
    return {
        "trade_count": trade_count,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "net_pnl": float(pnl.sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "win_rate": float(wins / trade_count) if trade_count else 0.0,
    }


def build_trade_period_summary_table(trade_history: pd.DataFrame) -> pd.DataFrame:
    if trade_history.empty:
        return pd.DataFrame()

    frame = trade_history.copy()
    time_column = "exit_time" if "exit_time" in frame.columns and frame["exit_time"].notna().any() else "signal_time"
    frame["_summary_time"] = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
    frame = frame.loc[frame["_summary_time"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["_summary_time"] = frame["_summary_time"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    now = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None)
    today_start = now.normalize()
    week_start = today_start - pd.Timedelta(days=today_start.weekday())
    month_start = today_start.replace(day=1)

    periods = [
        ("Today", frame["_summary_time"] >= today_start),
        ("This Week", frame["_summary_time"] >= week_start),
        ("This Month", frame["_summary_time"] >= month_start),
        ("All Saved", pd.Series(True, index=frame.index)),
    ]

    rows: list[dict[str, Any]] = []
    for label, mask in periods:
        summary = _summarize_trade_frame(frame.loc[mask].copy())
        rows.append(
            {
                "Period": label,
                "Trades": summary["trade_count"],
                "Wins": summary["wins"],
                "Losses": summary["losses"],
                "Flat": summary["flats"],
                "Win Rate": summary["win_rate"],
                "Net PnL": summary["net_pnl"],
                "Gross Profit": summary["gross_profit"],
                "Gross Loss": summary["gross_loss"],
            }
        )

    return pd.DataFrame(rows)


def build_saved_trade_history_table(trade_history: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    if trade_history.empty:
        return trade_history

    ordered = trade_history.copy()
    if "signal_time" in ordered.columns and "history_trade_key" in ordered.columns:
        missing_signal_time = ordered["signal_time"].isna()
        if missing_signal_time.any():
            inferred_signal_time = pd.to_datetime(
                ordered.loc[missing_signal_time, "history_trade_key"].astype(str).str.split("|").str[0],
                errors="coerce",
                utc=True,
            )
            ordered.loc[missing_signal_time, "signal_time"] = inferred_signal_time
    if "exit_time" in ordered.columns:
        exit_times = pd.to_datetime(ordered["exit_time"], errors="coerce", utc=True)
        ordered = ordered.loc[exit_times.notna()].copy()
    if ordered.empty:
        return pd.DataFrame()

    ordered = ordered.sort_values(
        by=[column for column in ("snapshot_created_at", "exit_time", "signal_time") if column in trade_history.columns],
        ascending=True,
    )
    columns = [
        "snapshot_created_at",
        "signal_time",
        "entry_time",
        "exit_time",
        "direction",
        "volume_lots",
        "risk_cash",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "exit_price",
        "session_name",
        "exit_reason",
        "realized_r",
        "pnl_cash",
        "equity_after",
        "setup_score",
        "expected_value",
    ]
    table = ordered.loc[:, [column for column in columns if column in ordered.columns]].tail(limit).copy()
    for column in ("snapshot_created_at", "signal_time", "entry_time", "exit_time"):
        if column in table.columns:
            table[column] = (
                table[column]
                .astype(str)
                .replace({"NaT": "", "nan": "", "None": ""})
            )
    for column in ("volume_lots",):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(3)
    for column in ("risk_cash", "entry_price", "stop_loss", "tp1", "tp2", "exit_price"):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
    table = table.rename(
        columns={
            "pnl_cash": "paper_pnl_cash",
            "equity_after": "paper_equity_after",
        }
    )
    return table.iloc[::-1].reset_index(drop=True)


def build_broker_trade_log_table(trade_log: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    if trade_log.empty:
        return trade_log

    columns = [
        "time_utc",
        "action",
        "direction",
        "confidence",
        "volume",
        "entry_price",
        "stop_loss",
        "take_profit",
        "spread_points",
        "retcode",
        "retcode_desc",
        "note",
    ]
    table = trade_log.loc[:, [column for column in columns if column in trade_log.columns]].tail(limit).copy()
    if "time_utc" in table.columns:
        table["time_utc"] = (
            table["time_utc"]
            .astype(str)
            .replace({"NaT": "", "nan": "", "None": ""})
        )
    for column in ("confidence", "volume", "entry_price", "stop_loss", "take_profit", "spread_points"):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
    return table.iloc[::-1].reset_index(drop=True)


# Trader-freundliche Session-Namen für die Buy/Sell-Ansicht
SESSION_DISPLAY_NAMES = {
    "Asia": "Tokyo / JPN",
    "London": "London",
    "New York": "New York",
    "Overlap": "London–NY Overlap",
    "Off Hours": "Off Hours",
}


def _rename_sessions(frame: pd.DataFrame, column: str = "session_name") -> pd.DataFrame:
    """Map internal session names to trader-friendly display names."""
    renamed = frame.copy()
    renamed[column] = renamed[column].map(lambda v: SESSION_DISPLAY_NAMES.get(v, v))
    return renamed


def build_session_buysell_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Buy vs Sell signal breakdown per trading session (London, New York, Tokyo/JPN).

    Uses setup_candidate (LONG_SETUP / SHORT_SETUP) as primary signal source,
    falls back to direction_15m if setup_candidate is unavailable.
    The raw directional signals show bias BEFORE the risk gate filters."""
    if "session_name" not in predictions.columns:
        return pd.DataFrame()

    frame = predictions.copy()

    # Primäre Signal-Quelle: setup_candidate (ungefiltert, vor Gating)
    if "setup_candidate" in frame.columns:
        frame["_buy"] = (frame["setup_candidate"] == "LONG_SETUP").astype(int)
        frame["_sell"] = (frame["setup_candidate"] == "SHORT_SETUP").astype(int)
    elif "direction_15m" in frame.columns:
        frame["_buy"] = (frame["direction_15m"] == 1).astype(int)
        frame["_sell"] = (frame["direction_15m"] == -1).astype(int)
    else:
        return pd.DataFrame()

    agg_dict: dict = {
        "total_bars": ("time", "count"),
        "buy_setups": ("_buy", "sum"),
        "sell_setups": ("_sell", "sum"),
    }
    if "buy_pressure_proxy" in frame.columns:
        agg_dict["avg_buy_pressure"] = ("buy_pressure_proxy", "mean")
    if "sell_pressure_proxy" in frame.columns:
        agg_dict["avg_sell_pressure"] = ("sell_pressure_proxy", "mean")

    grouped = frame.groupby("session_name", as_index=False).agg(**agg_dict)

    actionable = grouped["buy_setups"] + grouped["sell_setups"]
    safe_actionable = actionable.replace(0, 1)
    grouped["buy_pct"] = (grouped["buy_setups"] / safe_actionable * 100).round(1)
    grouped["sell_pct"] = (grouped["sell_setups"] / safe_actionable * 100).round(1)
    grouped["dominant"] = np.where(
        grouped["buy_setups"] > grouped["sell_setups"],
        "BUY ↑",
        np.where(grouped["sell_setups"] > grouped["buy_setups"], "SELL ↓", "NEUTRAL"),
    )

    rename_map = {
        "session_name": "Session",
        "total_bars": "Bars",
        "buy_setups": "Buy Setups",
        "sell_setups": "Sell Setups",
        "buy_pct": "Buy %",
        "sell_pct": "Sell %",
        "dominant": "Dominant",
    }
    if "avg_buy_pressure" in grouped.columns:
        rename_map["avg_buy_pressure"] = "Avg Buy Pressure"
    if "avg_sell_pressure" in grouped.columns:
        rename_map["avg_sell_pressure"] = "Avg Sell Pressure"

    return _rename_sessions(grouped.rename(columns=rename_map), column="Session")


def build_session_buysell_figure(predictions: pd.DataFrame) -> Any:
    """Stacked bar chart: Buy vs Sell setup count per session."""
    import plotly.graph_objects as go

    if "session_name" not in predictions.columns:
        figure = go.Figure()
        figure.update_layout(height=360, paper_bgcolor="#f4f0e8", plot_bgcolor="#fffaf0")
        return figure

    frame = predictions.copy()

    if "setup_candidate" in frame.columns:
        frame["_buy"] = (frame["setup_candidate"] == "LONG_SETUP").astype(int)
        frame["_sell"] = (frame["setup_candidate"] == "SHORT_SETUP").astype(int)
    elif "direction_15m" in frame.columns:
        frame["_buy"] = (frame["direction_15m"] == 1).astype(int)
        frame["_sell"] = (frame["direction_15m"] == -1).astype(int)
    else:
        figure = go.Figure()
        figure.update_layout(height=360, paper_bgcolor="#f4f0e8", plot_bgcolor="#fffaf0")
        return figure

    grouped = frame.groupby("session_name").agg(
        buy_setups=("_buy", "sum"),
        sell_setups=("_sell", "sum"),
    ).reset_index()

    # Sortierung: Overlap → London → New York → Asia → Off Hours
    order = ["Overlap", "London", "New York", "Asia", "Off Hours"]
    grouped["_sort"] = grouped["session_name"].map({name: idx for idx, name in enumerate(order)}).fillna(99)
    grouped = grouped.sort_values("_sort").drop(columns="_sort")

    grouped = _rename_sessions(grouped)
    sessions = grouped["session_name"].tolist()

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=sessions,
        y=grouped["buy_setups"],
        name="BUY (LONG Setups)",
        marker_color="#2a8c69",
        text=grouped["buy_setups"],
        textposition="inside",
    ))
    figure.add_trace(go.Bar(
        x=sessions,
        y=grouped["sell_setups"],
        name="SELL (SHORT Setups)",
        marker_color="#b9444b",
        text=grouped["sell_setups"],
        textposition="inside",
    ))
    figure.update_layout(
        barmode="stack",
        height=420,
        title="Buy vs Sell Setups per Session",
        paper_bgcolor="#f4f0e8",
        plot_bgcolor="#fffaf0",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
    )
    return figure
