"""
Helpers for loading research cockpit artifacts and building Streamlit visuals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from pipeline_contract import (
    DEFAULT_MT5_RESEARCH_FINAL_LEDGER_OUTPUT,
    DEFAULT_MT5_RESEARCH_TRADE_HISTORY_OUTPUT,
    DEFAULT_RESEARCH_OVERLAYS_OUTPUT,
    DEFAULT_RESEARCH_PAPER_LEDGER_OUTPUT,
    DEFAULT_RESEARCH_PREDICTIONS_OUTPUT,
    DEFAULT_RESEARCH_REPORT_OUTPUT,
    display_path,
    resolve_repo_path,
)
from mt5_client import fetch_recent_rates, load_exported_rates_csv
from oanda_client import (
    display_instrument,
    fetch_instrument_candles,
    normalize_oanda_instrument,
    resolve_api_token as resolve_oanda_token,
)
from twelvedata_client import fetch_time_series, resolve_api_key


def _load_json(path_like: str | Path) -> dict[str, Any]:
    with resolve_repo_path(path_like).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_research_bundle(report_path: str = DEFAULT_RESEARCH_REPORT_OUTPUT) -> dict[str, Any]:
    report = _load_json(report_path)
    outputs = report.get("outputs", {})
    predictions_path = outputs.get("predictions_output", DEFAULT_RESEARCH_PREDICTIONS_OUTPUT)
    overlays_path = outputs.get("overlays_output", DEFAULT_RESEARCH_OVERLAYS_OUTPUT)
    paper_path = outputs.get("paper_output", DEFAULT_RESEARCH_PAPER_LEDGER_OUTPUT)
    trade_history_path = outputs.get("trade_history_output", DEFAULT_MT5_RESEARCH_TRADE_HISTORY_OUTPUT)
    final_trade_ledger_path = outputs.get("final_trade_ledger_output", "")

    predictions = pd.read_csv(resolve_repo_path(predictions_path))
    if "time" in predictions.columns:
        predictions["time"] = pd.to_datetime(predictions["time"])
    paper_ledger_path = resolve_repo_path(paper_path)
    if paper_ledger_path.exists() and paper_ledger_path.stat().st_size > 0:
        try:
            paper_ledger = pd.read_csv(paper_ledger_path)
        except EmptyDataError:
            paper_ledger = pd.DataFrame()
        for column in ("signal_time", "entry_time", "exit_time"):
            if column in paper_ledger.columns:
                paper_ledger[column] = pd.to_datetime(paper_ledger[column], errors="coerce")
    else:
        paper_ledger = pd.DataFrame()

    trade_history_resolved = resolve_repo_path(trade_history_path)
    if trade_history_resolved.exists() and trade_history_resolved.stat().st_size > 0:
        try:
            trade_history = pd.read_csv(trade_history_resolved)
        except EmptyDataError:
            trade_history = pd.DataFrame()
        for column in ("signal_time", "entry_time", "exit_time", "snapshot_created_at"):
            if column in trade_history.columns:
                trade_history[column] = pd.to_datetime(trade_history[column], errors="coerce")
    else:
        trade_history = pd.DataFrame()

    final_trade_ledger_resolved = resolve_repo_path(final_trade_ledger_path or DEFAULT_MT5_RESEARCH_FINAL_LEDGER_OUTPUT)
    if final_trade_ledger_resolved.exists() and final_trade_ledger_resolved.stat().st_size > 0:
        try:
            final_trade_ledger = pd.read_csv(final_trade_ledger_resolved)
        except EmptyDataError:
            final_trade_ledger = pd.DataFrame()
        for column in ("signal_time", "entry_time", "exit_time", "snapshot_created_at", "source_end"):
            if column in final_trade_ledger.columns:
                final_trade_ledger[column] = pd.to_datetime(final_trade_ledger[column], errors="coerce")
    else:
        final_trade_ledger = trade_history.copy()

    overlays = _load_json(overlays_path)
    latest_signal = report.get("latest_signal", {})
    return {
        "report": report,
        "predictions": predictions,
        "paper_ledger": paper_ledger,
        "trade_history": trade_history,
        "final_trade_ledger": final_trade_ledger,
        "overlays": overlays,
        "latest_signal": latest_signal,
        "paths": {
            "report": display_path(report_path),
            "predictions": predictions_path,
            "overlays": overlays_path,
            "paper": paper_path,
            "trade_history": trade_history_path,
            "final_trade_ledger": final_trade_ledger_path or DEFAULT_MT5_RESEARCH_FINAL_LEDGER_OUTPUT,
        },
    }


def load_live_market_snapshot(
    provider: str = "auto",
    env_name: str = "TWELVEDATA_API_KEY",
    oanda_env_name: str = "OANDA_API_TOKEN",
    mt5_symbol: str = "XAUUSD",
    symbol: str = "XAU/USD",
    outputsize: int = 120,
) -> dict[str, Any]:
    provider_value = provider.strip().lower()
    payload: dict[str, Any] | None = None
    selected_provider = provider_value

    def try_mt5_payload() -> dict[str, Any] | None:
        try:
            return fetch_recent_rates(
                symbol=mt5_symbol or symbol or "XAUUSD",
                timeframe="M1",
                count=outputsize,
            )
        except Exception:
            try:
                return load_exported_rates_csv(
                    symbol=mt5_symbol or symbol or "XAUUSD",
                    timeframe="M1",
                )
            except Exception:
                return None

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
    elif provider_value == "auto":
        mt5_payload = try_mt5_payload()
        if mt5_payload is not None:
            selected_provider = "mt5"
            payload = mt5_payload
        elif resolve_oanda_token(oanda_env_name):
            selected_provider = "oanda"
        elif resolve_api_key(env_name):
            selected_provider = "twelvedata"
        else:
            selected_provider = "twelvedata"

    if selected_provider == "oanda":
        api_token = resolve_oanda_token(oanda_env_name)
        if not api_token:
            return {
                "enabled": False,
                "error": f"Environment variable {oanda_env_name} is not set.",
            }
        instrument = normalize_oanda_instrument(symbol or "XAU_USD")
        try:
            payload = fetch_instrument_candles(
                api_token=api_token,
                instrument=instrument,
                granularity="M1",
                count=outputsize,
            )
        except Exception as exc:
            return {
                "enabled": False,
                "error": str(exc),
            }
    elif selected_provider == "twelvedata":
        api_key = resolve_api_key(env_name)
        if not api_key:
            return {
                "enabled": False,
                "error": f"Environment variable {env_name} is not set.",
            }
        td_symbol = (symbol or "XAU/USD").strip().upper().replace("_", "/")
        try:
            payload = fetch_time_series(api_key=api_key, symbol=td_symbol, outputsize=outputsize)
        except Exception as exc:
            return {
                "enabled": False,
                "error": str(exc),
            }
    elif selected_provider != "mt5":
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
    return {
        "enabled": True,
        "provider": selected_provider,
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
        "display_symbol": (
            payload.get("meta", {}).get("symbol", mt5_symbol)
            if selected_provider == "mt5"
            else
            display_instrument(payload.get("meta", {}).get("instrument", symbol))
            if selected_provider == "oanda"
            else payload.get("meta", {}).get("symbol", symbol)
        ),
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

    curve = report.get("backtest", {}).get("equity_curve", [])
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
    frame["_summary_time"] = pd.to_datetime(frame[time_column], errors="coerce")
    frame = frame.loc[frame["_summary_time"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

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

    ordered = trade_history.sort_values(
        by=[column for column in ("snapshot_created_at", "exit_time", "signal_time") if column in trade_history.columns],
        ascending=True,
    )
    columns = [
        "snapshot_created_at",
        "signal_time",
        "entry_time",
        "exit_time",
        "direction",
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
            table[column] = table[column].astype(str)
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
