from __future__ import annotations

import sys
import textwrap
import time
from zoneinfo import ZoneInfo
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from pipeline_contract import (
    DEFAULT_MT5_RESEARCH_REPORT_OUTPUT,
    DEFAULT_RESEARCH_REPORT_OUTPUT,
    resolve_repo_path,
)
from research_streamlit_data import (
    build_blocked_signal_table,
    build_calibration_figure,
    build_candlestick_figure,
    build_equity_curve_figure,
    build_hour_bias_figure,
    build_learning_checks_table,
    build_model_metric_table,
    build_paper_ledger_table,
    build_probability_figure,
    build_recent_signal_table,
    build_saved_trade_history_table,
    build_session_buysell_figure,
    build_session_buysell_table,
    build_session_profile_figure,
    build_trade_period_summary_table,
    build_walk_forward_table,
    load_live_market_snapshot,
    load_mt5_timeframe_ribbon,
    load_research_bundle,
)


st.set_page_config(
    page_title="XAUUSD Research Cockpit",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
          --sand: #f4f0e8;
          --paper: #fffaf0;
          --ink: #18242f;
          --accent: #b86a2d;
          --forest: #2a8c69;
          --danger: #b9444b;
          --slate: #23395d;
          --gold: #d6b25e;
        }

        .stApp {
          background:
            radial-gradient(circle at top left, rgba(184,106,45,0.12), transparent 32%),
            radial-gradient(circle at bottom right, rgba(35,57,93,0.10), transparent 28%),
            linear-gradient(180deg, var(--sand), #efe6d5 70%, #e9ddc9 100%);
          color: var(--ink);
          font-family: "IBM Plex Sans", sans-serif;
        }

        h1, h2, h3 {
          font-family: "Fraunces", serif !important;
          letter-spacing: -0.02em;
          color: var(--ink);
        }

        /* Automatisches Responsive-Wrapping für die Streamlit-Spalten, 
           damit die Cards nie zerquetscht werden (verhindert das Kacke-Aussehen) */
        [data-testid="column"] {
          min-width: 260px !important;
        }

        section[data-testid="stSidebar"] {
          background: rgba(255, 250, 240, 0.84);
          border-right: 1px solid rgba(24, 36, 47, 0.08);
        }

        .cockpit-hero {
          background: linear-gradient(140deg, rgba(255,250,240,0.88), rgba(235,223,198,0.82));
          border: 1px solid rgba(24,36,47,0.08);
          border-radius: 24px;
          padding: 1.4rem 1.6rem;
          margin-bottom: 1rem;
          box-shadow: 0 14px 40px rgba(24, 36, 47, 0.08);
        }

        .cockpit-card {
          background: rgba(255, 250, 240, 0.92);
          border: 1px solid rgba(24,36,47,0.08);
          border-radius: 18px;
          padding: 1rem 1.1rem;
          box-shadow: 0 10px 28px rgba(24, 36, 47, 0.05);
        }

        .metric-label {
          font-size: 0.82rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: rgba(24,36,47,0.64);
        }

        .metric-value {
          font-size: 1.6rem;
          font-weight: 700;
          color: var(--ink);
        }

        .metric-note {
          font-size: 0.90rem;
          color: rgba(24,36,47,0.72);
        }

        .ribbon-stack {
          display: flex;
          flex-direction: column;
          gap: 0.16rem;
        }

        .ribbon-box {
          border-radius: 8px;
          padding: 0.18rem 0.30rem;
          border: 1px solid rgba(24,36,47,0.10);
          box-shadow: 0 4px 10px rgba(24, 36, 47, 0.03);
          color: #fffaf0;
        }

        .ribbon-inline {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.28rem;
        }

        .ribbon-up {
          background: linear-gradient(135deg, rgba(42,140,105,0.96), rgba(24,112,84,0.96));
        }

        .ribbon-down {
          background: linear-gradient(135deg, rgba(185,68,75,0.96), rgba(138,42,49,0.96));
        }

        .ribbon-flat {
          background: linear-gradient(135deg, rgba(180,155,87,0.96), rgba(149,123,58,0.96));
        }

        .ribbon-time {
          font-size: 0.52rem;
          opacity: 0.88;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          line-height: 1.1;
        }

        .ribbon-direction {
          font-size: 0.63rem;
          font-weight: 700;
          letter-spacing: 0.05em;
          line-height: 1.1;
        }

        .ribbon-move {
          font-size: 0.52rem;
          opacity: 0.82;
          line-height: 1.1;
        }

        .popup-ribbon-container {
          position: fixed;
          bottom: 24px;
          right: 32px;
          z-index: 99999;
          background: rgba(24, 36, 47, 0.95);
          border: 1px solid rgba(255, 250, 240, 0.1);
          border-radius: 12px;
          padding: 0.8rem;
          box-shadow: 0 16px 40px rgba(0, 0, 0, 0.25);
          width: 140px;
        }

        .popup-ribbon-title {
          font-size: 0.65rem;
          text-transform: uppercase;
          color: #fffaf0;
          opacity: 0.9;
          margin-bottom: 0.6rem;
          text-align: center;
          letter-spacing: 0.05em;
          font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="cockpit-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_count(value: object) -> str:
    return f"{_coerce_int(value):,}"


def build_trade_learning_card(report: dict) -> tuple[str, str]:
    learning = report.get("learning", {})
    closed_trades = _coerce_int(learning.get("closed_trades", 0))
    actionable_signals = _coerce_int(learning.get("actionable_signals", 0))

    required_closed_trades = 0
    for check in learning.get("checks", []):
        if str(check.get("check", "")) == "closed_trades":
            required_closed_trades = _coerce_int(check.get("required", 0))
            break

    if required_closed_trades > 0:
        progress_note = f"{closed_trades:,}/{required_closed_trades:,} needed for batch retrain"
    else:
        progress_note = f"{actionable_signals:,} actionable signals tracked"

    readiness_note = "retrain ready" if learning.get("retrain_ready") else "collecting more trade feedback"
    return format_count(closed_trades), f"{progress_note} | {readiness_note}"


def build_neural_connections_card(report: dict) -> tuple[str, str]:
    learning = report.get("learning", {})
    neural = report.get("neural", {})
    closed_trades = _coerce_int(learning.get("closed_trades", 0))
    trainable_connections = _coerce_int(neural.get("trainable_parameter_count", neural.get("parameter_count", 0)))
    neural_status = str(neural.get("status", "unknown")).strip() or "unknown"

    if trainable_connections > 0:
        return format_count(trainable_connections), f"{closed_trades:,} closed trade samples available for retraining"
    if neural_status == "disabled":
        return "baseline only", f"{closed_trades:,} trade samples tracked while the neural layer is off"
    if neural_status == "ok":
        return "trained", f"{closed_trades:,} trade samples tracked | saved report has no connection count"
    return neural_status.upper(), f"{closed_trades:,} trade samples tracked"


def build_learning_explainer(report: dict) -> str:
    neural = report.get("neural", {})
    trainable_connections = _coerce_int(neural.get("trainable_parameter_count", neural.get("parameter_count", 0)))
    if trainable_connections > 0:
        return (
            "Trade Learning counts closed LONG/SHORT paper trades eligible for retraining. "
            "Neural Connections shows the active PatchTST trainable weights."
        )
    return (
        "Trade Learning counts closed LONG/SHORT paper trades eligible for retraining. "
        "This MT5 live path is still on the fast baseline, so it does not expose per-weight neural updates yet."
    )


def format_utc_and_berlin(timestamp_like: str) -> str:
    timestamp = str(timestamp_like or "").strip()
    if not timestamp:
        return ""
    parsed = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(parsed):
        return timestamp
    parsed_utc = parsed.tz_localize(UTC) if parsed.tzinfo is None else parsed.tz_convert(UTC)
    parsed_berlin = parsed_utc.tz_convert(BERLIN_TZ)
    return (
        f"{parsed_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} | "
        f"{parsed_berlin.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )


def build_overlap_window_note(report: dict) -> str:
    mt5_live_source = report.get("mt5_live_source", {})
    hunt_windows = list(mt5_live_source.get("hunt_windows", []))
    hunt_timezone = str(mt5_live_source.get("hunt_timezone", "UTC"))
    if hunt_windows:
        segments = []
        for window in hunt_windows:
            name = str(window.get("name", "Hunt")).strip() or "Hunt"
            start = str(window.get("start", "")).strip()
            end = str(window.get("end", "")).strip()
            max_trades = int(window.get("max_trades", 0) or 0)
            limit_note = f" max {max_trades}" if max_trades > 0 else ""
            segments.append(f"{name} {start}-{end} {hunt_timezone}{limit_note}")
        return "Setup hunt windows: " + " | ".join(segments)

    anchor_raw = str(mt5_live_source.get("end", "") or "")
    anchor = pd.to_datetime(anchor_raw, errors="coerce")
    if pd.isna(anchor):
        anchor = pd.Timestamp.now(tz=UTC)
    elif anchor.tzinfo is None:
        anchor = anchor.tz_localize(UTC)
    else:
        anchor = anchor.tz_convert(UTC)

    window_start_utc = anchor.normalize() + pd.Timedelta(hours=13)
    window_end_utc = anchor.normalize() + pd.Timedelta(hours=16, minutes=59)
    window_start_berlin = window_start_utc.tz_convert(BERLIN_TZ)
    window_end_berlin = window_end_utc.tz_convert(BERLIN_TZ)
    return (
        "Setup hunt window: "
        f"{window_start_utc.strftime('%H:%M')} - {window_end_utc.strftime('%H:%M')} UTC = "
        f"{window_start_berlin.strftime('%H:%M')} - {window_end_berlin.strftime('%H:%M %Z')}"
    )


def build_execution_policy_note(report: dict) -> str:
    policy = report.get("execution_policy", {})
    if not policy:
        return ""
    broker_mode = str(policy.get("broker_execution_mode", "unknown")).replace("_", " ")
    setup_source = str(policy.get("trade_setup_source", "unknown")).replace("_", " ")
    learning_source = str(policy.get("learning_source", "unknown")).replace("_", " ")
    streamlit_scope = str(policy.get("streamlit_scope", "unknown")).replace("_", " ")
    requires_directive = bool(policy.get("requires_trade_directive", False))
    sync_error = str(policy.get("trade_directive_sync_error", "")).strip()
    synced_targets = policy.get("trade_directive_synced_targets", [])
    if requires_directive:
        if sync_error:
            directive_sync = "error"
        elif synced_targets:
            directive_sync = "ready"
        else:
            directive_sync = "pending"
    else:
        directive_sync = "not required"
    return (
        f"Broker execution: {broker_mode} | "
        f"trade setup source: {setup_source} | "
        f"learning source: {learning_source} | "
        f"Streamlit scope: {streamlit_scope} | "
        f"trade directive sync: {directive_sync}"
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_bundle(report_path: str, refresh_key: int = 0) -> dict:
    return load_research_bundle(report_path)


@st.cache_data(ttl=300, show_spinner=False)
def cached_live_snapshot(provider: str, symbol: str, mt5_symbol: str, outputsize: int, enabled: bool, refresh_key: int = 0) -> dict:
    if not enabled:
        return {"enabled": False, "error": "Live snapshot disabled in the sidebar."}
    return load_live_market_snapshot(provider=provider, symbol=symbol, mt5_symbol=mt5_symbol, outputsize=outputsize)


@st.cache_data(ttl=300, show_spinner=False)
def cached_mt5_timeframe_ribbon(mt5_symbol: str, enabled: bool, refresh_key: int = 0) -> dict:
    if not enabled:
        return {"enabled": False, "error": "Ribbon disabled because live market snapshot is turned off."}
    return load_mt5_timeframe_ribbon(mt5_symbol=mt5_symbol)


def build_refresh_key(settings: dict[str, object]) -> int:
    if not bool(settings.get("auto_refresh_enabled", False)):
        return 0
    interval = max(5, int(settings.get("auto_refresh_seconds", 15)))
    return int(time.time() // interval)


def get_auto_refresh_run_every(settings: dict[str, object]) -> str | None:
    if not bool(settings.get("auto_refresh_enabled", False)):
        return None
    interval = max(5, int(settings.get("auto_refresh_seconds", 15)))
    return f"{interval}s"


def get_ui_settings() -> dict[str, object]:
    settings = st.session_state.get("_ui_settings")
    if isinstance(settings, dict):
        return settings
    settings = sidebar_state()
    st.session_state["_ui_settings"] = settings
    return settings


def resolve_active_report_path(report_source: str, live_provider: str, custom_path: str) -> str:
    if report_source == "Custom":
        return custom_path.strip() or DEFAULT_RESEARCH_REPORT_OUTPUT
    if report_source == "Research Baseline":
        return DEFAULT_RESEARCH_REPORT_OUTPUT
    if report_source == "MT5 Live Paper":
        return DEFAULT_MT5_RESEARCH_REPORT_OUTPUT

    mt5_report = resolve_repo_path(DEFAULT_MT5_RESEARCH_REPORT_OUTPUT)
    if live_provider in {"Auto", "MT5 Local", "MT5 Exporter"} and mt5_report.exists():
        return DEFAULT_MT5_RESEARCH_REPORT_OUTPUT
    return DEFAULT_RESEARCH_REPORT_OUTPUT


def sidebar_state() -> dict[str, object]:
    st.sidebar.markdown("## Research Control")
    live_provider = st.sidebar.selectbox("Live provider", options=["Auto", "MT5 Local", "MT5 Exporter", "OANDA", "Twelve Data"], index=0)
    report_source = st.sidebar.selectbox(
        "Report source",
        options=["Auto", "MT5 Live Paper", "Research Baseline", "Custom"],
        index=0,
        help="Auto prefers the MT5 live paper-trading report whenever MT5 Local is active and a local MT5 report exists.",
    )
    custom_default = DEFAULT_MT5_RESEARCH_REPORT_OUTPUT if live_provider in {"Auto", "MT5 Local", "MT5 Exporter"} else DEFAULT_RESEARCH_REPORT_OUTPUT
    custom_report_path = (
        st.sidebar.text_input("Custom report path", value=custom_default)
        if report_source == "Custom"
        else custom_default
    )
    report_path = resolve_active_report_path(report_source, live_provider, custom_report_path)
    st.sidebar.caption(f"Active report: `{report_path}`")
    chart_bars = st.sidebar.slider("Chart bars", min_value=120, max_value=720, value=240, step=20)
    probability_bars = st.sidebar.slider("Probability bars", min_value=60, max_value=600, value=180, step=20)
    live_enabled = st.sidebar.toggle("Load live market snapshot", value=True)
    if live_provider in {"MT5 Local", "MT5 Exporter"}:
        default_symbol = "XAUUSD"
    elif live_provider == "OANDA":
        default_symbol = "XAU_USD"
    else:
        default_symbol = "XAU/USD"
    live_symbol = st.sidebar.text_input("Live symbol / instrument", value=default_symbol, key=f"live_symbol_{live_provider.lower().replace(' ', '_')}")
    live_window = st.sidebar.slider("Live snapshot bars", min_value=20, max_value=240, value=120, step=20)
    auto_refresh_enabled = st.sidebar.toggle("Auto refresh live UI", value=True)
    auto_refresh_seconds = st.sidebar.slider(
        "Refresh every (sec)",
        min_value=5,
        max_value=120,
        value=5,
        step=5,
        disabled=not auto_refresh_enabled,
    )
    if st.sidebar.button("Refresh cached data"):
        st.cache_data.clear()

    st.sidebar.markdown("## Run Pipeline")
    live_command = (
        "python src/install_mt5_exporter.py\n"
        "python src/mt5_keychain_cli.py run-research-pipeline -- --source-mode auto\n"
        "python src/run_mt5_research_worker.py --poll-seconds 15"
        if live_provider in {"MT5 Local", "MT5 Exporter"}
        else
        "python src/run_live_oanda_pipeline.py"
        if live_provider == "OANDA"
        else "python src/run_live_twelvedata_pipeline.py"
    )
    st.sidebar.code(
        live_command
        if live_provider != "Auto"
        else "python src/mt5_keychain_cli.py run-research-pipeline -- --source-mode auto\npython src/run_live_oanda_pipeline.py\npython src/run_live_twelvedata_pipeline.py",
        language="bash",
    )
    st.sidebar.caption(
        "Install research extras first: pip install -r requirements-research.txt. "
        "Ohne Torch bleibt das Cockpit auf der LightGBM-Baseline. "
        "Auto tries MT5 Local first when it is available on the same machine as a running MetaTrader terminal. "
        "On macOS the MT5 exporter CSV fallback is often the most reliable path."
    )

    return {
        "report_path": report_path,
        "report_source": report_source,
        "chart_bars": chart_bars,
        "probability_bars": probability_bars,
        "live_provider": live_provider,
        "live_enabled": live_enabled,
        "live_symbol": live_symbol,
        "mt5_symbol": live_symbol if live_provider in {"MT5 Local", "MT5 Exporter"} else "XAUUSD",
        "live_window": live_window,
        "auto_refresh_enabled": auto_refresh_enabled,
        "auto_refresh_seconds": auto_refresh_seconds,
    }


def load_page_state(settings: dict[str, object]) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict, dict]:
    refresh_key = build_refresh_key(settings)
    try:
        bundle = cached_bundle(str(settings["report_path"]), refresh_key=refresh_key)
    except FileNotFoundError:
        st.error("Research report not found yet. Run the research pipeline first.")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load research artifacts: {exc}")
        st.stop()

    predictions = bundle["predictions"]
    paper_ledger = bundle["paper_ledger"]
    trade_history = bundle.get("trade_history", pd.DataFrame())
    final_trade_ledger = bundle.get("final_trade_ledger", pd.DataFrame())
    report = bundle["report"]
    overlays = bundle["overlays"]
    live_snapshot = cached_live_snapshot(
        str(settings["live_provider"]),
        str(settings["live_symbol"]),
        str(settings["mt5_symbol"]),
        int(settings["live_window"]),
        bool(settings["live_enabled"]),
        refresh_key=refresh_key,
    )
    mt5_ribbon = cached_mt5_timeframe_ribbon(str(settings["mt5_symbol"]), bool(settings["live_enabled"]), refresh_key=refresh_key)
    return settings, predictions, paper_ledger, report, overlays | {"_live_snapshot": live_snapshot, "_mt5_m30_ribbon": mt5_ribbon, "_trade_history": trade_history, "_final_trade_ledger": final_trade_ledger}


def render_m30_ribbon_card(ribbon: dict) -> None:
    if not ribbon.get("enabled"):
        return

    symbol = ribbon.get('symbol', 'XAUUSD')
    segments = []
    for bar in ribbon.get("bars", []):
        direction = str(bar.get("direction", "FLAT")).upper()
        css_class = "ribbon-up" if direction == "UP" else "ribbon-down" if direction == "DOWN" else "ribbon-flat"
        segments.append(
            f'<div class="ribbon-box {css_class}">'
            f'<div class="ribbon-inline">'
            f'<div class="ribbon-time">{bar.get("timeframe", "")}</div>'
            f'<div class="ribbon-direction">{direction}</div>'
            f'</div>'
            f"</div>"
        )
    ribbon_html = textwrap.dedent(
        f"""
        <div class="popup-ribbon-container">
          <div class="popup-ribbon-title">{symbol} MT5</div>
          <div class="ribbon-stack">{''.join(segments)}</div>
        </div>
        """
    ).strip()
    st.markdown(ribbon_html, unsafe_allow_html=True)


def render_header(report: dict) -> None:
    dataset_summary = report.get("dataset_summary", {})
    notes = report.get("notes", [])
    mt5_live_source = report.get("mt5_live_source", {})
    st.markdown(
        """
        <div class="cockpit-hero">
          <h1 style="margin:0;">XAUUSD Research Cockpit</h1>
          <p style="margin:0.4rem 0 0 0; max-width: 80ch;">
            Evaluation-first cockpit for calibrated probabilities, session behavior, structure overlays,
            and walk-forward robustness. This view is deliberately built around evidence, not vanity accuracy.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if notes:
        st.caption(" | ".join(notes))
    if mt5_live_source:
        st.caption(
            "MT5 live source: "
            f"{mt5_live_source.get('symbol', 'XAUUSD')} "
            f"| provider {mt5_live_source.get('provider', 'mt5_export')} "
            f"| rows {int(mt5_live_source.get('rows', 0)):,} "
            f"| {mt5_live_source.get('start', 'n/a')} -> {mt5_live_source.get('end', 'n/a')}"
        )
        st.caption(build_overlap_window_note(report))
    execution_policy_note = build_execution_policy_note(report)
    if execution_policy_note:
        st.caption(execution_policy_note)
    st.caption(
        f"Rows: standardized {dataset_summary.get('standardized_rows', 0):,} | "
        f"research {dataset_summary.get('research_rows', 0):,} | "
        f"labels {dataset_summary.get('label_rows', 0):,} | "
        f"predictions {dataset_summary.get('prediction_rows', 0):,} | "
        f"paper trades {dataset_summary.get('paper_trade_rows', 0):,}"
    )


def render_dashboard_page(settings: dict[str, object]) -> None:
    _, predictions, paper_ledger, report, overlay_bundle = load_page_state(settings)
    live_snapshot = overlay_bundle["_live_snapshot"]
    mt5_ribbon = overlay_bundle["_mt5_m30_ribbon"]
    final_trade_ledger = overlay_bundle.get("_final_trade_ledger", pd.DataFrame())
    latest = report.get("latest_signal", {})
    learning = report.get("learning", {})
    manual_override = report.get("manual_override", {})
    trade_learning_value, trade_learning_note = build_trade_learning_card(report)
    render_header(report)

    top_cols = st.columns(6)
    with top_cols[0]:
        render_metric_card("Recommended", str(latest.get("signal", "N/A")), format_utc_and_berlin(str(latest.get("time", ""))))
    with top_cols[1]:
        render_metric_card("Gate Status", str(latest.get("gate_status", "N/A")), latest.get("session_name", ""))
    with top_cols[2]:
        render_metric_card("Setup Score", f"{float(latest.get('setup_score', 0.0)):.1%}", latest.get("regime", ""))
    with top_cols[3]:
        render_metric_card("Expected Value", f"{float(latest.get('expected_value', 0.0)):.4f}", latest.get("reason_blocked", "tradeable"))
    with top_cols[4]:
        render_metric_card("Entry / SL", f"{float(latest.get('entry_price', 0.0)):.2f}", f"SL {float(latest.get('stop_loss', 0.0)):.2f}")
    with top_cols[5]:
        render_metric_card("Paper Status", str(latest.get("paper_status", "N/A")), latest.get("paper_reason_blocked", ""))

    risk_cols = st.columns(5)
    paper_summary = report.get("paper_trading", report.get("current_run_paper_trading", {}))
    current_run_summary = report.get("current_run_paper_trading", {})
    with risk_cols[0]:
        render_metric_card("Paper Trades", str(paper_summary.get("trade_count", 0)), "Closed trades")
    with risk_cols[1]:
        render_metric_card("Profit Factor", f"{float(paper_summary.get('profit_factor', 0.0)):.2f}", "Paper ledger")
    with risk_cols[2]:
        render_metric_card("Max Drawdown", f"{float(paper_summary.get('max_drawdown', 0.0)):.1%}", "Risk ceiling")
    with risk_cols[3]:
        render_metric_card("Trade Learning", trade_learning_value, trade_learning_note)
    with risk_cols[4]:
        blockers = learning.get("retrain_blockers", [])
        render_metric_card("Learning Gate", str(len(blockers)), "all checks passed" if not blockers else "|".join(blockers))

    capital_cols = st.columns(5)
    starting_equity = float(paper_summary.get("starting_equity", 0.0))
    ending_equity = float(paper_summary.get("ending_equity", starting_equity))
    net_pnl_cash = float(paper_summary.get("net_pnl_cash", ending_equity - starting_equity))
    risk_per_trade = float(paper_summary.get("risk_per_trade", 0.0))
    risk_cash = ending_equity * risk_per_trade
    sizing_mode = str(paper_summary.get("position_sizing_mode", "") or "equity_risk")
    with capital_cols[0]:
        render_metric_card("Capital", f"{ending_equity:.2f}", f"start {starting_equity:.2f}")
    with capital_cols[1]:
        render_metric_card("Risk / Trade", f"{risk_cash:.2f}", f"{risk_per_trade:.2%} of current capital | {sizing_mode}")
    with capital_cols[2]:
        render_metric_card("Net PnL", f"{net_pnl_cash:+.2f}", "paper equity delta")
    with capital_cols[3]:
        render_metric_card("Wins", str(int(paper_summary.get("win_count", 0))), "closed winners")
    with capital_cols[4]:
        render_metric_card("Losses", str(int(paper_summary.get("loss_count", 0))), "closed losers")

    frozen_summary = report.get("frozen_paper_trading", {})
    current_run_trade_count = int(current_run_summary.get("trade_count", 0))
    if frozen_summary:
        st.caption(
            "Paper cards above now use the saved closed-trade history across runs. "
            f"The latest reconstructed window currently contributes {current_run_trade_count} closed trade(s)."
        )
        frozen_cols = st.columns(3)
        with frozen_cols[0]:
            render_metric_card("Frozen Trades", str(int(frozen_summary.get("trade_count", 0))), "saved closed trades")
        with frozen_cols[1]:
            render_metric_card("Frozen Capital", f"{float(frozen_summary.get('ending_equity', frozen_summary.get('starting_equity', 0.0))):.2f}", f"start {float(frozen_summary.get('starting_equity', 0.0)):.2f}")
        with frozen_cols[2]:
            render_metric_card("Frozen Net PnL", f"{float(frozen_summary.get('net_pnl_cash', 0.0)):+.2f}", "saved closed-trade history")

    saved_summary = build_trade_period_summary_table(final_trade_ledger)
    if not saved_summary.empty:
        st.markdown("#### Frozen Final Trade History")
        st.caption("These cards use the frozen final ledger, so each closed trade is counted once instead of replaying every reconstructed MT5 run.")
        history_cols = st.columns(len(saved_summary))
        for idx, row in saved_summary.reset_index(drop=True).iterrows():
            note = (
                f"{int(row['Trades'])} trades | "
                f"W {int(row['Wins'])} / L {int(row['Losses'])} | "
                f"PnL {float(row['Net PnL']):+.2f}"
            )
            with history_cols[idx]:
                render_metric_card(str(row["Period"]), f"{float(row['Win Rate']):.1%}", note)

    override_cols = st.columns([1.0, 1.2, 1.0])
    with override_cols[0]:
        render_metric_card("Override Credits", str(int(manual_override.get("remaining_credits", 0))), "paper-only extra trades")
    with override_cols[1]:
        override_start = str(manual_override.get("start_after", "") or "")
        override_note = "future signals only"
        if override_start:
            override_note = f"after last processed signal | {format_utc_and_berlin(override_start)}"
        render_metric_card("Override Start", override_start or "immediate", override_note)
    with override_cols[2]:
        render_metric_card("Override Used", str(len(manual_override.get("used_signal_times", []))), "|".join(manual_override.get("allowed_risk_blockers", [])) or "none")

    live_col, signal_col = st.columns([1.05, 1.50])
    with live_col:
        st.subheader("Live Snapshot")
        if live_snapshot.get("enabled"):
            latest_live = live_snapshot["latest"]
            delta_text = f"{live_snapshot['change']:+.2f} ({live_snapshot['change_pct']:+.2f}%)"
            render_metric_card(
                str(live_snapshot.get("display_symbol") or live_snapshot["meta"].get("symbol", "XAU/USD")),
                f"{float(latest_live['close']):.2f}",
                f"{latest_live['datetime']} | {delta_text}",
            )
            provider_name = str(live_snapshot.get("provider", "unknown")).replace("_", " ").title()
            st.caption(f"Provider: {provider_name} | {live_snapshot.get('volume_note', '')}")
            freshness_note = str(live_snapshot.get("freshness_note", "")).strip()
            if freshness_note:
                if bool(live_snapshot.get("stale", False)):
                    st.warning(freshness_note)
                else:
                    st.caption(freshness_note)
        else:
            st.warning(live_snapshot.get("error", "Live market snapshot unavailable."))

    with signal_col:
        st.subheader("Trade Decision")
        decision_frame = pd.DataFrame(
            [
                {"Layer": "5m Timing", "Signal": {1: "LONG", 0: "HOLD", -1: "SHORT"}.get(int(latest.get("signal_5m", 0)), "HOLD") if latest.get("signal_5m") is not None else "N/A"},
                {"Layer": "15m Bias", "Signal": {1: "LONG", 0: "HOLD", -1: "SHORT"}.get(int(latest.get("signal_15m", 0)), "HOLD") if latest.get("signal_15m") is not None else "N/A"},
                {"Layer": "60m Confirmation", "Signal": {1: "LONG", 0: "HOLD", -1: "SHORT"}.get(int(latest.get("signal_60m", 0)), "HOLD") if latest.get("signal_60m") is not None else "N/A"},
            ]
        )
        st.dataframe(decision_frame, width="stretch", hide_index=True)
        st.caption(f"Blocked reasons: {latest.get('reason_blocked', 'none') or 'none'}")
        st.caption(f"Confluence tags: {latest.get('confluence_tags', 'N/A')}")

    st.subheader("Probability Curves")
    st.plotly_chart(
        build_probability_figure(predictions, bars=int(settings["probability_bars"])),
        width="stretch",
    )

    lower_cols = st.columns([1.15, 0.85])
    with lower_cols[0]:
        st.subheader("Recent Signals")
        st.dataframe(build_recent_signal_table(predictions, limit=20), width="stretch", hide_index=True)
    with lower_cols[1]:
        st.subheader("Recent Blocked Setups")
        blocked_table = build_blocked_signal_table(predictions, limit=12)
        if blocked_table.empty:
            st.info("No blocked setups in the current window.")
        else:
            st.dataframe(blocked_table, width="stretch", hide_index=True)

    st.subheader("Latest Paper Trades")
    paper_table = build_paper_ledger_table(paper_ledger, limit=12)
    if paper_table.empty:
        st.info("No paper trades are closed yet in the current report.")
    else:
        st.dataframe(paper_table, width="stretch", hide_index=True)


def render_chart_structure_page(settings: dict[str, object]) -> None:
    _, predictions, _, report, overlay_bundle = load_page_state(settings)
    render_header(report)
    st.subheader("Market Structure And Execution Geometry")
    st.plotly_chart(
        build_candlestick_figure(predictions, overlay_bundle, bars=int(settings["chart_bars"])),
        width="stretch",
    )

    latest = report.get("latest_signal", {})
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown("#### Latest Trade Geometry")
        geometry = pd.DataFrame(
            [
                ["Entry", latest.get("entry_price")],
                ["Stop Loss", latest.get("stop_loss")],
                ["TP1", latest.get("tp1")],
                ["TP2", latest.get("tp2")],
            ],
            columns=["Level", "Price"],
        )
        st.dataframe(geometry, width="stretch", hide_index=True)
    with col2:
        st.markdown("#### Overlay Counts")
        st.dataframe(
            pd.DataFrame(
                [{"Overlay": key, "Count": value} for key, value in report.get("overlay_counts", {}).items()]
            ),
            width="stretch",
            hide_index=True,
        )


def render_sessions_psychology_page(settings: dict[str, object]) -> None:
    _, predictions, paper_ledger, report, overlay_bundle = load_page_state(settings)
    final_trade_ledger = overlay_bundle.get("_final_trade_ledger", pd.DataFrame())
    render_header(report)
    st.subheader("Session Behavior, Flow Proxies, And Timing Psychology")
    upper = st.columns(2)
    with upper[0]:
        st.plotly_chart(build_session_profile_figure(report), width="stretch")
    with upper[1]:
        st.plotly_chart(build_hour_bias_figure(predictions), width="stretch")

    session_summary = predictions.groupby("session_name", as_index=False).agg(
        bars=("time", "count"),
        avg_buy_pressure=("buy_pressure_proxy", "mean"),
        avg_sell_pressure=("sell_pressure_proxy", "mean"),
        avg_expected_value=("expected_value", "mean"),
        avg_setup_score=("setup_score", "mean"),
        entry_rate=("entry_zone_present", "mean"),
        signal_rate=("recommended_trade", lambda values: float(pd.Series(values).isin(["LONG", "SHORT"]).mean())),
    )
    st.markdown("#### Session Summary Table")
    st.dataframe(session_summary, width="stretch", hide_index=True)
    if not paper_ledger.empty:
        session_scoreboard = paper_ledger.groupby("session_name", as_index=False).agg(
            trades=("trade_id", "count"),
            avg_r=("realized_r", "mean"),
            win_rate=("realized_r", lambda values: float((pd.Series(values) > 0).mean())),
            pnl_cash=("pnl_cash", "sum"),
        )
        st.markdown("#### Current Report Session Scoreboard")
        st.dataframe(session_scoreboard, width="stretch", hide_index=True)

    if not final_trade_ledger.empty:
        saved_session_scoreboard = final_trade_ledger.groupby("session_name", as_index=False).agg(
            trades=("final_trade_key", "count") if "final_trade_key" in final_trade_ledger.columns else ("signal_time", "count"),
            avg_r=("realized_r", "mean"),
            win_rate=("realized_r", lambda values: float((pd.Series(values) > 0).mean())),
            pnl_cash=("pnl_cash", "sum"),
        )
        st.markdown("#### Frozen Final Session Scoreboard")
        st.caption("This scoreboard uses the frozen final ledger so each session trade is counted only once.")
        st.dataframe(saved_session_scoreboard, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Session Buy / Sell Breakdown")
    buysell_chart_col, buysell_table_col = st.columns([1.0, 1.2])
    with buysell_chart_col:
        st.plotly_chart(build_session_buysell_figure(predictions), width="stretch")
    with buysell_table_col:
        buysell_table = build_session_buysell_table(predictions)
        if buysell_table.empty:
            st.info("No session buy/sell data available.")
        else:
            st.dataframe(buysell_table, width="stretch", hide_index=True)


def render_model_lab_page(settings: dict[str, object]) -> None:
    _, predictions, _, report, _ = load_page_state(settings)
    render_header(report)
    st.subheader("Model Lab")

    metric_cols = st.columns(3)
    baseline = report.get("baseline", {})
    neural = report.get("neural", {})
    ensemble = report.get("ensemble", {})
    with metric_cols[0]:
        render_metric_card("Walk-Forward Splits", str(baseline.get("walk_forward_splits", "N/A")), "Chronological only")
    with metric_cols[1]:
        render_metric_card("Neural Status", str(neural.get("status", "unknown")), "PatchTST layer")
    with metric_cols[2]:
        render_metric_card("Ensemble Status", str(ensemble.get("status", "unknown")), "Probability blend")

    learning = report.get("learning", {})
    trade_learning_value, trade_learning_note = build_trade_learning_card(report)
    neural_connections_value, neural_connections_note = build_neural_connections_card(report)
    learning_cols = st.columns(5)
    with learning_cols[0]:
        render_metric_card("Batch Retrain", "READY" if learning.get("retrain_ready") else "NOT READY", learning.get("recommended_action", "collect_more_feedback"))
    with learning_cols[1]:
        render_metric_card("Trade Learning", trade_learning_value, trade_learning_note)
    with learning_cols[2]:
        render_metric_card("Direction Mix", f"L {int(learning.get('long_trades', 0))} / S {int(learning.get('short_trades', 0))}", learning.get("dominant_direction", ""))
    with learning_cols[3]:
        render_metric_card("Session Concentration", f"{float(learning.get('dominant_session_share', 0.0)):.1%}", learning.get("dominant_session", ""))
    with learning_cols[4]:
        render_metric_card("Neural Connections", neural_connections_value, neural_connections_note)

    st.caption(build_learning_explainer(report))

    st.markdown("#### Baseline vs Neural Metrics")
    st.dataframe(build_model_metric_table(report), width="stretch", hide_index=True)

    st.markdown("#### Safe Learning Checks")
    learning_checks = build_learning_checks_table(report)
    if learning_checks.empty:
        st.info("No learning safety checks are available in the current report.")
    else:
        st.dataframe(learning_checks, width="stretch", hide_index=True)

    horizon = st.segmented_control("Calibration horizon", options=[5, 15, 60], default=15)
    st.plotly_chart(build_calibration_figure(report, horizon=int(horizon)), width="stretch")

    st.markdown("#### Latest 15m Probability Slice")
    latest_probs = predictions[
        [
            "time",
            "prob_short_15m",
            "prob_hold_15m",
            "prob_long_15m",
            "recommended_trade",
            "reason_blocked",
        ]
    ].tail(15).copy()
    latest_probs["time"] = latest_probs["time"].astype(str)
    st.dataframe(latest_probs.iloc[::-1], width="stretch", hide_index=True)

    if neural.get("notes"):
        st.caption("Neural notes: " + " | ".join(str(note) for note in neural["notes"]))


def render_backtest_walkforward_page(settings: dict[str, object]) -> None:
    _, predictions, paper_ledger, report, overlay_bundle = load_page_state(settings)
    final_trade_ledger = overlay_bundle.get("_final_trade_ledger", pd.DataFrame())
    render_header(report)
    st.subheader("Paper Trading And Walk-Forward Evidence")

    backtest = report.get("paper_trading", report.get("current_run_paper_trading", report.get("backtest", {})))
    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card("Trades", str(backtest.get("trade_count", 0)), "Closed paper trades")
    with metric_cols[1]:
        render_metric_card("Win Rate", f"{float(backtest.get('win_rate', 0.0)):.1%}", "Paper ledger")
    with metric_cols[2]:
        render_metric_card("Profit Factor", f"{float(backtest.get('profit_factor', 0.0)):.3f}", "Paper ledger")
    with metric_cols[3]:
        render_metric_card("Expectancy R", f"{float(backtest.get('expectancy_r', 0.0)):.3f}", "Per paper trade")
    with metric_cols[4]:
        render_metric_card("Max Drawdown", f"{float(backtest.get('max_drawdown', 0.0)):.1%}", "Equity curve")

    frozen_summary = report.get("frozen_paper_trading", {})
    if frozen_summary:
        frozen_metric_cols = st.columns(3)
        with frozen_metric_cols[0]:
            render_metric_card("Frozen Trades", str(int(frozen_summary.get("trade_count", 0))), "saved closed trades")
        with frozen_metric_cols[1]:
            render_metric_card("Frozen Capital", f"{float(frozen_summary.get('ending_equity', frozen_summary.get('starting_equity', 0.0))):.2f}", f"start {float(frozen_summary.get('starting_equity', 0.0)):.2f}")
        with frozen_metric_cols[2]:
            render_metric_card("Frozen Net PnL", f"{float(frozen_summary.get('net_pnl_cash', 0.0)):+.2f}", "saved closed-trade history")

    st.plotly_chart(build_equity_curve_figure(report), width="stretch")

    walk_forward = build_walk_forward_table(report)
    st.markdown("#### Walk-Forward Fold Metrics")
    if walk_forward.empty:
        st.info("Walk-forward fold metrics are not available in the current MT5 live paper mode.")
    else:
        st.dataframe(walk_forward, width="stretch", hide_index=True)

    st.markdown("#### Frozen Final Trade Summary")
    period_summary = build_trade_period_summary_table(final_trade_ledger)
    if period_summary.empty:
        st.info("No frozen final trades are available yet.")
    else:
        st.caption("These numbers come from the frozen final ledger, not from the latest reconstructed run only.")
        st.dataframe(period_summary, width="stretch", hide_index=True)

    st.markdown("#### Current Report Closed Paper Trades")
    paper_table = build_paper_ledger_table(paper_ledger, limit=25)
    if paper_table.empty:
        st.info("No closed paper trades are present in the current report window.")
    else:
        st.dataframe(paper_table, width="stretch", hide_index=True)

    st.markdown("#### Frozen Final Closed Trades")
    saved_trade_table = build_saved_trade_history_table(final_trade_ledger, limit=200)
    if saved_trade_table.empty:
        st.info("No frozen final trade rows are available yet.")
    else:
        st.dataframe(saved_trade_table, width="stretch", hide_index=True)

    st.markdown("#### Recently Blocked Signals")
    blocked = build_blocked_signal_table(predictions, limit=25)
    if blocked.empty:
        st.info("No blocked signals in the current report window.")
    else:
        st.dataframe(blocked, width="stretch", hide_index=True)


def render_dashboard() -> None:
    settings = get_ui_settings()
    run_every = get_auto_refresh_run_every(settings)

    @st.fragment(run_every=run_every)
    def _dashboard_fragment() -> None:
        render_dashboard_page(settings)

    _dashboard_fragment()


def render_chart_structure() -> None:
    settings = get_ui_settings()
    run_every = get_auto_refresh_run_every(settings)

    @st.fragment(run_every=run_every)
    def _chart_structure_fragment() -> None:
        render_chart_structure_page(settings)

    _chart_structure_fragment()


def render_sessions_psychology() -> None:
    settings = get_ui_settings()
    run_every = get_auto_refresh_run_every(settings)

    @st.fragment(run_every=run_every)
    def _sessions_fragment() -> None:
        render_sessions_psychology_page(settings)

    _sessions_fragment()


def render_model_lab() -> None:
    settings = get_ui_settings()
    run_every = get_auto_refresh_run_every(settings)

    @st.fragment(run_every=run_every)
    def _model_lab_fragment() -> None:
        render_model_lab_page(settings)

    _model_lab_fragment()


def render_backtest_walkforward() -> None:
    settings = get_ui_settings()
    run_every = get_auto_refresh_run_every(settings)

    @st.fragment(run_every=run_every)
    def _backtest_fragment() -> None:
        render_backtest_walkforward_page(settings)

    _backtest_fragment()


def render_global_popup() -> None:
    settings = get_ui_settings()
    if not bool(settings.get("live_enabled", False)):
        return

    run_every = get_auto_refresh_run_every(settings)

    @st.fragment(run_every=run_every)
    def _popup_fragment() -> None:
        refresh_key = build_refresh_key(settings)
        ribbon = cached_mt5_timeframe_ribbon(str(settings["mt5_symbol"]), True, refresh_key=refresh_key)
        render_m30_ribbon_card(ribbon)

    _popup_fragment()


inject_styles()
app_settings = sidebar_state()
st.session_state["_ui_settings"] = app_settings
pages = [
    st.Page(render_dashboard, title="Dashboard", url_path="", default=True),
    st.Page(render_chart_structure, title="Chart / Structure"),
    st.Page(render_sessions_psychology, title="Sessions / Psychology"),
    st.Page(render_model_lab, title="Model Lab"),
    st.Page(render_backtest_walkforward, title="Backtest / Walk-Forward"),
]
navigation = st.navigation(pages)
navigation.run()

render_global_popup()
