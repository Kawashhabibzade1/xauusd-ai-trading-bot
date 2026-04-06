from __future__ import annotations

import sys
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
    build_session_profile_figure,
    build_walk_forward_table,
    load_live_market_snapshot,
    load_research_bundle,
)


st.set_page_config(
    page_title="XAUUSD Research Cockpit",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


@st.cache_data(ttl=60, show_spinner=False)
def cached_bundle(report_path: str) -> dict:
    return load_research_bundle(report_path)


@st.cache_data(ttl=60, show_spinner=False)
def cached_live_snapshot(provider: str, symbol: str, mt5_symbol: str, outputsize: int, enabled: bool) -> dict:
    if not enabled:
        return {"enabled": False, "error": "Live snapshot disabled in the sidebar."}
    return load_live_market_snapshot(provider=provider, symbol=symbol, mt5_symbol=mt5_symbol, outputsize=outputsize)


def resolve_active_report_path(report_source: str, live_provider: str, custom_path: str) -> str:
    if report_source == "Custom":
        return custom_path.strip() or DEFAULT_RESEARCH_REPORT_OUTPUT
    if report_source == "Research Baseline":
        return DEFAULT_RESEARCH_REPORT_OUTPUT
    if report_source == "MT5 Live Paper":
        return DEFAULT_MT5_RESEARCH_REPORT_OUTPUT

    mt5_report = resolve_repo_path(DEFAULT_MT5_RESEARCH_REPORT_OUTPUT)
    if live_provider in {"Auto", "MT5 Local"} and mt5_report.exists():
        return DEFAULT_MT5_RESEARCH_REPORT_OUTPUT
    return DEFAULT_RESEARCH_REPORT_OUTPUT


def sidebar_state() -> dict[str, object]:
    st.sidebar.markdown("## Research Control")
    live_provider = st.sidebar.selectbox("Live provider", options=["Auto", "MT5 Local", "OANDA", "Twelve Data"], index=0)
    report_source = st.sidebar.selectbox(
        "Report source",
        options=["Auto", "MT5 Live Paper", "Research Baseline", "Custom"],
        index=0,
        help="Auto prefers the MT5 live paper-trading report whenever MT5 Local is active and a local MT5 report exists.",
    )
    custom_default = DEFAULT_MT5_RESEARCH_REPORT_OUTPUT if live_provider in {"Auto", "MT5 Local"} else DEFAULT_RESEARCH_REPORT_OUTPUT
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
    if live_provider == "MT5 Local":
        default_symbol = "XAUUSD"
    elif live_provider == "OANDA":
        default_symbol = "XAU_USD"
    else:
        default_symbol = "XAU/USD"
    live_symbol = st.sidebar.text_input("Live symbol / instrument", value=default_symbol, key=f"live_symbol_{live_provider.lower().replace(' ', '_')}")
    live_window = st.sidebar.slider("Live snapshot bars", min_value=20, max_value=240, value=120, step=20)
    if st.sidebar.button("Refresh cached data"):
        st.cache_data.clear()

    st.sidebar.markdown("## Run Pipeline")
    live_command = (
        "python src/install_mt5_exporter.py\n"
        "python src/run_mt5_research_pipeline.py --source-mode auto\n"
        "python src/run_mt5_research_worker.py --poll-seconds 15"
        if live_provider == "MT5 Local"
        else
        "python src/run_live_oanda_pipeline.py"
        if live_provider == "OANDA"
        else "python src/run_live_twelvedata_pipeline.py"
    )
    st.sidebar.code(
        live_command
        if live_provider != "Auto"
        else "python src/run_live_mt5_pipeline.py\npython src/run_live_oanda_pipeline.py\npython src/run_live_twelvedata_pipeline.py",
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
        "mt5_symbol": live_symbol if live_provider == "MT5 Local" else "XAUUSD",
        "live_window": live_window,
    }


def load_page_state() -> tuple[dict, pd.DataFrame, pd.DataFrame, dict, dict]:
    settings = sidebar_state()
    try:
        bundle = cached_bundle(str(settings["report_path"]))
    except FileNotFoundError:
        st.error("Research report not found yet. Run the research pipeline first.")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load research artifacts: {exc}")
        st.stop()

    predictions = bundle["predictions"]
    paper_ledger = bundle["paper_ledger"]
    report = bundle["report"]
    overlays = bundle["overlays"]
    live_snapshot = cached_live_snapshot(
        str(settings["live_provider"]),
        str(settings["live_symbol"]),
        str(settings["mt5_symbol"]),
        int(settings["live_window"]),
        bool(settings["live_enabled"]),
    )
    return settings, predictions, paper_ledger, report, overlays | {"_live_snapshot": live_snapshot}


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
    st.caption(
        f"Rows: standardized {dataset_summary.get('standardized_rows', 0):,} | "
        f"research {dataset_summary.get('research_rows', 0):,} | "
        f"labels {dataset_summary.get('label_rows', 0):,} | "
        f"predictions {dataset_summary.get('prediction_rows', 0):,} | "
        f"paper trades {dataset_summary.get('paper_trade_rows', 0):,}"
    )


def render_dashboard() -> None:
    settings, predictions, paper_ledger, report, overlay_bundle = load_page_state()
    live_snapshot = overlay_bundle["_live_snapshot"]
    latest = report.get("latest_signal", {})
    learning = report.get("learning", {})
    render_header(report)

    top_cols = st.columns(6)
    with top_cols[0]:
        render_metric_card("Recommended", str(latest.get("signal", "N/A")), latest.get("time", ""))
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
    paper_summary = report.get("paper_trading", {})
    with risk_cols[0]:
        render_metric_card("Paper Trades", str(paper_summary.get("trade_count", 0)), "Closed trades")
    with risk_cols[1]:
        render_metric_card("Profit Factor", f"{float(paper_summary.get('profit_factor', 0.0)):.2f}", "Paper ledger")
    with risk_cols[2]:
        render_metric_card("Max Drawdown", f"{float(paper_summary.get('max_drawdown', 0.0)):.1%}", "Risk ceiling")
    with risk_cols[3]:
        render_metric_card("Learning", "READY" if learning.get("retrain_ready") else "WAIT", f"{int(learning.get('closed_trades', 0))} closed trades")
    with risk_cols[4]:
        blockers = learning.get("retrain_blockers", [])
        render_metric_card("Learning Gate", str(len(blockers)), "all checks passed" if not blockers else "|".join(blockers))

    live_col, signal_col = st.columns([1.0, 1.4])
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


def render_chart_structure() -> None:
    settings, predictions, _, report, overlay_bundle = load_page_state()
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


def render_sessions_psychology() -> None:
    _, predictions, paper_ledger, report, _ = load_page_state()
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
        st.markdown("#### Session-Wise Paper Scoreboard")
        st.dataframe(session_scoreboard, width="stretch", hide_index=True)


def render_model_lab() -> None:
    _, predictions, _, report, _ = load_page_state()
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
    learning_cols = st.columns(4)
    with learning_cols[0]:
        render_metric_card("Batch Retrain", "READY" if learning.get("retrain_ready") else "NOT READY", learning.get("recommended_action", "collect_more_feedback"))
    with learning_cols[1]:
        render_metric_card("Closed Trades", str(int(learning.get("closed_trades", 0))), f"{int(learning.get('trading_days', 0))} trading days")
    with learning_cols[2]:
        render_metric_card("Direction Mix", f"L {int(learning.get('long_trades', 0))} / S {int(learning.get('short_trades', 0))}", learning.get("dominant_direction", ""))
    with learning_cols[3]:
        render_metric_card("Session Concentration", f"{float(learning.get('dominant_session_share', 0.0)):.1%}", learning.get("dominant_session", ""))

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


def render_backtest_walkforward() -> None:
    _, predictions, paper_ledger, report, _ = load_page_state()
    render_header(report)
    st.subheader("Paper Trading And Walk-Forward Evidence")

    backtest = report.get("paper_trading", report.get("backtest", {}))
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

    st.plotly_chart(build_equity_curve_figure(report), width="stretch")

    walk_forward = build_walk_forward_table(report)
    st.markdown("#### Walk-Forward Fold Metrics")
    st.dataframe(walk_forward, width="stretch", hide_index=True)

    st.markdown("#### Closed Paper Trades")
    paper_table = build_paper_ledger_table(paper_ledger, limit=25)
    if paper_table.empty:
        st.info("No closed paper trades are present in the current report window.")
    else:
        st.dataframe(paper_table, width="stretch", hide_index=True)

    st.markdown("#### Recently Blocked Signals")
    blocked = build_blocked_signal_table(predictions, limit=25)
    if blocked.empty:
        st.info("No blocked signals in the current report window.")
    else:
        st.dataframe(blocked, width="stretch", hide_index=True)


inject_styles()
pages = [
    st.Page(render_dashboard, title="Dashboard", url_path="", default=True),
    st.Page(render_chart_structure, title="Chart / Structure"),
    st.Page(render_sessions_psychology, title="Sessions / Psychology"),
    st.Page(render_model_lab, title="Model Lab"),
    st.Page(render_backtest_walkforward, title="Backtest / Walk-Forward"),
]
navigation = st.navigation(pages)
navigation.run()
