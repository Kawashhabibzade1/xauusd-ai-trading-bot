"""
Poll the MT5 exporter CSV and rerun the MT5 research/paper-trading pipeline whenever a new bar arrives.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mt5_client import resolve_export_file_path
from pipeline_contract import (
    DEFAULT_MT5_RESEARCH_REPORT_OUTPUT,
    DEFAULT_MT5_RESEARCH_WORKER_STATE_OUTPUT,
    json_dump,
)
from run_mt5_research_pipeline import run_mt5_research_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=5, help="How often to check the MT5 exporter CSV.")
    parser.add_argument("--once", action="store_true", help="Run only one cycle and exit.")
    parser.add_argument("--report-output", default=DEFAULT_MT5_RESEARCH_REPORT_OUTPUT, help="MT5 research report output path.")
    parser.add_argument("--worker-state-output", default=DEFAULT_MT5_RESEARCH_WORKER_STATE_OUTPUT, help="Worker state JSON output.")
    return parser.parse_args()


def run_cycle(report_output: str) -> dict:
    report = run_mt5_research_pipeline(report_output=report_output)
    latest_signal = report.get("latest_signal", {})
    paper = report.get("paper_trading", {})
    learning = report.get("learning", {})
    return {
        "last_run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "report_output": report_output,
        "latest_signal_time": latest_signal.get("time"),
        "latest_signal": latest_signal.get("signal"),
        "paper_trade_count": paper.get("trade_count", 0),
        "paper_precision": paper.get("precision", 0.0),
        "paper_profit_factor": paper.get("profit_factor", 0.0),
        "learning_closed_trades": learning.get("closed_trades", 0),
        "learning_retrain_ready": learning.get("retrain_ready", False),
        "learning_blockers": learning.get("retrain_blockers", []),
    }


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("MT5 RESEARCH WORKER")
    print("=" * 70)
    print(f"Report output : {args.report_output}")
    print(f"Poll seconds  : {args.poll_seconds}")
    print()

    last_mtime = None
    while True:
        source = Path(resolve_export_file_path())
        if source.exists():
            current_mtime = source.stat().st_mtime
            if last_mtime is None or current_mtime > last_mtime:
                state = run_cycle(args.report_output)
                state["source_file"] = str(source)
                state["source_mtime"] = current_mtime
                json_dump(state, args.worker_state_output)
                print(f"[{state['last_run_time']}] Updated report from MT5 exporter. latest={state['latest_signal']} trades={state['paper_trade_count']}")
                last_mtime = current_mtime
                if args.once:
                    break
        elif args.once:
            raise SystemExit(f"MT5 exporter CSV not found: {source}")

        if args.once:
            break
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
