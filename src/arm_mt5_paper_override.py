"""
Arm a one-time paper-only override for the local MT5 worker.
"""

from __future__ import annotations

import argparse
from typing import Any

from pipeline_contract import (
    DEFAULT_MT5_RESEARCH_MANUAL_OVERRIDE_OUTPUT,
    DEFAULT_MT5_RESEARCH_REPORT_OUTPUT,
    json_dump,
    resolve_repo_path,
)
from run_mt5_research_pipeline import OVERRIDEABLE_RISK_BLOCKERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credits", type=int, default=2, help="How many extra paper-trade credits to arm.")
    parser.add_argument("--report-input", default=DEFAULT_MT5_RESEARCH_REPORT_OUTPUT, help="Current MT5 research report used to anchor the override start time.")
    parser.add_argument("--manual-override-output", default=DEFAULT_MT5_RESEARCH_MANUAL_OVERRIDE_OUTPUT, help="Manual override JSON state path.")
    parser.add_argument("--start-after", default="", help="Optional explicit timestamp. Defaults to the latest signal time in the current report.")
    return parser.parse_args()


def load_latest_signal_time(report_input: str) -> str:
    import json

    with resolve_repo_path(report_input).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return str(report.get("latest_signal", {}).get("time", "")).strip()


def main() -> None:
    args = parse_args()
    start_after = args.start_after.strip() or load_latest_signal_time(args.report_input)
    payload: dict[str, Any] = {
        "enabled": True,
        "scope": "paper_only",
        "note": "One-time manual paper override to allow exactly two future MT5 paper trades when valid signals appear.",
        "initial_credits": max(0, int(args.credits)),
        "remaining_credits": max(0, int(args.credits)),
        "used_signal_times": [],
        "allowed_risk_blockers": sorted(OVERRIDEABLE_RISK_BLOCKERS),
        "start_after": start_after,
        "last_used_at": "",
    }
    json_dump(payload, args.manual_override_output)
    print("Armed MT5 paper override.")
    print(f"Credits     : {payload['remaining_credits']}")
    print(f"Start after : {payload['start_after'] or 'immediately'}")
    print(f"Output      : {args.manual_override_output}")


if __name__ == "__main__":
    main()
