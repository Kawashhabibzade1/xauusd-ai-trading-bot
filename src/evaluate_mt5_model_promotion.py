"""
Evaluate whether a candidate MT5 model is safe to promote over the current local paper-trading baseline.

The candidate metrics file is expected to be a small JSON document, for example:

{
  "candidate_name": "patchtst-v1",
  "paper_trading": {
    "trade_count": 42,
    "precision": 0.57,
    "profit_factor": 1.28,
    "expectancy_r": 0.09,
    "max_drawdown": 0.045
  }
}
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import yaml

from pipeline_contract import (
    DEFAULT_MT5_RESEARCH_CONFIG,
    DEFAULT_MT5_RESEARCH_PROMOTION_DECISION_OUTPUT,
    DEFAULT_MT5_RESEARCH_REPORT_OUTPUT,
    json_dump,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", required=True, help="JSON file with candidate paper-trading metrics.")
    parser.add_argument("--current-report", default=DEFAULT_MT5_RESEARCH_REPORT_OUTPUT, help="Current production MT5 paper-trading report.")
    parser.add_argument("--config", default=DEFAULT_MT5_RESEARCH_CONFIG, help="MT5 live research config with promotion thresholds.")
    parser.add_argument("--output", default=DEFAULT_MT5_RESEARCH_PROMOTION_DECISION_OUTPUT, help="Promotion decision JSON output.")
    return parser.parse_args()


def load_json(path_like: str) -> dict[str, Any]:
    with resolve_repo_path(path_like).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path_like: str) -> dict[str, Any]:
    with resolve_repo_path(path_like).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def metric(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(payload.get(key, default) if payload.get(key, default) is not None else default)


def evaluate_promotion(candidate_payload: dict[str, Any], current_report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["promotion"]
    candidate = candidate_payload.get("paper_trading", candidate_payload)
    current = current_report.get("paper_trading", {})

    current_trade_count = int(current.get("trade_count", 0))
    current_precision = metric(current, "precision")
    current_profit_factor = metric(current, "profit_factor")
    current_drawdown = metric(current, "max_drawdown")

    candidate_trade_count = int(candidate.get("trade_count", 0))
    candidate_precision = metric(candidate, "precision")
    candidate_profit_factor = metric(candidate, "profit_factor")
    candidate_expectancy = metric(candidate, "expectancy_r")
    candidate_drawdown = metric(candidate, "max_drawdown")

    checks = [
        {
            "check": "candidate_trade_count",
            "passed": candidate_trade_count >= int(thresholds["min_candidate_trades"]),
            "current": candidate_trade_count,
            "required": int(thresholds["min_candidate_trades"]),
        },
        {
            "check": "candidate_profit_factor_floor",
            "passed": candidate_profit_factor >= float(thresholds["min_profit_factor"]),
            "current": candidate_profit_factor,
            "required": float(thresholds["min_profit_factor"]),
        },
        {
            "check": "candidate_precision_floor",
            "passed": candidate_precision >= float(thresholds["min_precision"]),
            "current": candidate_precision,
            "required": float(thresholds["min_precision"]),
        },
        {
            "check": "candidate_expectancy_floor",
            "passed": candidate_expectancy >= float(thresholds["min_expectancy_r"]),
            "current": candidate_expectancy,
            "required": float(thresholds["min_expectancy_r"]),
        },
        {
            "check": "candidate_drawdown_ceiling",
            "passed": candidate_drawdown <= float(thresholds["max_drawdown"]),
            "current": candidate_drawdown,
            "required": f"<= {float(thresholds['max_drawdown']):.4f}",
        },
    ]

    if current_trade_count >= int(thresholds["min_candidate_trades"]):
        checks.extend(
            [
                {
                    "check": "profit_factor_delta_vs_current",
                    "passed": candidate_profit_factor >= current_profit_factor + float(thresholds["require_profit_factor_delta"]),
                    "current": candidate_profit_factor - current_profit_factor,
                    "required": f">= {float(thresholds['require_profit_factor_delta']):.4f}",
                },
                {
                    "check": "precision_regression_vs_current",
                    "passed": candidate_precision >= current_precision - float(thresholds["max_precision_regression"]),
                    "current": candidate_precision - current_precision,
                    "required": f">= -{float(thresholds['max_precision_regression']):.4f}",
                },
                {
                    "check": "drawdown_delta_vs_current",
                    "passed": candidate_drawdown <= current_drawdown + float(thresholds["max_drawdown_delta"]),
                    "current": candidate_drawdown - current_drawdown,
                    "required": f"<= {float(thresholds['max_drawdown_delta']):.4f}",
                },
            ]
        )

    blockers = [check["check"] for check in checks if not check["passed"]]
    return {
        "candidate_name": candidate_payload.get("candidate_name", "unnamed_candidate"),
        "promote": not blockers,
        "blockers": blockers,
        "checks": checks,
        "current_baseline": {
            "trade_count": current_trade_count,
            "precision": current_precision,
            "profit_factor": current_profit_factor,
            "max_drawdown": current_drawdown,
        },
        "candidate": {
            "trade_count": candidate_trade_count,
            "precision": candidate_precision,
            "profit_factor": candidate_profit_factor,
            "expectancy_r": candidate_expectancy,
            "max_drawdown": candidate_drawdown,
        },
    }


def main() -> None:
    args = parse_args()
    candidate = load_json(args.candidate_metrics)
    current_report = load_json(args.current_report)
    config = load_config(args.config)
    decision = evaluate_promotion(candidate, current_report, config)
    json_dump(decision, args.output)

    print("MT5 model promotion decision")
    print(f"Candidate : {decision['candidate_name']}")
    print(f"Promote   : {decision['promote']}")
    print(f"Output    : {args.output}")
    if decision["blockers"]:
        print("Blockers  : " + ", ".join(decision["blockers"]))


if __name__ == "__main__":
    main()
