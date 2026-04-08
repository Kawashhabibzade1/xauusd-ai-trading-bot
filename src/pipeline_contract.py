"""
Shared paths, feature contract, and validation helpers for the XAUUSD pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RAW_INPUT = "data/xauusd_m1_2022_2025.csv"
DEFAULT_STANDARDIZED_OUTPUT = "data/processed/xauusd_m1_standardized.csv"
DEFAULT_OVERLAP_OUTPUT = "data/processed/xauusd_m1_overlap.csv"
DEFAULT_FEATURE_OUTPUT = "data/processed/xauusd_features.csv"
DEFAULT_LABEL_OUTPUT = "data/processed/xauusd_labeled.csv"
DEFAULT_LIVE_RAW_INPUT = "data/live/xauusd_twelvedata_raw.csv"
DEFAULT_OANDA_LIVE_RAW_INPUT = "data/live/xauusd_oanda_raw.csv"
DEFAULT_MT5_LIVE_RAW_INPUT = "data/live/xauusd_mt5_raw.csv"
DEFAULT_LIVE_STANDARDIZED_OUTPUT = "data/live/processed/xauusd_m1_standardized.csv"
DEFAULT_LIVE_OVERLAP_OUTPUT = "data/live/processed/xauusd_m1_overlap.csv"
DEFAULT_LIVE_FEATURE_OUTPUT = "data/live/processed/xauusd_features.csv"
DEFAULT_LIVE_LABEL_OUTPUT = "data/live/processed/xauusd_labeled.csv"
DEFAULT_OANDA_LIVE_STANDARDIZED_OUTPUT = "data/live/oanda_processed/xauusd_m1_standardized.csv"
DEFAULT_OANDA_LIVE_OVERLAP_OUTPUT = "data/live/oanda_processed/xauusd_m1_overlap.csv"
DEFAULT_OANDA_LIVE_FEATURE_OUTPUT = "data/live/oanda_processed/xauusd_features.csv"
DEFAULT_OANDA_LIVE_LABEL_OUTPUT = "data/live/oanda_processed/xauusd_labeled.csv"
DEFAULT_MT5_LIVE_STANDARDIZED_OUTPUT = "data/live/mt5_processed/xauusd_m1_standardized.csv"
DEFAULT_MT5_LIVE_OVERLAP_OUTPUT = "data/live/mt5_processed/xauusd_m1_overlap.csv"
DEFAULT_MT5_LIVE_FEATURE_OUTPUT = "data/live/mt5_processed/xauusd_features.csv"
DEFAULT_MT5_LIVE_LABEL_OUTPUT = "data/live/mt5_processed/xauusd_labeled.csv"
DEFAULT_MODEL_PATH = "python_training/models/lightgbm_xauusd_v1.txt"
DEFAULT_FEATURE_LIST_PATH = "python_training/models/feature_list.json"
DEFAULT_METADATA_PATH = "python_training/models/model_metadata.json"
DEFAULT_BACKTEST_RESULTS_PATH = "python_training/models/backtest_results.json"
DEFAULT_LIVE_MODEL_PATH = "python_training/models/live_twelvedata/lightgbm_xauusd_live.txt"
DEFAULT_LIVE_FEATURE_LIST_PATH = "python_training/models/live_twelvedata/feature_list.json"
DEFAULT_LIVE_METADATA_PATH = "python_training/models/live_twelvedata/model_metadata.json"
DEFAULT_LIVE_BACKTEST_RESULTS_PATH = "python_training/models/live_twelvedata/backtest_results.json"
DEFAULT_LIVE_CONFIDENCE_RESULTS_PATH = "python_training/models/live_twelvedata/confidence_analysis.json"
DEFAULT_OANDA_LIVE_MODEL_PATH = "python_training/models/live_oanda/lightgbm_xauusd_live.txt"
DEFAULT_OANDA_LIVE_FEATURE_LIST_PATH = "python_training/models/live_oanda/feature_list.json"
DEFAULT_OANDA_LIVE_METADATA_PATH = "python_training/models/live_oanda/model_metadata.json"
DEFAULT_OANDA_LIVE_BACKTEST_RESULTS_PATH = "python_training/models/live_oanda/backtest_results.json"
DEFAULT_OANDA_LIVE_CONFIDENCE_RESULTS_PATH = "python_training/models/live_oanda/confidence_analysis.json"
DEFAULT_MT5_LIVE_MODEL_PATH = "python_training/models/live_mt5/lightgbm_xauusd_live.txt"
DEFAULT_MT5_LIVE_FEATURE_LIST_PATH = "python_training/models/live_mt5/feature_list.json"
DEFAULT_MT5_LIVE_METADATA_PATH = "python_training/models/live_mt5/model_metadata.json"
DEFAULT_MT5_LIVE_BACKTEST_RESULTS_PATH = "python_training/models/live_mt5/backtest_results.json"
DEFAULT_MT5_LIVE_CONFIDENCE_RESULTS_PATH = "python_training/models/live_mt5/confidence_analysis.json"
DEFAULT_FEATURE_CONFIG = "python_training/config/features.yaml"
DEFAULT_MODEL_CONFIG = "python_training/config/model_config.yaml"
DEFAULT_MT5_FEATURES_PATH = "mt5_expert_advisor/Files/config/features.json"
DEFAULT_MT5_MODEL_CONFIG_PATH = "mt5_expert_advisor/Files/config/model_config.json"
DEFAULT_MT5_ONNX_OUTPUT = "mt5_expert_advisor/Files/models/xauusd_ai_v1.onnx"
DEFAULT_MT5_VALIDATION_OUTPUT = "mt5_expert_advisor/Files/config/validation_set.csv"
DEFAULT_LIVE_MT5_FEATURES_PATH = "mt5_expert_advisor/Files/config/features_live.json"
DEFAULT_LIVE_MT5_MODEL_CONFIG_PATH = "mt5_expert_advisor/Files/config/model_config_live.json"
DEFAULT_LIVE_MT5_ONNX_OUTPUT = "mt5_expert_advisor/Files/models/xauusd_ai_live.onnx"
DEFAULT_LIVE_MT5_VALIDATION_OUTPUT = "mt5_expert_advisor/Files/config/validation_set_live.csv"
DEFAULT_LIVE_REPORT_PATH = "data/live/live_pipeline_report.json"
DEFAULT_OANDA_LIVE_MT5_FEATURES_PATH = "mt5_expert_advisor/Files/config/features_oanda_live.json"
DEFAULT_OANDA_LIVE_MT5_MODEL_CONFIG_PATH = "mt5_expert_advisor/Files/config/model_config_oanda_live.json"
DEFAULT_OANDA_LIVE_MT5_ONNX_OUTPUT = "mt5_expert_advisor/Files/models/xauusd_ai_oanda_live.onnx"
DEFAULT_OANDA_LIVE_MT5_VALIDATION_OUTPUT = "mt5_expert_advisor/Files/config/validation_set_oanda_live.csv"
DEFAULT_OANDA_LIVE_REPORT_PATH = "data/live/live_oanda_pipeline_report.json"
DEFAULT_MT5_LIVE_MT5_FEATURES_PATH = "mt5_expert_advisor/Files/config/features_mt5_live.json"
DEFAULT_MT5_LIVE_MT5_MODEL_CONFIG_PATH = "mt5_expert_advisor/Files/config/model_config_mt5_live.json"
DEFAULT_MT5_LIVE_MT5_ONNX_OUTPUT = "mt5_expert_advisor/Files/models/xauusd_ai_mt5_live.onnx"
DEFAULT_MT5_LIVE_MT5_VALIDATION_OUTPUT = "mt5_expert_advisor/Files/config/validation_set_mt5_live.csv"
DEFAULT_MT5_LIVE_REPORT_PATH = "data/live/live_mt5_pipeline_report.json"
DEFAULT_RESEARCH_CONFIG = "python_training/config/research_config.yaml"
DEFAULT_MT5_RESEARCH_CONFIG = "python_training/config/research_config_mt5_live.yaml"
DEFAULT_RESEARCH_STANDARDIZED_OUTPUT = "data/research/xauusd_m1_standardized.csv"
DEFAULT_RESEARCH_READY_OUTPUT = "data/research/xauusd_research_ready.csv"
DEFAULT_RESEARCH_LABEL_OUTPUT = "data/research/xauusd_research_labels.csv"
DEFAULT_RESEARCH_OVERLAYS_OUTPUT = "data/research/xauusd_research_overlays.json"
DEFAULT_RESEARCH_PREDICTIONS_OUTPUT = "data/research/xauusd_research_predictions.csv"
DEFAULT_RESEARCH_PAPER_LEDGER_OUTPUT = "data/research/xauusd_paper_ledger.csv"
DEFAULT_RESEARCH_REPORT_OUTPUT = "data/research/xauusd_research_report.json"
DEFAULT_MT5_RESEARCH_STANDARDIZED_OUTPUT = "data/live/research_mt5/xauusd_m1_standardized.csv"
DEFAULT_MT5_RESEARCH_READY_OUTPUT = "data/live/research_mt5/xauusd_research_ready.csv"
DEFAULT_MT5_RESEARCH_LABEL_OUTPUT = "data/live/research_mt5/xauusd_research_labels.csv"
DEFAULT_MT5_RESEARCH_OVERLAYS_OUTPUT = "data/live/research_mt5/xauusd_research_overlays.json"
DEFAULT_MT5_RESEARCH_PREDICTIONS_OUTPUT = "data/live/research_mt5/xauusd_research_predictions.csv"
DEFAULT_MT5_RESEARCH_PAPER_LEDGER_OUTPUT = "data/live/research_mt5/xauusd_paper_ledger.csv"
DEFAULT_MT5_RESEARCH_REPORT_OUTPUT = "data/live/research_mt5/xauusd_research_report.json"
DEFAULT_MT5_RESEARCH_FEEDBACK_OUTPUT = "data/live/research_mt5/xauusd_learning_feedback.csv"
DEFAULT_MT5_RESEARCH_LEARNING_STATUS_OUTPUT = "data/live/research_mt5/learning_status.json"
DEFAULT_MT5_RESEARCH_PROMOTION_DECISION_OUTPUT = "data/live/research_mt5/promotion_decision.json"
DEFAULT_MT5_RESEARCH_WORKER_STATE_OUTPUT = "data/live/research_mt5/worker_state.json"
DEFAULT_MT5_RESEARCH_MANUAL_OVERRIDE_OUTPUT = "data/live/research_mt5/manual_override.json"
DEFAULT_MT5_RESEARCH_NOTIFICATION_STATE_OUTPUT = "data/live/research_mt5/notification_state.json"
DEFAULT_MT5_RESEARCH_TRADE_HISTORY_OUTPUT = "data/live/research_mt5/xauusd_trade_history.csv"
DEFAULT_MT5_RESEARCH_FINAL_LEDGER_OUTPUT = "data/live/research_mt5/xauusd_final_trade_ledger.csv"
DEFAULT_MT5_RESEARCH_SNAPSHOT_DIR = "data/live/research_mt5/run_snapshots"
DEFAULT_MT5_TRADE_DIRECTIVE_OUTPUT = "mt5_expert_advisor/Files/config/mt5_trade_directive.csv"
DEFAULT_RESEARCH_BASELINE_DIR = "python_training/models/research/baseline"
DEFAULT_RESEARCH_NEURAL_DIR = "python_training/models/research/patchtst"
DEFAULT_RESEARCH_ENSEMBLE_DIR = "python_training/models/research/ensemble"

BASE_COLUMNS = ("time", "open", "high", "low", "close", "volume")
LABEL_EXCLUDE_COLUMNS = BASE_COLUMNS + ("forward_return_15m", "label")
MODEL_CLASS_MAPPING = {0: "SHORT", 1: "HOLD", 2: "LONG"}


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path_like: str | Path) -> str:
    path = resolve_repo_path(path_like)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_parent_dir(path_like: str | Path) -> Path:
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_feature_config(config_path: str | Path = DEFAULT_FEATURE_CONFIG) -> dict:
    path = resolve_repo_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict) or "ordered_features" not in config:
        raise ValueError(f"Feature config at {display_path(path)} is missing 'ordered_features'.")

    ordered = config["ordered_features"]
    if not isinstance(ordered, list) or not ordered:
        raise ValueError(f"Feature config at {display_path(path)} has an invalid 'ordered_features' list.")

    if len(set(ordered)) != len(ordered):
        raise ValueError(f"Feature config at {display_path(path)} contains duplicate feature names.")

    config["ordered_features"] = [str(feature) for feature in ordered]
    return config


def get_ordered_features(config_path: str | Path = DEFAULT_FEATURE_CONFIG) -> list[str]:
    return load_feature_config(config_path)["ordered_features"]


def get_discrete_features(config_path: str | Path = DEFAULT_FEATURE_CONFIG) -> set[str]:
    config = load_feature_config(config_path)
    discrete = config.get("discrete_features", [])
    return {str(feature) for feature in discrete}


def get_feature_count(config_path: str | Path = DEFAULT_FEATURE_CONFIG) -> int:
    return len(get_ordered_features(config_path))


def assert_ordered_features(
    feature_columns: Sequence[str],
    config_path: str | Path = DEFAULT_FEATURE_CONFIG,
    context: str = "feature set",
) -> list[str]:
    expected = get_ordered_features(config_path)
    actual = list(feature_columns)
    if actual != expected:
        missing = [feature for feature in expected if feature not in actual]
        extra = [feature for feature in actual if feature not in expected]
        first_mismatch = next(
            (
                (index, exp, got)
                for index, (exp, got) in enumerate(zip(expected, actual))
                if exp != got
            ),
            None,
        )

        problems: list[str] = []
        if missing:
            problems.append(f"missing={missing}")
        if extra:
            problems.append(f"extra={extra}")
        if first_mismatch is not None:
            index, expected_name, actual_name = first_mismatch
            problems.append(
                f"first_order_mismatch=index {index}: expected '{expected_name}' got '{actual_name}'"
            )
        if len(actual) != len(expected):
            problems.append(f"expected_count={len(expected)} actual_count={len(actual)}")

        raise ValueError(f"{context} does not match the 68-feature contract: {'; '.join(problems)}")

    return expected


def json_dump(data: object, path_like: str | Path) -> Path:
    path = ensure_parent_dir(path_like)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return path


def write_feature_list(feature_names: Iterable[str], path_like: str | Path) -> Path:
    return json_dump(list(feature_names), path_like)


def build_mt5_model_config(
    feature_names: Sequence[str],
    model_filename: str,
    num_trees: int,
    confidence_threshold: float = 0.55,
) -> dict:
    return {
        "model_info": {
            "file": model_filename,
            "type": "LightGBM",
            "num_features": len(feature_names),
            "num_classes": len(MODEL_CLASS_MAPPING),
            "num_trees": num_trees,
            "validation_mode": "validation_first",
        },
        "trading_config": {
            "confidence_threshold": confidence_threshold,
            "validation_mode": True,
            "max_trades_per_day": 0,
        },
        "class_mapping": {str(key): value for key, value in MODEL_CLASS_MAPPING.items()},
        "feature_names": list(feature_names),
        "feature_contract_version": 1,
    }
