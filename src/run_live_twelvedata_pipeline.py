"""
Fetch recent XAU/USD bars from Twelve Data and run the live pipeline end to end.

This mode is explicitly experimental because Twelve Data's XAU/USD feed does not
currently include volume, while the 68-feature model contract expects volume-aware
bars. When source volume is missing, this script synthesizes it before running the
existing pipeline.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from analyze_confidence import analyze_confidence_from_labeled
from backtest_simple import run_signal_simulation
from create_labels import create_labeled_frame
from export_mt5_validation_set import export_validation_fixture
from export_to_onnx_simple import export_model_to_onnx
from feature_engineering import compute_feature_frame
from filter_overlap import filter_overlap_frame
from pipeline_contract import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_LIVE_BACKTEST_RESULTS_PATH,
    DEFAULT_LIVE_CONFIDENCE_RESULTS_PATH,
    DEFAULT_LIVE_FEATURE_LIST_PATH,
    DEFAULT_LIVE_FEATURE_OUTPUT,
    DEFAULT_LIVE_LABEL_OUTPUT,
    DEFAULT_LIVE_METADATA_PATH,
    DEFAULT_LIVE_MODEL_PATH,
    DEFAULT_LIVE_MT5_FEATURES_PATH,
    DEFAULT_LIVE_MT5_MODEL_CONFIG_PATH,
    DEFAULT_LIVE_MT5_ONNX_OUTPUT,
    DEFAULT_LIVE_MT5_VALIDATION_OUTPUT,
    DEFAULT_LIVE_OVERLAP_OUTPUT,
    DEFAULT_LIVE_RAW_INPUT,
    DEFAULT_LIVE_REPORT_PATH,
    DEFAULT_LIVE_STANDARDIZED_OUTPUT,
    DEFAULT_MODEL_CONFIG,
    MODEL_CLASS_MAPPING,
    display_path,
    ensure_parent_dir,
    json_dump,
)
from run_sample_demo import verify_dependencies
from train_lightgbm import train_model_from_labeled
from twelvedata_client import fetch_and_write_time_series_csv, resolve_api_key
from validate_merged_data import load_and_standardize, validate_standardized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-env", default="TWELVEDATA_API_KEY", help="Environment variable containing the Twelve Data API key.")
    parser.add_argument("--symbol", default="XAU/USD", help="Twelve Data symbol.")
    parser.add_argument("--interval", default="1min", help="Twelve Data interval.")
    parser.add_argument("--outputsize", type=int, default=5000, help="Number of recent bars to request from Twelve Data.")
    parser.add_argument(
        "--volume-mode",
        choices=("constant", "range_proxy"),
        default="range_proxy",
        help="Fallback volume synthesis mode when the Twelve Data feed has no volume.",
    )
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="Feature contract YAML path.")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="Training config YAML path.")
    parser.add_argument("--raw-output", default=DEFAULT_LIVE_RAW_INPUT, help="Fetched raw OHLCV CSV output.")
    parser.add_argument("--standardized-output", default=DEFAULT_LIVE_STANDARDIZED_OUTPUT, help="Standardized CSV output.")
    parser.add_argument("--overlap-output", default=DEFAULT_LIVE_OVERLAP_OUTPUT, help="Filtered overlap CSV output.")
    parser.add_argument("--feature-output", default=DEFAULT_LIVE_FEATURE_OUTPUT, help="Engineered features CSV output.")
    parser.add_argument("--label-output", default=DEFAULT_LIVE_LABEL_OUTPUT, help="Labeled live dataset CSV output.")
    parser.add_argument("--model-output", default=DEFAULT_LIVE_MODEL_PATH, help="Live LightGBM model output path.")
    parser.add_argument("--feature-list-output", default=DEFAULT_LIVE_FEATURE_LIST_PATH, help="Live feature list JSON output.")
    parser.add_argument("--metadata-output", default=DEFAULT_LIVE_METADATA_PATH, help="Live model metadata JSON output.")
    parser.add_argument(
        "--confidence-output",
        default=DEFAULT_LIVE_CONFIDENCE_RESULTS_PATH,
        help="Confidence analysis JSON output.",
    )
    parser.add_argument("--backtest-output", default=DEFAULT_LIVE_BACKTEST_RESULTS_PATH, help="Approximate live backtest JSON output.")
    parser.add_argument(
        "--validation-output",
        default=DEFAULT_LIVE_MT5_VALIDATION_OUTPUT,
        help="Live MT5 validation fixture CSV output.",
    )
    parser.add_argument("--onnx-output", default=DEFAULT_LIVE_MT5_ONNX_OUTPUT, help="Live ONNX model output path.")
    parser.add_argument(
        "--mt5-features-output",
        default=DEFAULT_LIVE_MT5_FEATURES_PATH,
        help="MT5 live feature-list JSON output.",
    )
    parser.add_argument(
        "--mt5-config-output",
        default=DEFAULT_LIVE_MT5_MODEL_CONFIG_PATH,
        help="MT5 live model-config JSON output.",
    )
    parser.add_argument("--report-output", default=DEFAULT_LIVE_REPORT_PATH, help="Live pipeline report JSON output.")
    parser.add_argument("--skip-onnx-runtime-check", action="store_true", help="Skip optional onnxruntime verification.")
    parser.add_argument("--skip-confidence-analysis", action="store_true", help="Skip confidence threshold analysis.")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip approximate backtest generation.")
    return parser.parse_args()


def run_live_pipeline(
    api_env: str = "TWELVEDATA_API_KEY",
    symbol: str = "XAU/USD",
    interval: str = "1min",
    outputsize: int = 5000,
    volume_mode: str = "range_proxy",
    feature_config: str = DEFAULT_FEATURE_CONFIG,
    model_config_path: str = DEFAULT_MODEL_CONFIG,
    raw_output: str = DEFAULT_LIVE_RAW_INPUT,
    standardized_output: str = DEFAULT_LIVE_STANDARDIZED_OUTPUT,
    overlap_output: str = DEFAULT_LIVE_OVERLAP_OUTPUT,
    feature_output: str = DEFAULT_LIVE_FEATURE_OUTPUT,
    label_output: str = DEFAULT_LIVE_LABEL_OUTPUT,
    model_output: str = DEFAULT_LIVE_MODEL_PATH,
    feature_list_output: str = DEFAULT_LIVE_FEATURE_LIST_PATH,
    metadata_output: str = DEFAULT_LIVE_METADATA_PATH,
    confidence_output: str = DEFAULT_LIVE_CONFIDENCE_RESULTS_PATH,
    backtest_output: str = DEFAULT_LIVE_BACKTEST_RESULTS_PATH,
    validation_output: str = DEFAULT_LIVE_MT5_VALIDATION_OUTPUT,
    onnx_output: str = DEFAULT_LIVE_MT5_ONNX_OUTPUT,
    mt5_features_output: str = DEFAULT_LIVE_MT5_FEATURES_PATH,
    mt5_config_output: str = DEFAULT_LIVE_MT5_MODEL_CONFIG_PATH,
    report_output: str = DEFAULT_LIVE_REPORT_PATH,
    skip_onnx_runtime_check: bool = False,
    skip_confidence_analysis: bool = False,
    skip_backtest: bool = False,
) -> dict:
    verify_dependencies(skip_onnx_runtime_check)

    api_key = resolve_api_key(api_env)
    if not api_key:
        raise SystemExit(
            f"Twelve Data API key not found in environment variable {api_env}. "
            "Export it in your shell before running the live pipeline."
        )

    fetch_result = fetch_and_write_time_series_csv(
        api_key=api_key,
        output_path=raw_output,
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
        volume_mode=volume_mode,
    )

    standardized = load_and_standardize(raw_output)
    standardized_stats = validate_standardized(standardized)
    standardized_output_path = ensure_parent_dir(standardized_output)
    standardized.to_csv(standardized_output_path, index=False)

    overlap = filter_overlap_frame(standardized)
    overlap_output_path = ensure_parent_dir(overlap_output)
    overlap.to_csv(overlap_output_path, index=False)

    features = compute_feature_frame(overlap, feature_config)
    if features.empty:
        raise SystemExit(
            "Live feature generation produced 0 rows. Increase the Twelve Data history window or adjust the live pipeline inputs."
        )
    feature_output_path = ensure_parent_dir(feature_output)
    features.to_csv(feature_output_path, index=False)

    labeled = create_labeled_frame(features, feature_config=feature_config)
    if labeled.empty:
        raise SystemExit(
            "Live label generation produced 0 rows. The fetched history window is too small for the current feature and label contract."
        )
    label_output_path = ensure_parent_dir(label_output)
    labeled.to_csv(label_output_path, index=False)

    training_result = train_model_from_labeled(
        labeled=labeled,
        feature_config=feature_config,
        model_config_path=model_config_path,
        model_output_path=model_output,
        feature_list_output_path=feature_list_output,
        metadata_output_path=metadata_output,
    )

    confidence_result = None
    if not skip_confidence_analysis:
        confidence_result = analyze_confidence_from_labeled(
            labeled=labeled,
            model_path=model_output,
            feature_config=feature_config,
            feature_list_path=feature_list_output,
        )
        json_dump(confidence_result, confidence_output)

    backtest_result = None
    if not skip_backtest:
        backtest_result = run_signal_simulation(
            labeled=labeled,
            model_path=model_output,
            feature_list_path=feature_list_output,
            feature_config=feature_config,
            output_path=backtest_output,
        )

    fixture_result = export_validation_fixture(
        input_path=overlap_output,
        model_path=model_output,
        output_path=validation_output,
        feature_config=feature_config,
    )

    onnx_result = export_model_to_onnx(
        model_path=model_output,
        feature_list_path=feature_list_output,
        feature_config=feature_config,
        output_path=onnx_output,
        mt5_features_output=mt5_features_output,
        mt5_config_output=mt5_config_output,
        skip_runtime_check=skip_onnx_runtime_check,
    )

    validation_df = pd.read_csv(fixture_result["output_path"])
    latest_prediction = validation_df.iloc[-1].to_dict()
    predicted_class = int(latest_prediction["expected_class"])

    report = {
        "mode": "live_twelvedata",
        "symbol": fetch_result["symbol"],
        "interval": fetch_result["interval"],
        "source_timezone": fetch_result["timezone"],
        "source_rows": fetch_result["rows"],
        "source_start": fetch_result["start"],
        "source_end": fetch_result["end"],
        "source_has_volume": fetch_result["has_source_volume"],
        "synthesized_volume": fetch_result["synthesized_volume"],
        "volume_mode": fetch_result["volume_mode"],
        "standardized_rows": len(standardized),
        "standardized_start": str(standardized_stats["start"]),
        "standardized_end": str(standardized_stats["end"]),
        "overlap_rows": len(overlap),
        "overlap_start": str(overlap["time"].min()),
        "overlap_end": str(overlap["time"].max()),
        "feature_rows": len(features),
        "feature_time": str(features["time"].iloc[-1]),
        "labeled_rows": len(labeled),
        "validation_rows": fixture_result["rows"],
        "runtime_status": onnx_result["runtime_status"],
        "prediction": {
            "time": str(latest_prediction["time"]),
            "label": MODEL_CLASS_MAPPING[predicted_class],
            "prob_short": float(latest_prediction["prob_short"]),
            "prob_hold": float(latest_prediction["prob_hold"]),
            "prob_long": float(latest_prediction["prob_long"]),
        },
        "training": {
            "accuracy": float(training_result["accuracy"]),
            "best_iteration": int(training_result["best_iteration"]),
            "train_samples": int(training_result["metadata"]["train_samples"]),
            "test_samples": int(training_result["metadata"]["test_samples"]),
        },
        "confidence": confidence_result,
        "backtest": backtest_result,
        "artifacts": {
            "raw_output": raw_output,
            "standardized_output": standardized_output,
            "overlap_output": overlap_output,
            "feature_output": feature_output,
            "label_output": label_output,
            "model_output": model_output,
            "feature_list_output": feature_list_output,
            "metadata_output": metadata_output,
            "confidence_output": confidence_output if confidence_result else None,
            "backtest_output": backtest_output if backtest_result else None,
            "validation_output": validation_output,
            "onnx_output": onnx_output,
            "mt5_features_output": mt5_features_output,
            "mt5_config_output": mt5_config_output,
            "report_output": report_output,
        },
        "notes": [
            "This live path is experimental.",
            "Twelve Data XAU/USD bars do not include volume, so the pipeline synthesized volume before feature engineering."
            if fetch_result["synthesized_volume"]
            else "Source volume was available from the data feed.",
            "Training, confidence analysis, and backtest outputs come from the recent fetched window only, not from the full historical dataset originally envisioned by the project.",
        ],
    }
    json_dump(report, report_output)
    return report


def print_live_summary(report: dict) -> None:
    print("Generated live artifacts:")
    print(f"  source rows       : {report['source_rows']:,}")
    print(f"  overlap rows      : {report['overlap_rows']:,}")
    print(f"  feature rows      : {report['feature_rows']:,}")
    print(f"  labeled rows      : {report['labeled_rows']:,}")
    print(f"  validation rows   : {report['validation_rows']:,}")
    print()
    print("Current live prediction:")
    print(f"  time              : {report['prediction']['time']}")
    print(f"  label             : {report['prediction']['label']}")
    print(
        "  probabilities     : "
        f"SHORT={report['prediction']['prob_short']:.2%}, "
        f"HOLD={report['prediction']['prob_hold']:.2%}, "
        f"LONG={report['prediction']['prob_long']:.2%}"
    )
    print()
    print("Notes:")
    for note in report["notes"]:
        print(f"  - {note}")
    print()
    print(f"Live report         : {display_path(report['artifacts']['report_output'])}")


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("XAUUSD TWELVE DATA LIVE PIPELINE")
    print("=" * 70)
    print("This run fetches recent Twelve Data bars, synthesizes volume if needed, and executes the live pipeline.")
    print()

    report = run_live_pipeline(
        api_env=args.api_env,
        symbol=args.symbol,
        interval=args.interval,
        outputsize=args.outputsize,
        volume_mode=args.volume_mode,
        feature_config=args.feature_config,
        model_config_path=args.model_config,
        raw_output=args.raw_output,
        standardized_output=args.standardized_output,
        overlap_output=args.overlap_output,
        feature_output=args.feature_output,
        label_output=args.label_output,
        model_output=args.model_output,
        feature_list_output=args.feature_list_output,
        metadata_output=args.metadata_output,
        confidence_output=args.confidence_output,
        backtest_output=args.backtest_output,
        validation_output=args.validation_output,
        onnx_output=args.onnx_output,
        mt5_features_output=args.mt5_features_output,
        mt5_config_output=args.mt5_config_output,
        report_output=args.report_output,
        skip_onnx_runtime_check=args.skip_onnx_runtime_check,
        skip_confidence_analysis=args.skip_confidence_analysis,
        skip_backtest=args.skip_backtest,
    )
    print_live_summary(report)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
