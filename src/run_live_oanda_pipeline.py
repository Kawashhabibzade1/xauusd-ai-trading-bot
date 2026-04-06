"""
Fetch recent XAU_USD bars from OANDA and run the live pipeline end to end.

This path is preferred over Twelve Data when you want a live candle feed that
includes OANDA's candle volume field.
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
from oanda_client import fetch_and_write_candles_csv, normalize_oanda_instrument, resolve_api_token, resolve_rest_url
from pipeline_contract import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_OANDA_LIVE_BACKTEST_RESULTS_PATH,
    DEFAULT_OANDA_LIVE_CONFIDENCE_RESULTS_PATH,
    DEFAULT_OANDA_LIVE_FEATURE_OUTPUT,
    DEFAULT_OANDA_LIVE_FEATURE_LIST_PATH,
    DEFAULT_OANDA_LIVE_LABEL_OUTPUT,
    DEFAULT_OANDA_LIVE_METADATA_PATH,
    DEFAULT_OANDA_LIVE_MODEL_PATH,
    DEFAULT_OANDA_LIVE_MT5_FEATURES_PATH,
    DEFAULT_OANDA_LIVE_MT5_MODEL_CONFIG_PATH,
    DEFAULT_OANDA_LIVE_MT5_ONNX_OUTPUT,
    DEFAULT_OANDA_LIVE_MT5_VALIDATION_OUTPUT,
    DEFAULT_OANDA_LIVE_OVERLAP_OUTPUT,
    DEFAULT_OANDA_LIVE_RAW_INPUT,
    DEFAULT_OANDA_LIVE_REPORT_PATH,
    DEFAULT_OANDA_LIVE_STANDARDIZED_OUTPUT,
    MODEL_CLASS_MAPPING,
    display_path,
    ensure_parent_dir,
    json_dump,
)
from run_sample_demo import verify_dependencies
from train_lightgbm import train_model_from_labeled
from validate_merged_data import load_and_standardize, validate_standardized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-env", default="OANDA_API_TOKEN", help="Environment variable containing the OANDA API token.")
    parser.add_argument("--api-url-env", default="OANDA_API_URL", help="Optional environment variable overriding the OANDA REST base URL.")
    parser.add_argument("--mode-env", default="OANDA_ENV", help="Environment variable selecting OANDA practice/live mode.")
    parser.add_argument("--instrument", default="XAU_USD", help="OANDA instrument, for example XAU_USD.")
    parser.add_argument("--granularity", default="M1", help="OANDA candle granularity.")
    parser.add_argument("--count", type=int, default=5000, help="Number of recent OANDA candles to request.")
    parser.add_argument("--price", choices=("M", "B", "A"), default="M", help="OANDA price component to request.")
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="Feature contract YAML path.")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="Training config YAML path.")
    parser.add_argument("--raw-output", default=DEFAULT_OANDA_LIVE_RAW_INPUT, help="Fetched raw OHLCV CSV output.")
    parser.add_argument("--standardized-output", default=DEFAULT_OANDA_LIVE_STANDARDIZED_OUTPUT, help="Standardized CSV output.")
    parser.add_argument("--overlap-output", default=DEFAULT_OANDA_LIVE_OVERLAP_OUTPUT, help="Filtered overlap CSV output.")
    parser.add_argument("--feature-output", default=DEFAULT_OANDA_LIVE_FEATURE_OUTPUT, help="Engineered features CSV output.")
    parser.add_argument("--label-output", default=DEFAULT_OANDA_LIVE_LABEL_OUTPUT, help="Labeled live dataset CSV output.")
    parser.add_argument("--model-output", default=DEFAULT_OANDA_LIVE_MODEL_PATH, help="Live LightGBM model output path.")
    parser.add_argument("--feature-list-output", default=DEFAULT_OANDA_LIVE_FEATURE_LIST_PATH, help="Live feature list JSON output.")
    parser.add_argument("--metadata-output", default=DEFAULT_OANDA_LIVE_METADATA_PATH, help="Live model metadata JSON output.")
    parser.add_argument("--confidence-output", default=DEFAULT_OANDA_LIVE_CONFIDENCE_RESULTS_PATH, help="Confidence analysis JSON output.")
    parser.add_argument("--backtest-output", default=DEFAULT_OANDA_LIVE_BACKTEST_RESULTS_PATH, help="Approximate live backtest JSON output.")
    parser.add_argument("--validation-output", default=DEFAULT_OANDA_LIVE_MT5_VALIDATION_OUTPUT, help="Live MT5 validation fixture CSV output.")
    parser.add_argument("--onnx-output", default=DEFAULT_OANDA_LIVE_MT5_ONNX_OUTPUT, help="Live ONNX model output path.")
    parser.add_argument("--mt5-features-output", default=DEFAULT_OANDA_LIVE_MT5_FEATURES_PATH, help="MT5 live feature-list JSON output.")
    parser.add_argument("--mt5-config-output", default=DEFAULT_OANDA_LIVE_MT5_MODEL_CONFIG_PATH, help="MT5 live model-config JSON output.")
    parser.add_argument("--report-output", default=DEFAULT_OANDA_LIVE_REPORT_PATH, help="Live pipeline report JSON output.")
    parser.add_argument("--skip-onnx-runtime-check", action="store_true", help="Skip optional onnxruntime verification.")
    parser.add_argument("--skip-confidence-analysis", action="store_true", help="Skip confidence threshold analysis.")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip approximate backtest generation.")
    return parser.parse_args()


def run_live_pipeline(
    api_env: str = "OANDA_API_TOKEN",
    api_url_env: str = "OANDA_API_URL",
    mode_env: str = "OANDA_ENV",
    instrument: str = "XAU_USD",
    granularity: str = "M1",
    count: int = 5000,
    price: str = "M",
    feature_config: str = DEFAULT_FEATURE_CONFIG,
    model_config_path: str = DEFAULT_MODEL_CONFIG,
    raw_output: str = DEFAULT_OANDA_LIVE_RAW_INPUT,
    standardized_output: str = DEFAULT_OANDA_LIVE_STANDARDIZED_OUTPUT,
    overlap_output: str = DEFAULT_OANDA_LIVE_OVERLAP_OUTPUT,
    feature_output: str = DEFAULT_OANDA_LIVE_FEATURE_OUTPUT,
    label_output: str = DEFAULT_OANDA_LIVE_LABEL_OUTPUT,
    model_output: str = DEFAULT_OANDA_LIVE_MODEL_PATH,
    feature_list_output: str = DEFAULT_OANDA_LIVE_FEATURE_LIST_PATH,
    metadata_output: str = DEFAULT_OANDA_LIVE_METADATA_PATH,
    confidence_output: str = DEFAULT_OANDA_LIVE_CONFIDENCE_RESULTS_PATH,
    backtest_output: str = DEFAULT_OANDA_LIVE_BACKTEST_RESULTS_PATH,
    validation_output: str = DEFAULT_OANDA_LIVE_MT5_VALIDATION_OUTPUT,
    onnx_output: str = DEFAULT_OANDA_LIVE_MT5_ONNX_OUTPUT,
    mt5_features_output: str = DEFAULT_OANDA_LIVE_MT5_FEATURES_PATH,
    mt5_config_output: str = DEFAULT_OANDA_LIVE_MT5_MODEL_CONFIG_PATH,
    report_output: str = DEFAULT_OANDA_LIVE_REPORT_PATH,
    skip_onnx_runtime_check: bool = False,
    skip_confidence_analysis: bool = False,
    skip_backtest: bool = False,
) -> dict:
    verify_dependencies(skip_onnx_runtime_check)

    api_token = resolve_api_token(api_env)
    if not api_token:
        raise SystemExit(
            f"OANDA API token not found in environment variable {api_env}. "
            "Set it in your shell or .env before running the live OANDA pipeline."
        )

    fetch_result = fetch_and_write_candles_csv(
        api_token=api_token,
        output_path=raw_output,
        instrument=instrument,
        granularity=granularity,
        count=count,
        price=price,
        rest_url=resolve_rest_url(api_url_env, mode_env),
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
            "Live feature generation produced 0 rows. Increase the OANDA history window or adjust the live pipeline inputs."
        )
    feature_output_path = ensure_parent_dir(feature_output)
    features.to_csv(feature_output_path, index=False)

    labeled = create_labeled_frame(features, feature_config=feature_config)
    if labeled.empty:
        raise SystemExit(
            "Live label generation produced 0 rows. The fetched OANDA history window is too small for the current feature and label contract."
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
        "mode": "live_oanda",
        "symbol": fetch_result["symbol"],
        "instrument": fetch_result["instrument"],
        "interval": fetch_result["interval"],
        "source_timezone": fetch_result["timezone"],
        "source_rows": fetch_result["rows"],
        "source_start": fetch_result["start"],
        "source_end": fetch_result["end"],
        "source_has_volume": fetch_result["has_source_volume"],
        "volume_mode": fetch_result["volume_mode"],
        "volume_note": fetch_result["volume_note"],
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
            "raw_output": display_path(fetch_result["output_path"]),
            "standardized_output": display_path(standardized_output_path),
            "overlap_output": display_path(overlap_output_path),
            "feature_output": display_path(feature_output_path),
            "label_output": display_path(label_output_path),
            "model_output": display_path(model_output),
            "feature_list_output": display_path(feature_list_output),
            "metadata_output": display_path(metadata_output),
            "validation_output": display_path(validation_output),
            "onnx_output": display_path(onnx_output),
            "report_output": display_path(report_output),
        },
        "notes": [
            fetch_result["volume_note"],
            "Training, confidence analysis, and backtest outputs come from the recent OANDA fetch window only, not a full broker-history regime study.",
        ],
    }
    json_dump(report, report_output)
    return report


def print_summary(report: dict) -> None:
    print("OANDA live pipeline artifacts generated:")
    print(f"  symbol           : {report['symbol']}")
    print(f"  source rows      : {report['source_rows']}")
    print(f"  source range     : {report['source_start']} -> {report['source_end']}")
    print(f"  overlap rows     : {report['overlap_rows']}")
    print(f"  feature rows     : {report['feature_rows']}")
    print(f"  labeled rows     : {report['labeled_rows']}")
    print(f"  validation rows  : {report['validation_rows']}")
    print(f"  latest prediction: {report['prediction']['label']} @ {report['prediction']['time']}")
    print(f"  runtime status   : {report['runtime_status']}")
    print()
    print("Key outputs:")
    for key, value in report["artifacts"].items():
        print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("XAUUSD OANDA LIVE PIPELINE")
    print("=" * 70)
    print(f"Instrument : {normalize_oanda_instrument(args.instrument)}")
    print(f"Granularity: {args.granularity}")
    print(f"Count      : {args.count}")
    print(f"Skip neural: n/a in this path")
    print()

    report = run_live_pipeline(
        api_env=args.api_env,
        api_url_env=args.api_url_env,
        mode_env=args.mode_env,
        instrument=args.instrument,
        granularity=args.granularity,
        count=args.count,
        price=args.price,
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
    print_summary(report)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
