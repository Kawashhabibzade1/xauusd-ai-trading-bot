"""
Fetch recent XAUUSD bars from a local MetaTrader 5 terminal and run the live pipeline end to end.

This path is local-only and is preferred when you want live bars with MT5 tick_volume.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from analyze_confidence import analyze_confidence_from_labeled
from backtest_simple import run_signal_simulation
from create_labels import create_labeled_frame
from export_mt5_validation_set import export_validation_fixture
from export_to_onnx_simple import export_model_to_onnx
from feature_engineering import compute_feature_frame
from filter_overlap import filter_hunt_windows_frame, normalize_hunt_windows
from mt5_client import (
    copy_exported_rates_csv,
    fetch_and_write_rates_csv,
    normalize_mt5_symbol,
    resolve_export_file_path,
    resolve_login,
    resolve_password,
    resolve_server,
    resolve_terminal_path,
    sync_mt5_file_artifacts,
)
from pipeline_contract import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_MT5_LIVE_BACKTEST_RESULTS_PATH,
    DEFAULT_MT5_LIVE_CONFIDENCE_RESULTS_PATH,
    DEFAULT_MT5_LIVE_FEATURE_OUTPUT,
    DEFAULT_MT5_LIVE_FEATURE_LIST_PATH,
    DEFAULT_MT5_LIVE_LABEL_OUTPUT,
    DEFAULT_MT5_LIVE_METADATA_PATH,
    DEFAULT_MT5_LIVE_MODEL_PATH,
    DEFAULT_MT5_LIVE_MT5_FEATURES_PATH,
    DEFAULT_MT5_LIVE_MT5_MODEL_CONFIG_PATH,
    DEFAULT_MT5_LIVE_MT5_ONNX_OUTPUT,
    DEFAULT_MT5_LIVE_MT5_VALIDATION_OUTPUT,
    DEFAULT_MT5_LIVE_OVERLAP_OUTPUT,
    DEFAULT_MT5_LIVE_RAW_INPUT,
    DEFAULT_MT5_LIVE_REPORT_PATH,
    DEFAULT_MT5_LIVE_STANDARDIZED_OUTPUT,
    MODEL_CLASS_MAPPING,
    display_path,
    ensure_parent_dir,
    json_dump,
)
from run_sample_demo import verify_dependencies
from train_lightgbm import train_model_from_labeled
from validate_merged_data import load_and_standardize, validate_standardized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD", help="MT5 symbol, for example XAUUSD.")
    parser.add_argument("--timeframe", default="M1", help="MT5 timeframe, for example M1.")
    parser.add_argument("--count", type=int, default=5000, help="Number of recent MT5 bars to request.")
    parser.add_argument(
        "--source-mode",
        choices=["auto", "bridge", "export"],
        default="auto",
        help="Use direct MT5 bridge, exporter CSV, or auto-try bridge then exporter CSV.",
    )
    parser.add_argument("--terminal-env", default="MT5_TERMINAL_PATH", help="Environment variable with the MT5 terminal path.")
    parser.add_argument("--login-env", default="MT5_LOGIN", help="Optional environment variable with the MT5 login.")
    parser.add_argument("--password-env", default="MT5_PASSWORD", help="Optional environment variable with the MT5 password.")
    parser.add_argument("--server-env", default="MT5_SERVER", help="Optional environment variable with the MT5 server.")
    parser.add_argument(
        "--export-input",
        default="",
        help="Optional MT5 exporter CSV path. If omitted, the local MQL5/Files/xauusd_mt5_live.csv path is used.",
    )
    parser.add_argument("--prefer-real-volume", action="store_true", help="Use real_volume when available; otherwise tick_volume is used.")
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="Feature contract YAML path.")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="Training config YAML path.")
    parser.add_argument("--raw-output", default=DEFAULT_MT5_LIVE_RAW_INPUT, help="Fetched raw OHLCV CSV output.")
    parser.add_argument("--standardized-output", default=DEFAULT_MT5_LIVE_STANDARDIZED_OUTPUT, help="Standardized CSV output.")
    parser.add_argument("--overlap-output", default=DEFAULT_MT5_LIVE_OVERLAP_OUTPUT, help="Filtered overlap CSV output.")
    parser.add_argument("--feature-output", default=DEFAULT_MT5_LIVE_FEATURE_OUTPUT, help="Engineered features CSV output.")
    parser.add_argument("--label-output", default=DEFAULT_MT5_LIVE_LABEL_OUTPUT, help="Labeled live dataset CSV output.")
    parser.add_argument("--model-output", default=DEFAULT_MT5_LIVE_MODEL_PATH, help="Live LightGBM model output path.")
    parser.add_argument("--feature-list-output", default=DEFAULT_MT5_LIVE_FEATURE_LIST_PATH, help="Live feature list JSON output.")
    parser.add_argument("--metadata-output", default=DEFAULT_MT5_LIVE_METADATA_PATH, help="Live model metadata JSON output.")
    parser.add_argument("--confidence-output", default=DEFAULT_MT5_LIVE_CONFIDENCE_RESULTS_PATH, help="Confidence analysis JSON output.")
    parser.add_argument("--backtest-output", default=DEFAULT_MT5_LIVE_BACKTEST_RESULTS_PATH, help="Approximate live backtest JSON output.")
    parser.add_argument("--validation-output", default=DEFAULT_MT5_LIVE_MT5_VALIDATION_OUTPUT, help="Live MT5 validation fixture CSV output.")
    parser.add_argument("--onnx-output", default=DEFAULT_MT5_LIVE_MT5_ONNX_OUTPUT, help="Live ONNX model output path.")
    parser.add_argument("--mt5-features-output", default=DEFAULT_MT5_LIVE_MT5_FEATURES_PATH, help="MT5 live feature-list JSON output.")
    parser.add_argument("--mt5-config-output", default=DEFAULT_MT5_LIVE_MT5_MODEL_CONFIG_PATH, help="MT5 live model-config JSON output.")
    parser.add_argument("--report-output", default=DEFAULT_MT5_LIVE_REPORT_PATH, help="Live pipeline report JSON output.")
    parser.add_argument("--skip-onnx-runtime-check", action="store_true", help="Skip optional onnxruntime verification.")
    parser.add_argument("--skip-confidence-analysis", action="store_true", help="Skip confidence threshold analysis.")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip approximate backtest generation.")
    return parser.parse_args(argv)


def run_live_pipeline(
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    count: int = 5000,
    source_mode: str = "auto",
    terminal_env: str = "MT5_TERMINAL_PATH",
    login_env: str = "MT5_LOGIN",
    password_env: str = "MT5_PASSWORD",
    server_env: str = "MT5_SERVER",
    export_input: str = "",
    prefer_real_volume: bool = False,
    feature_config: str = DEFAULT_FEATURE_CONFIG,
    model_config_path: str = DEFAULT_MODEL_CONFIG,
    raw_output: str = DEFAULT_MT5_LIVE_RAW_INPUT,
    standardized_output: str = DEFAULT_MT5_LIVE_STANDARDIZED_OUTPUT,
    overlap_output: str = DEFAULT_MT5_LIVE_OVERLAP_OUTPUT,
    feature_output: str = DEFAULT_MT5_LIVE_FEATURE_OUTPUT,
    label_output: str = DEFAULT_MT5_LIVE_LABEL_OUTPUT,
    model_output: str = DEFAULT_MT5_LIVE_MODEL_PATH,
    feature_list_output: str = DEFAULT_MT5_LIVE_FEATURE_LIST_PATH,
    metadata_output: str = DEFAULT_MT5_LIVE_METADATA_PATH,
    confidence_output: str = DEFAULT_MT5_LIVE_CONFIDENCE_RESULTS_PATH,
    backtest_output: str = DEFAULT_MT5_LIVE_BACKTEST_RESULTS_PATH,
    validation_output: str = DEFAULT_MT5_LIVE_MT5_VALIDATION_OUTPUT,
    onnx_output: str = DEFAULT_MT5_LIVE_MT5_ONNX_OUTPUT,
    mt5_features_output: str = DEFAULT_MT5_LIVE_MT5_FEATURES_PATH,
    mt5_config_output: str = DEFAULT_MT5_LIVE_MT5_MODEL_CONFIG_PATH,
    report_output: str = DEFAULT_MT5_LIVE_REPORT_PATH,
    hunt_timezone: str = "UTC",
    hunt_windows: list[dict[str, Any]] | None = None,
    skip_onnx_runtime_check: bool = False,
    skip_confidence_analysis: bool = False,
    skip_backtest: bool = False,
) -> dict:
    verify_dependencies(skip_onnx_runtime_check)

    bridge_error: Exception | None = None
    fetch_result: dict | None = None

    if source_mode in {"auto", "bridge"}:
        try:
            fetch_result = fetch_and_write_rates_csv(
                output_path=raw_output,
                symbol=symbol,
                timeframe=timeframe,
                count=count,
                terminal_path=resolve_terminal_path(terminal_env),
                login=resolve_login(login_env),
                password=resolve_password(password_env),
                server=resolve_server(server_env),
                prefer_real_volume=prefer_real_volume,
            )
            fetch_result["provider"] = "mt5_local"
        except Exception as exc:
            bridge_error = exc
            if source_mode == "bridge":
                raise

    if fetch_result is None and source_mode in {"auto", "export"}:
        try:
            fetch_result = copy_exported_rates_csv(
                output_path=raw_output,
                input_path=resolve_export_file_path(export_input or None),
                symbol=symbol,
                timeframe=timeframe,
            )
        except Exception as exc:
            if bridge_error is not None:
                raise SystemExit(
                    "MT5 direct bridge failed and MT5 exporter fallback also failed.\n"
                    f"Bridge error: {bridge_error}\n"
                    f"Export error: {exc}"
                ) from exc
            raise

    if fetch_result is None:
        raise SystemExit("MT5 live pipeline could not fetch data from either the MT5 bridge or the exporter CSV.")

    standardized = load_and_standardize(raw_output)
    standardized_stats = validate_standardized(standardized)
    standardized_output_path = ensure_parent_dir(standardized_output)
    standardized.to_csv(standardized_output_path, index=False)

    active_hunt_windows = normalize_hunt_windows(hunt_windows) if hunt_windows else normalize_hunt_windows(
        [{"name": "Overlap", "start": "13:00", "end": "16:59", "max_trades": 0}]
    )
    active_hunt_timezone = str(hunt_timezone or "UTC")
    overlap = filter_hunt_windows_frame(standardized, timezone_name=active_hunt_timezone, windows=active_hunt_windows)
    overlap_output_path = ensure_parent_dir(overlap_output)
    overlap.to_csv(overlap_output_path, index=False)

    features = compute_feature_frame(overlap, feature_config)
    if features.empty:
        raise SystemExit(
            "Live feature generation produced 0 rows. Increase the MT5 history window or adjust the live pipeline inputs."
        )
    feature_output_path = ensure_parent_dir(feature_output)
    features.to_csv(feature_output_path, index=False)

    labeled = create_labeled_frame(features, feature_config=feature_config)
    if labeled.empty:
        raise SystemExit(
            "Live label generation produced 0 rows. The fetched MT5 history window is too small for the current feature and label contract."
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
    synced_mt5_files = []
    sync_error = None
    try:
        synced_mt5_files = sync_mt5_file_artifacts(
            [
                validation_output,
                onnx_output,
                mt5_features_output,
                mt5_config_output,
            ]
        )
    except Exception as exc:
        sync_error = str(exc)

    validation_df = pd.read_csv(fixture_result["output_path"])
    latest_prediction = validation_df.iloc[-1].to_dict()
    predicted_class = int(latest_prediction["expected_class"])

    report = {
        "mode": "live_mt5",
        "source_provider": fetch_result.get("provider", "mt5_local"),
        "symbol": fetch_result["symbol"],
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
        "hunt_timezone": active_hunt_timezone,
        "hunt_windows": active_hunt_windows,
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
            "mt5_file_sync": [item["target"] for item in synced_mt5_files],
        },
        "notes": [
            fetch_result["volume_note"],
            (
                f"MT5 exporter source file: {fetch_result['source_path']}"
                if fetch_result.get("source_path")
                else "MT5 direct bridge connected to the local terminal."
            ),
            "This MT5 path is local-only and requires a running MetaTrader terminal on the same machine.",
            "Training, confidence analysis, and backtest outputs come from the recent MT5 fetch window only, not a full historical regime study.",
            f"Active hunt timezone: {active_hunt_timezone}.",
            (
                f"Synced {len(synced_mt5_files)} MT5 artifact files into the local MQL5/Files directory."
                if synced_mt5_files
                else f"MT5 artifact sync skipped: {sync_error}"
                if sync_error
                else "MT5 artifact sync was not needed."
            ),
        ],
    }
    json_dump(report, report_output)
    return report


def run_from_args(args: argparse.Namespace) -> dict:
    return run_live_pipeline(
        symbol=args.symbol,
        timeframe=args.timeframe,
        count=args.count,
        source_mode=args.source_mode,
        terminal_env=args.terminal_env,
        login_env=args.login_env,
        password_env=args.password_env,
        server_env=args.server_env,
        export_input=args.export_input,
        prefer_real_volume=args.prefer_real_volume,
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
        hunt_timezone="UTC",
        hunt_windows=None,
        skip_onnx_runtime_check=args.skip_onnx_runtime_check,
        skip_confidence_analysis=args.skip_confidence_analysis,
        skip_backtest=args.skip_backtest,
    )


def print_summary(report: dict) -> None:
    print("MT5 live pipeline artifacts generated:")
    print(f"  source provider  : {report['source_provider']}")
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print("=" * 70)
    print("XAUUSD MT5 LIVE PIPELINE")
    print("=" * 70)
    print(f"Symbol     : {normalize_mt5_symbol(args.symbol)}")
    print(f"Timeframe  : {args.timeframe}")
    print(f"Count      : {args.count}")
    print(f"Source     : {args.source_mode}")
    print(f"Real volume: {args.prefer_real_volume}")
    print()

    report = run_from_args(args)
    print_summary(report)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
