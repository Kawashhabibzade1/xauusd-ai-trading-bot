"""
Serve a simple local dashboard for the sample demo or the live Twelve Data pipeline.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from pipeline_contract import DEFAULT_LIVE_REPORT_PATH, display_path, resolve_repo_path
from run_live_twelvedata_pipeline import run_live_pipeline
from run_sample_demo import DEFAULT_INPUT, DEFAULT_MODEL, run_demo_pipeline
from twelvedata_client import fetch_time_series, resolve_api_key


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "ui" / "demo"
FEATURE_SNAPSHOT_KEYS = [
    "close",
    "atr_14",
    "rsi_14",
    "macd",
    "bb_position",
    "volume_ratio",
    "h4_bias",
    "smc_quality_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=4174, help="Local port for the dashboard.")
    parser.add_argument(
        "--pipeline-mode",
        choices=("auto", "sample", "live"),
        default="auto",
        help="Which pipeline to serve. 'auto' uses live mode when a Twelve Data API key is available.",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Sample/demo OHLCV CSV input.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Saved LightGBM model used for the sample demo.")
    parser.add_argument(
        "--skip-onnx-runtime-check",
        action="store_true",
        help="Skip the optional onnxruntime inference verification while refreshing artifacts.",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Serve the UI using existing generated artifacts without rerunning the selected pipeline first.",
    )
    parser.add_argument(
        "--twelvedata-env",
        default="TWELVEDATA_API_KEY",
        help="Environment variable that stores the Twelve Data API key.",
    )
    parser.add_argument("--twelvedata-symbol", default="XAU/USD", help="Symbol used for the live Twelve Data market snapshot.")
    parser.add_argument("--twelvedata-outputsize", type=int, default=20, help="Number of recent Twelve Data bars to show in the live market preview.")
    parser.add_argument(
        "--twelvedata-pipeline-outputsize",
        type=int,
        default=5000,
        help="Number of recent Twelve Data bars to fetch for the live pipeline.",
    )
    parser.add_argument(
        "--twelvedata-volume-mode",
        choices=("constant", "range_proxy"),
        default="range_proxy",
        help="Experimental fallback volume mode when the Twelve Data feed has no source volume.",
    )
    return parser.parse_args()


def human_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def preview_frame(path: Path, rows: int = 5) -> dict:
    frame = pd.read_csv(path)
    if "time" in frame.columns:
        frame["time"] = frame["time"].astype(str)
    head = frame.head(rows)
    return {
        "row_count": int(len(frame)),
        "columns": frame.columns.tolist(),
        "rows": head.to_dict(orient="records"),
    }


def build_live_market_payload(time_series: dict | None, error: str | None = None) -> dict | None:
    if error:
        return {
            "enabled": False,
            "error": error,
        }

    if time_series is None:
        return None

    values = time_series["values"]
    latest = values[-1]
    previous = values[-2] if len(values) > 1 else latest
    change = latest["close"] - previous["close"]
    change_pct = (change / previous["close"] * 100.0) if previous["close"] else 0.0
    high_lookback = max(item["high"] for item in values)
    low_lookback = min(item["low"] for item in values)
    meta = time_series["meta"]

    preview_rows = []
    for item in values[-10:]:
        preview_rows.append(
            {
                "datetime": item["datetime"],
                "open": item["open"],
                "high": item["high"],
                "low": item["low"],
                "close": item["close"],
            }
        )

    return {
        "enabled": True,
        "symbol": meta.get("symbol", "XAU/USD"),
        "type": meta.get("type", "Unknown"),
        "last_time": latest["datetime"],
        "last_close": latest["close"],
        "change": change,
        "change_pct": change_pct,
        "high_lookback": high_lookback,
        "low_lookback": low_lookback,
        "interval": meta.get("interval", "1min"),
        "has_volume": time_series["has_volume"],
        "note": (
            "Twelve Data XAU/USD bars do not include volume, so live prediction mode uses an explicit experimental volume fallback."
            if not time_series["has_volume"]
            else "Live market bars are available and include source volume."
        ),
        "preview": {
            "row_count": len(values),
            "columns": ["datetime", "open", "high", "low", "close"],
            "rows": preview_rows,
        },
    }


def build_feature_snapshot(path: Path) -> dict:
    features_df = pd.read_csv(path)
    snapshot = features_df.iloc[-1].to_dict()
    snapshot["time"] = str(snapshot["time"])
    return {
        "time": snapshot["time"],
        "values": [{"label": key, "value": snapshot[key]} for key in FEATURE_SNAPSHOT_KEYS],
    }


def build_artifact_list(path_map: list[tuple[str, str | None]]) -> list[dict]:
    artifacts = []
    for label, path_like in path_map:
        if not path_like:
            continue
        path = resolve_repo_path(path_like)
        if not path.exists():
            continue
        artifacts.append(
            {
                "label": label,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size": human_size(path.stat().st_size),
            }
        )
    return artifacts


def load_existing_sample_result() -> dict:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    standardized = pd.read_csv(processed_dir / "xauusd_m1_standardized.csv")
    overlap = pd.read_csv(processed_dir / "xauusd_m1_overlap.csv")
    features = pd.read_csv(processed_dir / "xauusd_features.csv")
    validation = pd.read_csv(PROJECT_ROOT / "mt5_expert_advisor" / "Files" / "config" / "validation_set.csv")
    return {
        "standardized_rows": int(len(standardized)),
        "standardized_start": str(standardized["time"].min()),
        "standardized_end": str(standardized["time"].max()),
        "overlap_rows": int(len(overlap)),
        "overlap_start": str(overlap["time"].min()),
        "overlap_end": str(overlap["time"].max()),
        "feature_rows": int(len(features)),
        "feature_time": str(features["time"].iloc[-1]),
        "validation_rows": int(len(validation)),
        "runtime_status": "not refreshed",
    }


def build_sample_dashboard_payload(demo_result: dict, live_market: dict | None) -> dict:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    config_dir = PROJECT_ROOT / "mt5_expert_advisor" / "Files" / "config"
    models_dir = PROJECT_ROOT / "mt5_expert_advisor" / "Files" / "models"

    feature_snapshot = build_feature_snapshot(processed_dir / "xauusd_features.csv")
    validation_df = pd.read_csv(config_dir / "validation_set.csv")
    model_config = json.loads((config_dir / "model_config.json").read_text(encoding="utf-8"))
    validation_row = validation_df.iloc[-1].to_dict()
    predicted_label = model_config["class_mapping"][str(int(validation_row["expected_class"]))]

    return {
        "title": "XAUUSD Sample Demo Dashboard",
        "summary": {
            "metrics": [
                ["Standardized Bars", demo_result["standardized_rows"]],
                ["Overlap Bars", demo_result["overlap_rows"]],
                ["Feature Rows", demo_result["feature_rows"]],
                ["Validation Rows", demo_result["validation_rows"]],
            ],
            "range_text": f"Source range: {demo_result['standardized_start']} to {demo_result['standardized_end']} | Overlap: {demo_result['overlap_start']} to {demo_result['overlap_end']}",
            "runtime_badge": f"Sample Mode | ONNX Runtime: {demo_result['runtime_status']}",
        },
        "live_market": live_market,
        "context": {
            "prediction_label": "Sample Demo Prediction",
            "prediction_note": (
                "This prediction comes from the checked-in sample demo artifact dated "
                f"{feature_snapshot['time']}, not from the live Twelve Data feed."
            ),
            "prediction_time_label": f"Sample artifact time: {feature_snapshot['time']}",
            "feature_note": "The most recent engineered bar from the checked-in sample demo.",
        },
        "model": {
            "file": model_config["model_info"]["file"],
            "num_features": model_config["model_info"]["num_features"],
            "num_trees": model_config["model_info"]["num_trees"],
            "confidence_threshold": model_config["trading_config"]["confidence_threshold"],
            "onnx_size": human_size((models_dir / "xauusd_ai_v1.onnx").stat().st_size),
        },
        "prediction": {
            "label": predicted_label,
            "probabilities": [
                {"label": "SHORT", "value": float(validation_row["prob_short"])},
                {"label": "HOLD", "value": float(validation_row["prob_hold"])},
                {"label": "LONG", "value": float(validation_row["prob_long"])},
            ],
        },
        "feature_snapshot": feature_snapshot,
        "artifacts": build_artifact_list(
            [
                ("Standardized CSV", "data/processed/xauusd_m1_standardized.csv"),
                ("Overlap CSV", "data/processed/xauusd_m1_overlap.csv"),
                ("Feature CSV", "data/processed/xauusd_features.csv"),
                ("Validation Fixture", "mt5_expert_advisor/Files/config/validation_set.csv"),
                ("ONNX Model", "mt5_expert_advisor/Files/models/xauusd_ai_v1.onnx"),
            ]
        ),
        "previews": {
            "standardized": preview_frame(processed_dir / "xauusd_m1_standardized.csv", rows=5),
            "overlap": preview_frame(processed_dir / "xauusd_m1_overlap.csv", rows=5),
            "validation": preview_frame(config_dir / "validation_set.csv", rows=1),
        },
    }


def build_live_dashboard_payload(report: dict, live_market: dict | None) -> dict:
    artifacts = report["artifacts"]
    feature_output = resolve_repo_path(artifacts["feature_output"])
    validation_output = resolve_repo_path(artifacts["validation_output"])
    standardized_output = resolve_repo_path(artifacts["standardized_output"])
    overlap_output = resolve_repo_path(artifacts["overlap_output"])
    label_output = resolve_repo_path(artifacts["label_output"])
    model_config_output = resolve_repo_path(artifacts["mt5_config_output"])
    onnx_output = resolve_repo_path(artifacts["onnx_output"])

    feature_snapshot = build_feature_snapshot(feature_output)
    validation_df = pd.read_csv(validation_output)
    labeled_df = pd.read_csv(label_output)
    model_config = json.loads(model_config_output.read_text(encoding="utf-8"))
    validation_row = validation_df.iloc[-1].to_dict()
    split_idx = int(0.8 * len(labeled_df))
    test_label_dist = labeled_df.iloc[split_idx:]["label"].value_counts().sort_index().to_dict()
    observed_test_labels = [label for label, count in test_label_dist.items() if count > 0]

    label_name_map = {-1: "SHORT", 0: "HOLD", 1: "LONG"}
    if len(observed_test_labels) == 1:
        single_label_name = label_name_map.get(int(observed_test_labels[0]), str(observed_test_labels[0]))
        accuracy_note = (
            f"The held-out split currently contains only {single_label_name} labels, "
            "so the displayed accuracy is not a strong trading-quality validation signal."
        )
    else:
        accuracy_note = "The held-out split contains multiple label classes."

    metrics = [
        ["Fetched Bars", report["source_rows"]],
        ["Overlap Bars", report["overlap_rows"]],
        ["Feature Rows", report["feature_rows"]],
        ["Labeled Rows", report["labeled_rows"]],
        ["Validation Rows", report["validation_rows"]],
        ["Held-Out Accuracy", f"{report['training']['accuracy']:.2%}"],
    ]
    if report.get("confidence"):
        metrics.append(["Selected Threshold", f"{report['confidence']['selected_threshold']:.0%}"])
    if report.get("backtest"):
        metrics.append(["Approx Return", f"{report['backtest']['performance']['return_percent']:.1f}%"])

    volume_note = (
        f"Live prediction uses Twelve Data bars with synthesized volume mode '{report['volume_mode']}'."
        if report["synthesized_volume"]
        else "Live prediction uses source-provided volume from the feed."
    )

    return {
        "title": "XAUUSD Live Twelve Data Dashboard",
        "summary": {
            "metrics": metrics,
            "range_text": f"Source range: {report['source_start']} to {report['source_end']} | Overlap: {report['overlap_start']} to {report['overlap_end']}",
            "runtime_badge": f"Live Mode | ONNX Runtime: {report['runtime_status']}",
        },
        "live_market": live_market,
        "context": {
            "prediction_label": "Live Experimental Prediction",
            "prediction_note": (
                f"{volume_note} {accuracy_note} This run retrains and scores on the recent Twelve Data fetch window only, so treat it as experimental."
            ),
            "prediction_time_label": f"Latest live prediction bar: {report['prediction']['time']}",
            "feature_note": "The most recent engineered bar from the live Twelve Data pipeline.",
        },
        "model": {
            "file": model_config["model_info"]["file"],
            "num_features": model_config["model_info"]["num_features"],
            "num_trees": model_config["model_info"]["num_trees"],
            "confidence_threshold": model_config["trading_config"]["confidence_threshold"],
            "onnx_size": human_size(onnx_output.stat().st_size),
        },
        "prediction": {
            "label": report["prediction"]["label"],
            "probabilities": [
                {"label": "SHORT", "value": float(validation_row["prob_short"])},
                {"label": "HOLD", "value": float(validation_row["prob_hold"])},
                {"label": "LONG", "value": float(validation_row["prob_long"])},
            ],
        },
        "feature_snapshot": feature_snapshot,
        "artifacts": build_artifact_list(
            [
                ("Fetched Raw CSV", artifacts["raw_output"]),
                ("Standardized CSV", artifacts["standardized_output"]),
                ("Overlap CSV", artifacts["overlap_output"]),
                ("Feature CSV", artifacts["feature_output"]),
                ("Labeled CSV", artifacts["label_output"]),
                ("Live Model", artifacts["model_output"]),
                ("Confidence Analysis", artifacts.get("confidence_output")),
                ("Approx Backtest", artifacts.get("backtest_output")),
                ("Validation Fixture", artifacts["validation_output"]),
                ("Live ONNX Model", artifacts["onnx_output"]),
            ]
        ),
        "previews": {
            "standardized": preview_frame(standardized_output, rows=5),
            "overlap": preview_frame(overlap_output, rows=5),
            "validation": preview_frame(validation_output, rows=3),
        },
    }


def make_handler(payload: dict):
    class DemoHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/dashboard":
                body = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            target = "index.html" if parsed.path in ("", "/") else parsed.path.lstrip("/")
            file_path = (STATIC_DIR / target).resolve()
            if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return

            content = file_path.read_bytes()
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args) -> None:
            return

    return DemoHandler


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.twelvedata_env)
    use_live_pipeline = args.pipeline_mode == "live" or (args.pipeline_mode == "auto" and api_key is not None)

    live_market = None
    if api_key:
        try:
            time_series = fetch_time_series(
                api_key=api_key,
                symbol=args.twelvedata_symbol,
                outputsize=args.twelvedata_outputsize,
            )
            live_market = build_live_market_payload(time_series=time_series)
        except Exception as exc:
            live_market = build_live_market_payload(time_series=None, error=str(exc))

    if use_live_pipeline:
        if args.skip_refresh:
            report_path = resolve_repo_path(DEFAULT_LIVE_REPORT_PATH)
            if not report_path.exists():
                raise SystemExit(
                    f"Live dashboard refresh was skipped, but {display_path(report_path)} does not exist yet."
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = run_live_pipeline(
                api_env=args.twelvedata_env,
                symbol=args.twelvedata_symbol,
                outputsize=args.twelvedata_pipeline_outputsize,
                volume_mode=args.twelvedata_volume_mode,
                skip_onnx_runtime_check=args.skip_onnx_runtime_check,
            )
        payload = build_live_dashboard_payload(report, live_market)
    else:
        demo_result = load_existing_sample_result() if args.skip_refresh else run_demo_pipeline(
            input_path_like=args.input,
            model_path_like=args.model,
            skip_onnx_runtime_check=args.skip_onnx_runtime_check,
        )
        payload = build_sample_dashboard_payload(demo_result, live_market)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(payload))
    print(f"Dashboard available at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
