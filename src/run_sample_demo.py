"""
Run the checked-in sample/demo pipeline and regenerate the visible MT5 artifacts.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys


DEFAULT_INPUT = "data/xauusd_m1_2022_2025.csv"
DEFAULT_MODEL = "python_training/models/lightgbm_xauusd_v1.txt"
DEFAULT_FEATURE_LIST = "python_training/models/feature_list.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Sample/demo OHLCV CSV input.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Saved LightGBM model used for the demo.")
    parser.add_argument(
        "--skip-onnx-runtime-check",
        action="store_true",
        help="Skip the optional onnxruntime inference verification after export.",
    )
    return parser.parse_args()


def require_module(module_name: str, install_hint: str, display_name: str | None = None) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        extra = ""
        if module_name == "lightgbm" and platform.system() == "Darwin":
            extra = " On macOS, LightGBM also needs libomp: `brew install libomp`."
        dependency = display_name or module_name
        raise RuntimeError(
            f"Dependency check failed for {dependency}: {exc}. {install_hint}{extra}"
        ) from exc


def verify_dependencies(skip_onnx_runtime_check: bool) -> None:
    base_install_hint = "Install base dependencies with `pip install -r requirements.txt`."
    onnx_install_hint = "Install ONNX export dependencies with `pip install -r requirements-onnx.txt`."

    for module_name, display_name in (
        ("numpy", None),
        ("pandas", None),
        ("ta", None),
        ("yaml", "PyYAML"),
        ("lightgbm", None),
    ):
        require_module(module_name, base_install_hint, display_name)

    for module_name in ("onnx", "onnxmltools"):
        require_module(module_name, onnx_install_hint)

    if not skip_onnx_runtime_check:
        require_module("onnxruntime", onnx_install_hint)


def run_demo_pipeline(
    input_path_like: str = DEFAULT_INPUT,
    model_path_like: str = DEFAULT_MODEL,
    skip_onnx_runtime_check: bool = False,
) -> dict:
    verify_dependencies(skip_onnx_runtime_check)

    from export_mt5_validation_set import export_validation_fixture
    from export_to_onnx_simple import export_model_to_onnx
    from feature_engineering import compute_feature_frame
    from filter_overlap import filter_overlap_frame
    from pipeline_contract import (
        DEFAULT_FEATURE_CONFIG,
        DEFAULT_FEATURE_OUTPUT,
        DEFAULT_MT5_FEATURES_PATH,
        DEFAULT_MT5_MODEL_CONFIG_PATH,
        DEFAULT_MT5_ONNX_OUTPUT,
        DEFAULT_MT5_VALIDATION_OUTPUT,
        DEFAULT_OVERLAP_OUTPUT,
        DEFAULT_STANDARDIZED_OUTPUT,
        display_path,
        ensure_parent_dir,
        resolve_repo_path,
    )
    from validate_merged_data import load_and_standardize, validate_standardized

    input_path = resolve_repo_path(input_path_like)
    model_path = resolve_repo_path(model_path_like)
    feature_list_path = resolve_repo_path(DEFAULT_FEATURE_LIST)

    if not input_path.exists():
        raise SystemExit(f"Sample/demo input CSV not found: {display_path(input_path)}")
    if not model_path.exists():
        raise SystemExit(
            "Saved demo model not found: "
            f"{display_path(model_path)}. This demo mode expects the checked-in LightGBM model artifact."
        )
    if not feature_list_path.exists():
        raise SystemExit(
            "Saved feature list not found: "
            f"{display_path(feature_list_path)}. This demo mode expects the checked-in training artifacts."
        )

    standardized = load_and_standardize(input_path_like)
    standardized_stats = validate_standardized(standardized)
    standardized_path = ensure_parent_dir(DEFAULT_STANDARDIZED_OUTPUT)
    standardized.to_csv(standardized_path, index=False)

    overlap = filter_overlap_frame(standardized)
    overlap_path = ensure_parent_dir(DEFAULT_OVERLAP_OUTPUT)
    overlap.to_csv(overlap_path, index=False)

    features = compute_feature_frame(overlap, DEFAULT_FEATURE_CONFIG)
    if features.empty:
        raise SystemExit(
            "Feature generation produced 0 rows. The sample/demo input does not contain enough history for the 68-feature contract."
        )
    features_path = ensure_parent_dir(DEFAULT_FEATURE_OUTPUT)
    features.to_csv(features_path, index=False)

    fixture_result = export_validation_fixture(
        input_path=DEFAULT_OVERLAP_OUTPUT,
        model_path=model_path_like,
        output_path=DEFAULT_MT5_VALIDATION_OUTPUT,
        feature_config=DEFAULT_FEATURE_CONFIG,
    )
    if fixture_result["rows"] == 0:
        raise SystemExit("Validation fixture export produced 0 rows. Check the sample/demo input and saved model artifacts.")

    onnx_result = export_model_to_onnx(
        model_path=model_path_like,
        feature_list_path=DEFAULT_FEATURE_LIST,
        feature_config=DEFAULT_FEATURE_CONFIG,
        output_path=DEFAULT_MT5_ONNX_OUTPUT,
        mt5_features_output=DEFAULT_MT5_FEATURES_PATH,
        mt5_config_output=DEFAULT_MT5_MODEL_CONFIG_PATH,
        skip_runtime_check=skip_onnx_runtime_check,
    )

    return {
        "input_path": input_path,
        "model_path": model_path,
        "standardized_rows": len(standardized),
        "standardized_start": str(standardized_stats["start"]),
        "standardized_end": str(standardized_stats["end"]),
        "overlap_rows": len(overlap),
        "overlap_start": str(overlap["time"].min()),
        "overlap_end": str(overlap["time"].max()),
        "feature_rows": len(features),
        "feature_time": str(features["time"].iloc[-1]),
        "validation_rows": fixture_result["rows"],
        "runtime_status": onnx_result["runtime_status"],
        "standardized_path": standardized_path,
        "overlap_path": overlap_path,
        "features_path": features_path,
        "validation_path": fixture_result["output_path"],
        "onnx_path": onnx_result["output_path"],
        "mt5_features_path": onnx_result["mt5_features_output"],
        "mt5_config_path": onnx_result["mt5_config_output"],
    }


def print_demo_summary(result: dict) -> None:
    from pipeline_contract import display_path

    print("Generated demo artifacts:")
    print(f"  standardized rows : {result['standardized_rows']:,}")
    print(f"  overlap rows      : {result['overlap_rows']:,}")
    print(f"  feature rows      : {result['feature_rows']:,}")
    print(f"  validation rows   : {result['validation_rows']:,}")
    print()
    print("Output files:")
    print(f"  {display_path(result['standardized_path'])}")
    print(f"  {display_path(result['overlap_path'])}")
    print(f"  {display_path(result['features_path'])}")
    print(f"  {display_path(result['validation_path'])}")
    print(f"  {display_path(result['onnx_path'])}")
    print()
    print(f"ONNX runtime check : {result['runtime_status']}")


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("XAUUSD SAMPLE DEMO")
    print("=" * 70)
    print("This run regenerates demo artifacts from the checked-in sample CSV and saved LightGBM model.")
    print()

    result = run_demo_pipeline(
        input_path_like=args.input,
        model_path_like=args.model,
        skip_onnx_runtime_check=args.skip_onnx_runtime_check,
    )
    print_demo_summary(result)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
