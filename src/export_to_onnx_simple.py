"""
Canonical LightGBM -> ONNX exporter for MT5 validation mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb

from pipeline_contract import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_MT5_FEATURES_PATH,
    DEFAULT_MT5_MODEL_CONFIG_PATH,
    DEFAULT_MT5_ONNX_OUTPUT,
    assert_ordered_features,
    build_mt5_model_config,
    display_path,
    ensure_parent_dir,
    resolve_repo_path,
    write_feature_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Trained LightGBM model path.")
    parser.add_argument("--feature-list", default=DEFAULT_FEATURE_LIST_PATH, help="Feature list JSON path.")
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="Feature contract YAML path.")
    parser.add_argument("--output", default=DEFAULT_MT5_ONNX_OUTPUT, help="ONNX output path.")
    parser.add_argument("--mt5-features-output", default=DEFAULT_MT5_FEATURES_PATH, help="MT5 feature list JSON output.")
    parser.add_argument("--mt5-config-output", default=DEFAULT_MT5_MODEL_CONFIG_PATH, help="MT5 model config JSON output.")
    parser.add_argument("--skip-runtime-check", action="store_true", help="Skip optional onnxruntime inference verification.")
    return parser.parse_args()


def verify_runtime(onnx_path: Path, feature_count: int) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("Skipping runtime verification because onnxruntime is not installed.")
        return

    session = ort.InferenceSession(str(onnx_path))
    test_input = np.random.randn(1, feature_count).astype("float32")
    session.run(None, {session.get_inputs()[0].name: test_input})


def export_model_to_onnx(
    model_path: str = DEFAULT_MODEL_PATH,
    feature_list_path: str = DEFAULT_FEATURE_LIST_PATH,
    feature_config: str = DEFAULT_FEATURE_CONFIG,
    output_path: str = DEFAULT_MT5_ONNX_OUTPUT,
    mt5_features_output: str = DEFAULT_MT5_FEATURES_PATH,
    mt5_config_output: str = DEFAULT_MT5_MODEL_CONFIG_PATH,
    skip_runtime_check: bool = False,
) -> dict:
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as exc:
        raise SystemExit(
            "Missing ONNX export dependencies. Install requirements-onnx.txt before running the exporter."
        ) from exc

    with resolve_repo_path(feature_list_path).open("r", encoding="utf-8") as handle:
        feature_columns = json.load(handle)
    assert_ordered_features(feature_columns, feature_config, context="export feature list")

    model = lgb.Booster(model_file=str(resolve_repo_path(model_path)))
    initial_types = [("input", FloatTensorType([None, len(feature_columns)]))]
    onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_types, target_opset=12)

    output_file = ensure_parent_dir(output_path)
    with output_file.open("wb") as handle:
        handle.write(onnx_model.SerializeToString())

    write_feature_list(feature_columns, mt5_features_output)
    config = build_mt5_model_config(
        feature_columns,
        model_filename=resolve_repo_path(output_path).name,
        num_trees=model.num_trees(),
    )
    with ensure_parent_dir(mt5_config_output).open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    runtime_status = "skipped by flag"
    if not skip_runtime_check:
        verify_runtime(output_file, len(feature_columns))
        runtime_status = "OK"

    return {
        "feature_count": len(feature_columns),
        "output_path": output_file,
        "mt5_features_output": resolve_repo_path(mt5_features_output),
        "mt5_config_output": resolve_repo_path(mt5_config_output),
        "runtime_status": runtime_status,
    }


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("EXPORT LIGHTGBM MODEL TO ONNX")
    print("=" * 70)
    print(f"Model         : {display_path(args.model)}")
    print(f"Feature list  : {display_path(args.feature_list)}")
    print(f"ONNX output   : {display_path(args.output)}")
    print()

    result = export_model_to_onnx(
        model_path=args.model,
        feature_list_path=args.feature_list,
        feature_config=args.feature_config,
        output_path=args.output,
        mt5_features_output=args.mt5_features_output,
        mt5_config_output=args.mt5_config_output,
        skip_runtime_check=args.skip_runtime_check,
    )

    print(f"Runtime verification: {result['runtime_status']}")
    print(f"Feature count : {result['feature_count']}")
    print(f"ONNX saved    : {display_path(result['output_path'])}")
    print(f"MT5 features  : {display_path(result['mt5_features_output'])}")
    print(f"MT5 config    : {display_path(result['mt5_config_output'])}")


if __name__ == "__main__":
    main()
