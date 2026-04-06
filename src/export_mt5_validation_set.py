"""
Export a fixed validation fixture for MT5 feature and prediction parity checks.
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import pandas as pd

from feature_engineering import compute_feature_frame
from pipeline_contract import (
    BASE_COLUMNS,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_MODEL_PATH,
    DEFAULT_MT5_VALIDATION_OUTPUT,
    DEFAULT_OVERLAP_OUTPUT,
    assert_ordered_features,
    display_path,
    ensure_parent_dir,
    get_ordered_features,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_OVERLAP_OUTPUT, help="Overlap OHLCV CSV input.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Trained LightGBM model path.")
    parser.add_argument("--output", default=DEFAULT_MT5_VALIDATION_OUTPUT, help="Validation fixture CSV output.")
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="Feature contract YAML path.")
    parser.add_argument("--rows", type=int, default=100, help="Number of recent rows to export.")
    return parser.parse_args()


def export_validation_fixture(
    input_path: str = DEFAULT_OVERLAP_OUTPUT,
    model_path: str = DEFAULT_MODEL_PATH,
    output_path: str = DEFAULT_MT5_VALIDATION_OUTPUT,
    feature_config: str = DEFAULT_FEATURE_CONFIG,
    rows: int = 100,
) -> dict:
    overlap = pd.read_csv(resolve_repo_path(input_path))
    features = compute_feature_frame(overlap, feature_config)
    ordered_features = get_ordered_features(feature_config)
    assert_ordered_features(features.columns[len(BASE_COLUMNS):], feature_config, context="fixture features")

    model = lgb.Booster(model_file=str(resolve_repo_path(model_path)))
    prediction_input = features[ordered_features].values
    probabilities = model.predict(prediction_input)

    fixture = features.tail(rows).copy().reset_index(drop=True)
    fixture_probs = probabilities[-len(fixture) :]
    fixture["epoch_utc"] = pd.to_datetime(fixture["time"], utc=True).map(lambda ts: int(ts.timestamp()))
    fixture["expected_class"] = fixture_probs.argmax(axis=1)
    fixture["prob_short"] = fixture_probs[:, 0]
    fixture["prob_hold"] = fixture_probs[:, 1]
    fixture["prob_long"] = fixture_probs[:, 2]

    ordered_columns = (
        ["epoch_utc", "time"]
        + ordered_features
        + ["expected_class", "prob_short", "prob_hold", "prob_long"]
    )
    fixture = fixture.loc[:, ordered_columns]

    output_file = ensure_parent_dir(output_path)
    fixture.to_csv(output_file, index=False, float_format="%.10f")

    return {
        "rows": len(fixture),
        "feature_count": len(ordered_features),
        "output_path": output_file,
    }


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("EXPORT MT5 VALIDATION FIXTURE")
    print("=" * 70)
    print(f"Input : {display_path(args.input)}")
    print(f"Model : {display_path(args.model)}")
    print(f"Output: {display_path(args.output)}")
    print()

    result = export_validation_fixture(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        feature_config=args.feature_config,
        rows=args.rows,
    )

    print(f"Rows exported : {result['rows']:,}")
    print(f"Feature count : {result['feature_count']}")
    print(f"Saved         : {display_path(result['output_path'])}")


if __name__ == "__main__":
    main()
