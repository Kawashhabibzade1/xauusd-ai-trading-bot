"""
Create 3-class training labels from engineered features using 15-minute forward returns.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline_contract import (
    BASE_COLUMNS,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_FEATURE_OUTPUT,
    DEFAULT_LABEL_OUTPUT,
    assert_ordered_features,
    display_path,
    ensure_parent_dir,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_FEATURE_OUTPUT, help="Feature CSV input.")
    parser.add_argument("--output", default=DEFAULT_LABEL_OUTPUT, help="Labeled CSV output.")
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="Feature contract YAML path.")
    parser.add_argument("--buy-threshold", type=float, default=0.0005, help="Forward-return threshold for LONG.")
    parser.add_argument("--sell-threshold", type=float, default=-0.0005, help="Forward-return threshold for SHORT.")
    return parser.parse_args()


def create_labeled_frame(
    features: pd.DataFrame,
    feature_config: str = DEFAULT_FEATURE_CONFIG,
    buy_threshold: float = 0.0005,
    sell_threshold: float = -0.0005,
) -> pd.DataFrame:
    labeled = features.copy()
    labeled["time"] = pd.to_datetime(labeled["time"])
    feature_columns = [column for column in labeled.columns if column not in BASE_COLUMNS]
    assert_ordered_features(feature_columns, feature_config, context="label input features")

    labeled["forward_return_15m"] = labeled["close"].shift(-15).div(labeled["close"]) - 1.0
    labeled["label"] = np.where(
        labeled["forward_return_15m"] > buy_threshold,
        1,
        np.where(labeled["forward_return_15m"] < sell_threshold, -1, 0),
    )
    labeled = labeled.dropna(subset=["forward_return_15m"]).reset_index(drop=True)
    return labeled


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("CREATE TRAINING LABELS")
    print("=" * 70)
    print(f"Input : {display_path(args.input)}")
    print(f"Output: {display_path(args.output)}")
    print()

    features = pd.read_csv(resolve_repo_path(args.input))
    labeled = create_labeled_frame(
        features,
        feature_config=args.feature_config,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )

    output_path = ensure_parent_dir(args.output)
    labeled.to_csv(output_path, index=False)

    distribution = labeled["label"].value_counts().sort_index().to_dict()
    print(f"Rows              : {len(labeled):,}")
    print(f"Thresholds        : buy>{args.buy_threshold:.4%}, sell<{args.sell_threshold:.4%}")
    print(f"Class distribution: {distribution}")
    print(f"Saved             : {display_path(output_path)}")


if __name__ == "__main__":
    main()
