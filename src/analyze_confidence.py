"""
Analyze model performance across confidence thresholds.
"""

from __future__ import annotations

import argparse

import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from pipeline_contract import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_LABEL_OUTPUT,
    DEFAULT_MODEL_PATH,
    LABEL_EXCLUDE_COLUMNS,
    assert_ordered_features,
    display_path,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_LABEL_OUTPUT, help="Labeled CSV input.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="LightGBM model path.")
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="Feature contract YAML path.")
    parser.add_argument("--feature-list", default=DEFAULT_FEATURE_LIST_PATH, help="Feature list JSON path.")
    return parser.parse_args()


def analyze_confidence_from_labeled(
    labeled: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
    feature_config: str = DEFAULT_FEATURE_CONFIG,
    feature_list_path: str = DEFAULT_FEATURE_LIST_PATH,
) -> dict:
    frame = labeled.copy()
    frame["time"] = pd.to_datetime(frame["time"])

    with resolve_repo_path(feature_list_path).open("r", encoding="utf-8") as handle:
        feature_columns = json.load(handle)
    assert_ordered_features(feature_columns, feature_config, context="feature list")

    data_feature_columns = [column for column in frame.columns if column not in LABEL_EXCLUDE_COLUMNS]
    assert_ordered_features(data_feature_columns, feature_config, context="analysis input features")

    model = lgb.Booster(model_file=str(resolve_repo_path(model_path)))

    split_idx = int(0.8 * len(frame))
    X_test = frame[feature_columns].iloc[split_idx:].values
    y_test = (frame["label"].iloc[split_idx:] + 1).values
    if len(y_test) == 0:
        raise ValueError("Confidence analysis requires test rows after the train/test split.")

    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    max_proba = np.max(y_pred_proba, axis=1)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    results = []

    for threshold in thresholds:
        conf_mask = max_proba >= threshold
        if conf_mask.sum() == 0:
            continue

        y_test_conf = y_test[conf_mask]
        y_pred_conf = y_pred[conf_mask]
        accuracy = accuracy_score(y_test_conf, y_pred_conf)
        results.append(
            {
                "threshold": float(threshold),
                "samples": int(conf_mask.sum()),
                "accuracy": float(accuracy),
                "pct_total": float(conf_mask.sum() / len(y_test) * 100.0),
                "short_acc": float(((y_pred_conf == 0) & (y_test_conf == 0)).sum() / max((y_test_conf == 0).sum(), 1)),
                "hold_acc": float(((y_pred_conf == 1) & (y_test_conf == 1)).sum() / max((y_test_conf == 1).sum(), 1)),
                "long_acc": float(((y_pred_conf == 2) & (y_test_conf == 2)).sum() / max((y_test_conf == 2).sum(), 1)),
            }
        )

    if not results:
        raise ValueError("No confidence-threshold results were produced.")

    selected = next((result for result in results if result["accuracy"] >= 0.60 and result["pct_total"] >= 1.0), None)
    selection_reason = "target_met"
    if selected is None:
        selected = max(results, key=lambda result: result["accuracy"])
        selection_reason = "highest_accuracy_fallback"

    conf_mask = max_proba >= selected["threshold"]
    y_test_conf = y_test[conf_mask]
    y_pred_conf = y_pred[conf_mask]

    return {
        "test_samples": int(len(y_test)),
        "threshold_results": results,
        "selected_threshold": float(selected["threshold"]),
        "selected_accuracy": float(selected["accuracy"]),
        "selected_retained_pct": float(selected["pct_total"]),
        "selection_reason": selection_reason,
        "classification_report": classification_report(
            y_test_conf,
            y_pred_conf,
            labels=[0, 1, 2],
            target_names=["SHORT", "HOLD", "LONG"],
            digits=3,
            zero_division=0,
        ),
    }


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 70)
    print(f"Input : {display_path(args.input)}")
    print(f"Model : {display_path(args.model)}")
    print()

    labeled = pd.read_csv(resolve_repo_path(args.input))
    result = analyze_confidence_from_labeled(
        labeled=labeled,
        model_path=args.model,
        feature_config=args.feature_config,
        feature_list_path=args.feature_list,
    )

    for threshold_result in result["threshold_results"]:
        print(
            f"Threshold >= {threshold_result['threshold']:.0%}: samples={threshold_result['samples']:,}, "
            f"accuracy={threshold_result['accuracy']:.2%}, retained={threshold_result['pct_total']:.1f}%"
        )

    print()
    if result["selection_reason"] == "highest_accuracy_fallback":
        print("No threshold met the 60% accuracy and 1% retention target.")
        print(f"Falling back to highest-accuracy threshold: {result['selected_threshold']:.0%}")
    else:
        print(f"Recommended threshold: {result['selected_threshold']:.0%}")

    print()
    print("Detailed classification report:")
    print(result["classification_report"])
    print(
        f"Selected threshold summary: threshold={result['selected_threshold']:.0%}, "
        f"accuracy={result['selected_accuracy']:.2%}, retained={result['selected_retained_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
