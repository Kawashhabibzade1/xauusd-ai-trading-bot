"""
Train the canonical 68-feature LightGBM classifier for XAUUSD.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import json
import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pipeline_contract import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_LABEL_OUTPUT,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_MODEL_PATH,
    LABEL_EXCLUDE_COLUMNS,
    assert_ordered_features,
    display_path,
    ensure_parent_dir,
    resolve_repo_path,
    write_feature_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_LABEL_OUTPUT, help="Labeled CSV input.")
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="Feature contract YAML path.")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="Training config YAML path.")
    parser.add_argument("--model-output", default=DEFAULT_MODEL_PATH, help="LightGBM model output path.")
    parser.add_argument("--feature-list-output", default=DEFAULT_FEATURE_LIST_PATH, help="Feature list JSON output.")
    parser.add_argument("--metadata-output", default=DEFAULT_METADATA_PATH, help="Model metadata JSON output.")
    return parser.parse_args()


def load_training_config(path_like: str) -> dict:
    with resolve_repo_path(path_like).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_training_rows(row_count: int, test_split_ratio: float) -> int:
    if row_count == 0:
        raise ValueError(
            "Insufficient labeled rows for training: found 0 rows. "
            "The checked-in sample/demo dataset is too small to retrain the model; use a larger historical dataset."
        )

    split_idx = int((1.0 - test_split_ratio) * row_count)
    train_rows = split_idx
    test_rows = row_count - split_idx
    if train_rows == 0 or test_rows == 0:
        raise ValueError(
            "Insufficient labeled rows for training split: "
            f"total={row_count}, train={train_rows}, test={test_rows}. "
            "Provide a larger historical dataset or adjust the training split."
        )

    return split_idx


def train_model_from_labeled(
    labeled: pd.DataFrame,
    feature_config: str = DEFAULT_FEATURE_CONFIG,
    model_config_path: str = DEFAULT_MODEL_CONFIG,
    model_output_path: str = DEFAULT_MODEL_PATH,
    feature_list_output_path: str = DEFAULT_FEATURE_LIST_PATH,
    metadata_output_path: str = DEFAULT_METADATA_PATH,
) -> dict:
    training_config = load_training_config(model_config_path)
    params = dict(training_config["lightgbm"])
    training_options = training_config["training"]

    frame = labeled.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    feature_columns = [column for column in frame.columns if column not in LABEL_EXCLUDE_COLUMNS]
    ordered_features = assert_ordered_features(feature_columns, feature_config, context="training features")

    X = frame[ordered_features].values
    y = frame["label"].values + 1

    split_idx = validate_training_rows(len(frame), training_options["test_split_ratio"])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=ordered_features)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=training_options["num_boost_round"],
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=training_options["early_stopping_rounds"]),
            lgb.log_evaluation(period=training_options["log_evaluation_period"]),
        ],
    )

    best_iteration = int(model.best_iteration or training_options["num_boost_round"])
    y_pred_proba = model.predict(X_test, num_iteration=best_iteration)
    y_pred = y_pred_proba.argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    report_text = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["SHORT", "HOLD", "LONG"],
        digits=3,
        zero_division=0,
    )

    model_output = ensure_parent_dir(model_output_path)
    model.save_model(model_output)
    feature_list_output = write_feature_list(ordered_features, feature_list_output_path)

    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "model_type": "LightGBM",
        "num_features": len(ordered_features),
        "num_classes": 3,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": float(accuracy),
        "best_iteration": best_iteration,
        "params": params,
        "feature_config": display_path(feature_config),
    }
    metadata_output = ensure_parent_dir(metadata_output_path)
    with metadata_output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "model_output": model_output,
        "feature_list_output": feature_list_output,
        "metadata_output": metadata_output,
        "ordered_features": ordered_features,
        "best_iteration": best_iteration,
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("TRAIN LIGHTGBM MODEL")
    print("=" * 70)
    print(f"Input         : {display_path(args.input)}")
    print(f"Feature config: {display_path(args.feature_config)}")
    print(f"Model config  : {display_path(args.model_config)}")
    print()

    labeled = pd.read_csv(resolve_repo_path(args.input))
    result = train_model_from_labeled(
        labeled=labeled,
        feature_config=args.feature_config,
        model_config_path=args.model_config,
        model_output_path=args.model_output,
        feature_list_output_path=args.feature_list_output,
        metadata_output_path=args.metadata_output,
    )
    metadata = result["metadata"]

    print()
    print("Accuracy:")
    print(f"  overall: {metadata['accuracy']:.2%}")
    print()
    print("Classification report:")
    print(result["classification_report"])
    print("Confusion matrix:")
    print(result["confusion_matrix"])
    print()
    print(f"Model saved       : {display_path(result['model_output'])}")
    print(f"Feature list saved: {display_path(args.feature_list_output)}")
    print(f"Metadata saved    : {display_path(args.metadata_output)}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
