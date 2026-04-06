"""
Approximate confidence-filtered signal simulation.

This is not an executable trading backtest. It is a coarse classifier-to-signal
simulation useful for relative comparisons only.
"""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline_contract import (
    DEFAULT_BACKTEST_RESULTS_PATH,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_LABEL_OUTPUT,
    DEFAULT_MODEL_PATH,
    LABEL_EXCLUDE_COLUMNS,
    assert_ordered_features,
    display_path,
    ensure_parent_dir,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_LABEL_OUTPUT, help="Labeled CSV input.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="LightGBM model path.")
    parser.add_argument("--feature-list", default=DEFAULT_FEATURE_LIST_PATH, help="Feature list JSON path.")
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="Feature contract YAML path.")
    parser.add_argument("--output", default=DEFAULT_BACKTEST_RESULTS_PATH, help="Approximate results JSON output.")
    parser.add_argument("--confidence-threshold", type=float, default=0.55, help="Signal confidence threshold.")
    parser.add_argument("--initial-capital", type=float, default=50.0, help="Initial equity for the simulation.")
    return parser.parse_args()


def run_signal_simulation(
    labeled: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
    feature_list_path: str = DEFAULT_FEATURE_LIST_PATH,
    feature_config: str = DEFAULT_FEATURE_CONFIG,
    confidence_threshold: float = 0.55,
    initial_capital: float = 50.0,
    output_path: str | None = None,
) -> dict:
    frame = labeled.copy()
    frame["time"] = pd.to_datetime(frame["time"])

    with resolve_repo_path(feature_list_path).open("r", encoding="utf-8") as handle:
        feature_columns = json.load(handle)
    assert_ordered_features(feature_columns, feature_config, context="feature list")

    data_feature_columns = [column for column in frame.columns if column not in LABEL_EXCLUDE_COLUMNS]
    assert_ordered_features(data_feature_columns, feature_config, context="simulation input features")

    model = lgb.Booster(model_file=str(resolve_repo_path(model_path)))

    split_idx = int(0.8 * len(frame))
    df_test = frame.iloc[split_idx:].copy().reset_index(drop=True)
    if df_test.empty:
        raise ValueError("Approximate simulation requires test rows after the train/test split.")

    y_pred_proba = model.predict(df_test[feature_columns].values)
    y_pred = np.argmax(y_pred_proba, axis=1)
    max_proba = np.max(y_pred_proba, axis=1)

    df_test["pred_class"] = y_pred
    df_test["pred_confidence"] = max_proba
    df_trades = df_test.loc[df_test["pred_confidence"] >= confidence_threshold].copy()

    equity = initial_capital
    equity_curve = [equity]
    trades = []

    for _, row in df_trades.iterrows():
        signal = row["pred_class"]
        if signal == 1:
            equity_curve.append(equity)
            continue

        forward_return = row["forward_return_15m"]
        net_return = forward_return if signal == 2 else -forward_return
        pnl = 2.50 if net_return > 0 else -2.50
        equity += pnl
        equity_curve.append(equity)
        trades.append(
            {
                "time": str(row["time"]),
                "signal": "LONG" if signal == 2 else "SHORT",
                "confidence": float(row["pred_confidence"]),
                "pnl": pnl,
            }
        )

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = int((trades_df["pnl"] > 0).sum()) if total_trades else 0
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()) if total_trades else 0.0
    gross_loss = float(abs(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum())) if total_trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    equity_series = pd.Series(equity_curve)
    max_drawdown = float(((equity_series - equity_series.cummax()) / equity_series.cummax()).min()) if len(equity_series) else 0.0

    results = {
        "simulation_type": "approximate_non_executable",
        "config": {
            "confidence_threshold": float(confidence_threshold),
            "initial_capital": float(initial_capital),
        },
        "signals_retained": int(len(df_trades)),
        "performance": {
            "total_trades": total_trades,
            "win_rate": float(winning_trades / total_trades) if total_trades else 0.0,
            "profit_factor": float(profit_factor),
            "ending_equity": float(equity),
            "return_percent": float(((equity - initial_capital) / initial_capital) * 100.0),
            "max_drawdown": max_drawdown * 100.0,
        },
    }

    if output_path:
        output_file = ensure_parent_dir(output_path)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        results["output_path"] = str(output_file)

    return results


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("APPROXIMATE SIGNAL SIMULATION")
    print("=" * 70)
    print("This output is not a trading-grade backtest.")
    print()

    labeled = pd.read_csv(resolve_repo_path(args.input))
    results = run_signal_simulation(
        labeled=labeled,
        model_path=args.model,
        feature_list_path=args.feature_list,
        feature_config=args.feature_config,
        confidence_threshold=args.confidence_threshold,
        initial_capital=args.initial_capital,
        output_path=args.output,
    )

    print(f"Signals retained : {results['signals_retained']:,}")
    print(f"Trades simulated : {results['performance']['total_trades']:,}")
    print(f"Ending equity    : {results['performance']['ending_equity']:.2f}")
    print(f"Approx result    : {display_path(args.output)}")


if __name__ == "__main__":
    main()
