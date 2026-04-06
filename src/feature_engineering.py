"""
Feature engineering for the canonical 68-feature XAUUSD model contract.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from pipeline_contract import (
    BASE_COLUMNS,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_FEATURE_OUTPUT,
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
    parser.add_argument("--output", default=DEFAULT_FEATURE_OUTPUT, help="Feature CSV output.")
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="Feature contract YAML path.")
    return parser.parse_args()


def compute_feature_frame(df: pd.DataFrame, feature_config: str = DEFAULT_FEATURE_CONFIG) -> pd.DataFrame:
    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"])

    atr_14 = AverageTrueRange(frame["high"], frame["low"], frame["close"], window=14)
    frame["atr_14"] = atr_14.average_true_range()

    atr_5 = AverageTrueRange(frame["high"], frame["low"], frame["close"], window=5)
    frame["atr_5"] = atr_5.average_true_range()

    rsi = RSIIndicator(frame["close"], window=14)
    frame["rsi_14"] = rsi.rsi()

    ema12 = EMAIndicator(frame["close"], window=12)
    frame["ema_12"] = ema12.ema_indicator()
    ema26 = EMAIndicator(frame["close"], window=26)
    frame["ema_26"] = ema26.ema_indicator()
    frame["ema_12_slope"] = frame["ema_12"].diff()
    frame["ema_26_slope"] = frame["ema_26"].diff()

    frame["sma_50"] = SMAIndicator(frame["close"], window=50).sma_indicator()
    frame["sma_200"] = SMAIndicator(frame["close"], window=200).sma_indicator()

    macd = MACD(frame["close"])
    frame["macd"] = macd.macd()
    frame["macd_signal"] = macd.macd_signal()
    frame["macd_histogram"] = macd.macd_diff()

    bb = BollingerBands(frame["close"], window=20, window_dev=2)
    frame["bb_upper"] = bb.bollinger_hband()
    frame["bb_lower"] = bb.bollinger_lband()
    frame["bb_middle"] = bb.bollinger_mavg()
    frame["bb_width"] = frame["bb_upper"] - frame["bb_lower"]
    frame["bb_position"] = (frame["close"] - frame["bb_lower"]) / (frame["bb_upper"] - frame["bb_lower"] + 0.0001)

    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    frame["vwap"] = (frame["volume"] * typical_price).rolling(60).sum() / (frame["volume"].rolling(60).sum() + 0.0001)
    frame["price_to_vwap"] = frame["close"] / (frame["vwap"] + 0.0001)

    stoch = StochasticOscillator(frame["high"], frame["low"], frame["close"], window=14, smooth_window=3)
    frame["stoch_k"] = stoch.stoch()
    frame["stoch_d"] = stoch.stoch_signal()

    frame["swing_high_dist"] = frame["high"].rolling(5).max() - frame["close"]
    frame["swing_low_dist"] = frame["close"] - frame["low"].rolling(5).min()
    frame["bullish_ob"] = ((frame["close"].shift(1) > frame["open"].shift(1)) & (frame["close"] < frame["open"])).astype(int)
    frame["bearish_ob"] = ((frame["close"].shift(1) < frame["open"].shift(1)) & (frame["close"] > frame["open"])).astype(int)
    frame["fvg_up"] = frame["low"] - frame["high"].shift(2)
    frame["fvg_down"] = frame["low"].shift(2) - frame["high"]
    frame["fvg_size"] = frame[["fvg_up", "fvg_down"]].max(axis=1)
    frame["liquidity_sweep_high"] = (
        (frame["high"] > frame["high"].rolling(20).max().shift(1)) & (frame["close"] < frame["open"])
    ).astype(int)
    frame["liquidity_sweep_low"] = (
        (frame["low"] < frame["low"].rolling(20).min().shift(1)) & (frame["close"] > frame["open"])
    ).astype(int)

    session_high = frame["high"].rolling(240).max()
    session_low = frame["low"].rolling(240).min()
    frame["premium_discount"] = (frame["close"] - session_low) / (session_high - session_low + 0.0001)

    frame["bar_direction"] = np.sign(frame["close"] - frame["open"])
    frame["delta"] = frame["volume"] * frame["bar_direction"]
    frame["cvd"] = frame["delta"].cumsum()
    frame["price_change"] = frame["close"].pct_change()
    frame["cvd_change"] = frame["cvd"].pct_change()
    frame["cvd_divergence"] = frame["price_change"] - frame["cvd_change"]
    frame["volume_ma"] = frame["volume"].rolling(20).mean()
    frame["volume_ratio"] = frame["volume"] / (frame["volume_ma"] + 0.0001)
    frame["price_range_norm"] = (frame["high"] - frame["low"]) / (frame["atr_14"] + 0.0001)
    frame["absorption_score"] = frame["volume_ratio"] * (1.0 / (frame["price_range_norm"] + 0.001))

    frame["hour"] = frame["time"].dt.hour
    frame["minute"] = frame["time"].dt.minute
    frame["dayofweek"] = frame["time"].dt.dayofweek
    frame["minutes_since_london"] = (frame["hour"] * 60 + frame["minute"]) - (8 * 60)
    frame["minutes_since_ny"] = (frame["hour"] * 60 + frame["minute"]) - (13 * 60 + 30)
    frame["session_position"] = (frame["minutes_since_ny"] / (3.5 * 60)).clip(0, 1)

    frame["atr_percentile"] = frame["atr_14"].rolling(240).apply(
        lambda values: (values.iloc[-1] <= values).sum() / len(values) if len(values) else 0.5,
        raw=False,
    )
    frame["tick_volatility"] = frame["close"].rolling(10).std()
    frame["range_expansion"] = (frame["high"] - frame["low"]) / (frame["high"].shift(1) - frame["low"].shift(1) + 0.0001)
    atr_mean = frame["atr_14"].rolling(240).mean()
    atr_std = frame["atr_14"].rolling(240).std()
    frame["volatility_regime"] = ((frame["atr_14"] - atr_mean) / (atr_std + 0.0001)).fillna(0)
    frame["true_range"] = frame["high"] - frame["low"]
    frame["tr_percentile"] = frame["true_range"].rolling(60).apply(
        lambda values: (values.iloc[-1] <= values).sum() / len(values) if len(values) else 0.5,
        raw=False,
    )
    frame["price_velocity"] = frame["close"].diff(3) / 3
    frame["price_acceleration"] = frame["price_velocity"].diff()

    frame["returns_1m"] = frame["close"].pct_change()
    frame["returns_5m"] = frame["close"].pct_change(5)
    frame["returns_15m"] = frame["close"].pct_change(15)
    frame["momentum"] = frame["close"] - frame["close"].shift(14)
    frame["dist_to_high"] = (frame["high"].rolling(50).max() - frame["close"]) / (frame["atr_14"] + 0.0001)
    frame["dist_to_low"] = (frame["close"] - frame["low"].rolling(50).min()) / (frame["atr_14"] + 0.0001)
    frame["sentiment"] = 0.0

    h4_high = frame["high"].rolling(240).max()
    h4_low = frame["low"].rolling(240).min()
    h4_mid = (h4_high + h4_low) / 2.0
    frame["h4_bias"] = np.where(
        (frame["close"] > h4_mid) & (frame["close"] > frame["close"].shift(240)),
        1,
        np.where((frame["close"] < h4_mid) & (frame["close"] < frame["close"].shift(240)), -1, 0),
    )
    frame["in_discount"] = (frame["close"] < h4_mid).astype(int)
    frame["in_premium"] = (frame["close"] > h4_mid).astype(int)
    frame["inducement_taken"] = (frame["liquidity_sweep_high"] | frame["liquidity_sweep_low"]).astype(int)
    frame["entry_zone_present"] = (
        (frame["fvg_size"] > 0) | (frame["bullish_ob"] == 1) | (frame["bearish_ob"] == 1)
    ).astype(int)
    frame["smc_quality_score"] = (
        (frame["h4_bias"] != 0).astype(int)
        + ((frame["in_discount"] == 1) | (frame["in_premium"] == 1)).astype(int)
        + (frame["inducement_taken"] == 1).astype(int)
        + (frame["entry_zone_present"] == 1).astype(int)
    )

    ordered_features = get_ordered_features(feature_config)
    feature_only = frame.dropna(subset=ordered_features).copy()
    feature_only = feature_only.loc[:, list(BASE_COLUMNS) + ordered_features]
    assert_ordered_features(feature_only.columns[len(BASE_COLUMNS):], feature_config, context="engineered features")
    return feature_only.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("XAUUSD FEATURE ENGINEERING")
    print("=" * 70)
    print(f"Input         : {display_path(args.input)}")
    print(f"Output        : {display_path(args.output)}")
    print(f"Feature config: {display_path(args.feature_config)}")
    print()

    input_path = resolve_repo_path(args.input)
    overlap = pd.read_csv(input_path)
    features = compute_feature_frame(overlap, args.feature_config)
    ordered_features = get_ordered_features(args.feature_config)

    output_path = ensure_parent_dir(args.output)
    features.to_csv(output_path, index=False)

    print(f"Rows         : {len(features):,}")
    print(f"Feature count: {len(ordered_features)}")
    print(f"Date range   : {features['time'].min()} -> {features['time'].max()}")
    print(f"Saved        : {display_path(output_path)}")


if __name__ == "__main__":
    main()
