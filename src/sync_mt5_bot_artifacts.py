"""
Copy generated MT5 bot artifacts from the repo into the local MetaTrader 5 MQL5/Files directory.
"""

from __future__ import annotations

import argparse
import sys

from mt5_client import sync_mt5_file_artifacts
from pipeline_contract import (
    DEFAULT_MT5_LIVE_MT5_FEATURES_PATH,
    DEFAULT_MT5_LIVE_MT5_MODEL_CONFIG_PATH,
    DEFAULT_MT5_LIVE_MT5_ONNX_OUTPUT,
    DEFAULT_MT5_LIVE_MT5_VALIDATION_OUTPUT,
    DEFAULT_MT5_ONNX_OUTPUT,
    DEFAULT_MT5_TRADE_DIRECTIVE_OUTPUT,
    DEFAULT_MT5_FEATURES_PATH,
    DEFAULT_MT5_MODEL_CONFIG_PATH,
    DEFAULT_MT5_VALIDATION_OUTPUT,
)


VARIANT_ARTIFACTS = {
    "live_mt5": [
        DEFAULT_MT5_LIVE_MT5_ONNX_OUTPUT,
        DEFAULT_MT5_LIVE_MT5_FEATURES_PATH,
        DEFAULT_MT5_LIVE_MT5_MODEL_CONFIG_PATH,
        DEFAULT_MT5_LIVE_MT5_VALIDATION_OUTPUT,
    ],
    "live_mt5_research": [
        DEFAULT_MT5_LIVE_MT5_ONNX_OUTPUT,
        DEFAULT_MT5_LIVE_MT5_FEATURES_PATH,
        DEFAULT_MT5_LIVE_MT5_MODEL_CONFIG_PATH,
        DEFAULT_MT5_LIVE_MT5_VALIDATION_OUTPUT,
        DEFAULT_MT5_TRADE_DIRECTIVE_OUTPUT,
    ],
    "sample": [
        DEFAULT_MT5_ONNX_OUTPUT,
        DEFAULT_MT5_FEATURES_PATH,
        DEFAULT_MT5_MODEL_CONFIG_PATH,
        DEFAULT_MT5_VALIDATION_OUTPUT,
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_ARTIFACTS),
        default="live_mt5",
        help="Which repo artifact set to sync into MT5 MQL5/Files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = VARIANT_ARTIFACTS[args.variant]
    results = sync_mt5_file_artifacts(artifacts)

    print("=" * 70)
    print("SYNC MT5 BOT ARTIFACTS")
    print("=" * 70)
    print(f"Variant : {args.variant}")
    for item in results:
        print(f"{item['relative']}: {item['target']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
