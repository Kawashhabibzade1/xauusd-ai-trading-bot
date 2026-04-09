"""
Install the MT5 trading bot and its helper headers into the local MetaTrader 5 Experts directory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from install_mt5_exporter import compile_exporter
from mt5_client import get_mt5_experts_dir
from pipeline_contract import resolve_repo_path


BOT_SOURCE_ROOT = Path("mt5_expert_advisor")
BOT_MAIN_SOURCE = BOT_SOURCE_ROOT / "XAUUSD_AI_Bot.mq5"
BOT_SUPPORT_FILES = [
    BOT_SOURCE_ROOT / "FeatureEngine.mqh",
    BOT_SOURCE_ROOT / "Features_Orderflow.mqh",
    BOT_SOURCE_ROOT / "Features_Other.mqh",
    BOT_SOURCE_ROOT / "Features_PriceAction.mqh",
    BOT_SOURCE_ROOT / "Features_SMC.mqh",
    BOT_SOURCE_ROOT / "Features_Structure.mqh",
    BOT_SOURCE_ROOT / "Features_Technical.mqh",
    BOT_SOURCE_ROOT / "Features_Time.mqh",
    BOT_SOURCE_ROOT / "Features_Volatility.mqh",
]
DEMO_SAFE_INPUTS = [
    ("InpValidationMode", "false"),
    ("InpEnableDemoTrading", "true"),
    ("InpDemoOnly", "true"),
    ("InpRequireTradeDirective", "true"),
    ("InpMaxDirectiveEntryDriftPoints", "30"),
    ("InpSessionTradeLimit", "0"),
    ("InpModelName", "models\\xauusd_ai_mt5_live.onnx"),
    ("InpUseRiskBasedSizing", "true"),
    ("InpRiskPerTradePercent", "0.25"),
    ("InpDemoMaxLotSize", "0.10"),
    ("InpFixedLotSize", "0.01"),
    ("InpConfidenceThresh", "0.55"),
    ("InpStopAtrMultiple", "1.00"),
    ("InpTakeProfitRR", "1.50"),
    ("InpAllowSignalFlip", "false"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-dir", default="OpenAI", help="Subdirectory inside MQL5/Experts.")
    parser.add_argument("--skip-compile", action="store_true", help="Only copy the bot sources without compiling them.")
    return parser.parse_args()


def install_bot_sources(relative_dir: str = "OpenAI") -> dict[str, object]:
    experts_dir = get_mt5_experts_dir()
    if experts_dir is None:
        raise RuntimeError("Could not find the MT5 MQL5/Experts directory on this machine.")

    target_dir = experts_dir / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for relative_path in [BOT_MAIN_SOURCE, *BOT_SUPPORT_FILES]:
        source = resolve_repo_path(relative_path)
        if not source.exists():
            raise FileNotFoundError(source)
        target = target_dir / source.name
        shutil.copy2(source, target)
        copied_files.append(str(target))

    return {
        "target_dir": str(target_dir),
        "main_target": str(target_dir / BOT_MAIN_SOURCE.name),
        "copied_files": copied_files,
    }


def main() -> None:
    args = parse_args()
    install_result = install_bot_sources(relative_dir=args.relative_dir)

    print("=" * 70)
    print("MT5 BOT INSTALL")
    print("=" * 70)
    print(f"Target dir    : {install_result['target_dir']}")
    print(f"Copied files  : {len(install_result['copied_files'])}")
    print(f"Main EA       : {install_result['main_target']}")

    compile_result = None
    if not args.skip_compile:
        try:
            compile_result = compile_exporter(Path(str(install_result["main_target"])))
        except Exception as exc:
            print(f"Compile: skipped ({exc})")
        else:
            print(f"Compile return code : {compile_result['returncode']}")
            print(f"Compiled .ex5 exists: {compile_result['compiled_exists']}")
            print(f"Compile success    : {compile_result['success']}")
            if compile_result["log_file"]:
                print(f"Compile log        : {compile_result['log_file']}")
            if compile_result["stdout"]:
                print(compile_result["stdout"])
            if compile_result["stderr"]:
                print(compile_result["stderr"])

    print()
    print("Next in MT5:")
    print("  1. Run the live MT5 pipeline so the current ONNX model is generated and synced.")
    print("  2. Open Navigator -> Expert Advisors -> OpenAI")
    print("  3. Attach XAUUSD_AI_Bot to an XAUUSD M1 chart on your demo account")
    print("  4. Load these demo-safe EA inputs before you arm Algo Trading:")
    for key, value in DEMO_SAFE_INPUTS:
        print(f"     {key}={value}")
    print("  5. Keep the MT5 paper/research worker running so the trade directive stays fresh")
    print("  6. Enable Algo Trading after the EA initializes cleanly")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
