"""
Install the one-shot MT5 demo proof-trade script into the local MetaTrader 5 Scripts directory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from install_mt5_exporter import compile_exporter
from mt5_client import get_mt5_scripts_dir
from pipeline_contract import resolve_repo_path


SCRIPT_SOURCE = Path("mt5_expert_advisor/XAUUSD_Demo_Proof_Trade.mq5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-dir", default="OpenAI", help="Subdirectory inside MQL5/Scripts.")
    parser.add_argument("--skip-compile", action="store_true", help="Only copy the script without compiling it.")
    return parser.parse_args()


def install_script(relative_dir: str = "OpenAI") -> dict[str, str]:
    scripts_dir = get_mt5_scripts_dir()
    if scripts_dir is None:
        raise RuntimeError("Could not find the MT5 MQL5/Scripts directory on this machine.")

    source = resolve_repo_path(SCRIPT_SOURCE)
    if not source.exists():
        raise FileNotFoundError(source)

    target_dir = scripts_dir / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    return {
        "source": str(source),
        "target_dir": str(target_dir),
        "target": str(target),
    }


def main() -> None:
    args = parse_args()
    install_result = install_script(relative_dir=args.relative_dir)

    print("=" * 70)
    print("MT5 DEMO PROOF SCRIPT INSTALL")
    print("=" * 70)
    print(f"Source      : {install_result['source']}")
    print(f"Target dir  : {install_result['target_dir']}")
    print(f"Target file : {install_result['target']}")

    if not args.skip_compile:
        try:
            compile_result = compile_exporter(Path(install_result["target"]))
        except Exception as exc:
            print(f"Compile: skipped ({exc})")
        else:
            print(f"Compile return code : {compile_result['returncode']}")
            print(f"Compiled .ex5 exists: {compile_result['compiled_exists']}")
            print(f"Compile success    : {compile_result['success']}")
            if compile_result["log_file"]:
                print(f"Compile log        : {compile_result['log_file']}")

    print()
    print("Next in MT5:")
    print("  1. Open Navigator -> Scripts -> OpenAI")
    print("  2. Drag XAUUSD_Demo_Proof_Trade onto the XAUUSD M1 chart once")
    print("  3. Keep InpLotSize=0.01 and InpHoldSeconds=60")
    print("  4. The script opens one demo trade, waits 60 seconds, and closes it")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
