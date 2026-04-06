"""
Install the MT5 live exporter into the local MetaTrader 5 Experts directory and try to compile it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from mt5_client import (
    get_mt5_metaeditor_exe,
    get_mt5_wine64,
    get_mt5_wine_prefix,
    install_exporter_source,
)


DEFAULT_EXPORTER_SOURCE = "mt5_expert_advisor/MT5_Live_Data_Exporter.mq5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_EXPORTER_SOURCE, help="Exporter source file inside the repo.")
    parser.add_argument("--relative-dir", default="OpenAI", help="Subdirectory inside MQL5/Experts.")
    parser.add_argument("--skip-compile", action="store_true", help="Only copy the exporter source without compiling it.")
    return parser.parse_args()


def _to_windows_path(path: Path, wine_prefix: Path) -> str:
    path = path.resolve()
    drive_c = (wine_prefix / "drive_c").resolve()
    try:
        relative = path.relative_to(drive_c)
    except ValueError:
        return "Z:" + str(path).replace("/", "\\")
    return "C:\\" + str(relative).replace("/", "\\")


def compile_exporter(target_file: Path) -> dict:
    wine_prefix = get_mt5_wine_prefix()
    wine64 = get_mt5_wine64()
    metaeditor = get_mt5_metaeditor_exe()
    if wine_prefix is None or wine64 is None or metaeditor is None:
        raise RuntimeError(
            "Could not find the local MT5 Wine/MetaEditor installation. The exporter source was copied, but compilation could not start automatically."
        )

    log_file = target_file.with_suffix(".log")

    env = os.environ.copy()
    env["WINEPREFIX"] = str(wine_prefix)
    command = [
        str(wine64),
        str(metaeditor),
        f"/compile:{target_file.name}",
        f"/log:{log_file.name}",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(target_file.parent),
    )
    ex5_file = target_file.with_suffix(".ex5")
    log_text = ""
    if log_file.exists():
        try:
            log_text = log_file.read_text(encoding="utf-16le", errors="replace")
        except Exception:
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
    success = ex5_file.exists() and "0 errors, 0 warnings" in log_text
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "log_file": str(log_file),
        "compiled_file": str(ex5_file),
        "compiled_exists": ex5_file.exists(),
        "success": success,
        "log_text": log_text.strip(),
    }


def main() -> None:
    args = parse_args()
    install_result = install_exporter_source(args.source, relative_dir=args.relative_dir)

    print("=" * 70)
    print("MT5 EXPORTER INSTALL")
    print("=" * 70)
    print(f"Source : {install_result['source']}")
    print(f"Target : {install_result['target']}")

    compile_result = None
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
            if compile_result["stdout"]:
                print(compile_result["stdout"])
            if compile_result["stderr"]:
                print(compile_result["stderr"])

    print()
    print("Next in MT5:")
    print("  1. Open Navigator -> Expert Advisors -> OpenAI")
    print("  2. Attach XAUUSD_Live_Data_Exporter to an XAUUSD M1 chart")
    print("  3. Enable Algo Trading")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
