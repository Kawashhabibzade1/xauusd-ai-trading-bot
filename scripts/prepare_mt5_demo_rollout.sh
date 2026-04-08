#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PY_BIN" ]]; then
  echo "Missing Python runtime at $PY_BIN"
  echo "Create the virtualenv first: python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt -r requirements-onnx.txt"
  exit 1
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

"$PY_BIN" src/install_mt5_exporter.py
"$PY_BIN" src/mt5_keychain_cli.py run-research-pipeline -- "$@"
"$PY_BIN" src/install_mt5_bot.py

echo
echo "Next manual MT5 step:"
echo "  1. Log into your demo account in the MT5 terminal"
echo "  2. Open XAUUSD or XAUUSD-* on M1"
echo "  3. Attach OpenAI -> XAUUSD_AI_Bot"
echo "  4. Load the demo-safe inputs printed by src/install_mt5_bot.py"
echo "  5. Enable Algo Trading after the EA initializes cleanly"
echo
echo "Keep the MT5 research worker running for fresh paper-trade directives:"
echo "  ./.venv/bin/python src/run_mt5_research_worker.py --poll-seconds 5"
