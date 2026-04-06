#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/.logs"
mkdir -p "$LOG_DIR"

PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
STREAMLIT_BIN="${STREAMLIT_BIN:-$ROOT_DIR/.venv/bin/streamlit}"

if [[ ! -x "$PY_BIN" ]]; then
  echo "Missing Python runtime at $PY_BIN"
  echo "Create the virtualenv first: python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt -r requirements-research.txt"
  exit 1
fi

if [[ ! -x "$STREAMLIT_BIN" ]]; then
  echo "Missing Streamlit runtime at $STREAMLIT_BIN"
  exit 1
fi

pkill -f "src/run_mt5_research_worker.py" || true
pkill -f "streamlit run streamlit_app.py --server.headless true --server.port 8502" || true

nohup env PYTHONPATH=src "$PY_BIN" -u src/run_mt5_research_worker.py --poll-seconds 30 > "$LOG_DIR/mt5_worker.log" 2>&1 &
WORKER_PID=$!

nohup env PYTHONPATH=src "$STREAMLIT_BIN" run streamlit_app.py --server.headless true --server.port 8502 > "$LOG_DIR/streamlit.log" 2>&1 &
STREAMLIT_PID=$!

echo "MT5 local stack started."
echo "Worker PID    : $WORKER_PID"
echo "Streamlit PID : $STREAMLIT_PID"
echo "App URL       : http://127.0.0.1:8502"
echo "Worker log    : $LOG_DIR/mt5_worker.log"
echo "Streamlit log : $LOG_DIR/streamlit.log"
