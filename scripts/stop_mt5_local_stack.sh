#!/usr/bin/env bash
set -euo pipefail

pkill -f "src/run_mt5_research_worker.py" || true
pkill -f "streamlit run streamlit_app.py --server.headless true --server.port 8502" || true

echo "MT5 local stack stopped."
