# XAUUSD AI Trading Bot

Validation-first XAUUSD pipeline for LightGBM training and MT5 ONNX inference.

## Current Status
- The checked-in `data/xauusd_m1_2022_2025.csv` is a one-day sample/demo dataset, not the full historical training set
- Fast local demo mode is available via `python src/run_sample_demo.py`
- Model contract is fixed to 68 ordered features
- MT5 integration is validation-first: feature computation, ONNX inference, fixture comparison, and signal logging
- Auto-trading, risk management, and Strategy Tester proof are still pending

## Fast Local Demo

Use this path if you want to regenerate visible outputs from the checked-in repo state right away. It rebuilds:
- `data/processed/xauusd_m1_standardized.csv`
- `data/processed/xauusd_m1_overlap.csv`
- `data/processed/xauusd_features.csv`
- `mt5_expert_advisor/Files/config/validation_set.csv`
- `mt5_expert_advisor/Files/models/xauusd_ai_v1.onnx`

### Demo prerequisites
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-onnx.txt
```

On macOS, `lightgbm` also needs `libomp`:
```bash
brew install libomp
```

### Run the one-command demo
```bash
python src/run_sample_demo.py
```

Optional:
```bash
python src/run_sample_demo.py --skip-onnx-runtime-check
```

### Open the local result dashboard
```bash
python src/serve_demo_ui.py
```

Then open `http://127.0.0.1:4174` in your browser.

If `TWELVEDATA_API_KEY` is set in your shell, the dashboard also shows a live `XAU/USD` market snapshot from Twelve Data. That live section is display-only because the `XAU/USD` feed does not include volume, while the local model pipeline expects volume-aware bars.

```bash
export TWELVEDATA_API_KEY=your_key_here
python src/serve_demo_ui.py
```

The Twelve Data helpers also fall back to a repo-local `.env` or `.env.local` file containing:
```bash
TWELVEDATA_API_KEY=your_key_here
```

Important:
- `Live Market Snapshot` uses current Twelve Data `XAU/USD` prices
- `Sample Demo Prediction` still comes from the checked-in sample artifact and is not computed from the live Twelve Data feed

Expected sample-data output on the checked-in repo state:
- standardized rows: about `1380`
- overlap rows: `240`
- feature rows: `1`
- validation fixture rows: `1`

## Live Twelve Data Mode

Use this path if you want the repo to run on current Twelve Data `XAU/USD` bars instead of the checked-in sample CSV.

```bash
export TWELVEDATA_API_KEY=your_key_here
python src/run_live_twelvedata_pipeline.py
```

This live mode fetches recent one-minute bars, writes them to `data/live/`, then runs:
- standardization
- overlap filtering
- feature engineering
- label creation
- LightGBM training on the fetched window
- confidence analysis
- approximate backtest generation
- MT5 validation fixture export
- ONNX export

Live outputs are written separately from the sample demo:
- `data/live/xauusd_twelvedata_raw.csv`
- `data/live/processed/xauusd_m1_standardized.csv`
- `data/live/processed/xauusd_m1_overlap.csv`
- `data/live/processed/xauusd_features.csv`
- `data/live/processed/xauusd_labeled.csv`
- `python_training/models/live_twelvedata/`
- `mt5_expert_advisor/Files/config/validation_set_live.csv`
- `mt5_expert_advisor/Files/models/xauusd_ai_live.onnx`

Important caveat:
- Twelve Data `XAU/USD` bars do not currently include volume
- the 68-feature contract in this repo depends on volume-aware features
- live mode therefore synthesizes volume before feature engineering
- treat live-mode predictions, retraining, confidence analysis, and approximate backtest results as experimental until you move to a feed with source volume or retrain a no-volume model explicitly

The default live fallback is:
```bash
python src/run_live_twelvedata_pipeline.py --volume-mode range_proxy
```

You can also force a flat neutral fallback:
```bash
python src/run_live_twelvedata_pipeline.py --volume-mode constant
```

## Dashboard Modes

The dashboard can now serve either the sample demo or the live Twelve Data pipeline.

Auto mode:
```bash
python src/serve_demo_ui.py
```

- if `TWELVEDATA_API_KEY` is set, the dashboard serves the live pipeline
- otherwise it serves the checked-in sample demo

Force sample mode:
```bash
python src/serve_demo_ui.py --pipeline-mode sample
```

Force live mode:
```bash
python src/serve_demo_ui.py --pipeline-mode live
```

If you want to reuse already-generated live outputs without refetching:
```bash
python src/serve_demo_ui.py --pipeline-mode live --skip-refresh
```

## Research Cockpit

Use this path if you want a research-first workspace around the current project instead of a single 15-minute classifier. The cockpit keeps the existing LightGBM path as baseline, adds multi-horizon research outputs, tradeable-setup labeling, structure overlays, session behavior analysis, calibrated probability review, paper-trading simulation, and a Streamlit UI.

### Install research extras
```bash
pip install -r requirements-research.txt
```

The neural stack is intentionally **phase 2** and disabled by default in the research config. The research pipeline therefore runs productively on the calibrated LightGBM baseline first.

### Run the research pipeline
```bash
python src/research_pipeline.py --skip-neural --input data/live/xauusd_twelvedata_raw.csv
```

If you already have proper broker or MT5 history with OHLCV or tick-volume, point the same command to that file instead:
```bash
python src/research_pipeline.py --skip-neural --input data/XAUUSD_1min.csv
```

Outputs are written to:
- `data/research/xauusd_m1_standardized.csv`
- `data/research/xauusd_research_ready.csv`
- `data/research/xauusd_research_labels.csv`
- `data/research/xauusd_research_overlays.json`
- `data/research/xauusd_research_predictions.csv`
- `data/research/xauusd_paper_ledger.csv`
- `data/research/xauusd_research_report.json`
- `python_training/models/research/baseline/`
- `python_training/models/research/patchtst/` only after the neural environment is re-enabled later

### Open the Streamlit cockpit
```bash
streamlit run streamlit_app.py
```

The cockpit is organized into five views:
- `Dashboard`: live snapshot, trade decision, gate status, blocked reasons, and recent paper trades
- `Chart / Structure`: candlesticks plus FVG, BOS, CHOCH, order block, entry, and trade-level overlays
- `Sessions / Psychology`: London, New York, Asia, overlap behavior, timing bias, session summary, and session-wise paper scoreboard
- `Model Lab`: calibration, trade precision, signal rate, and probability review
- `Backtest / Walk-Forward`: paper equity curve, expectancy in R, drawdown, blocked signals, and fold-by-fold walk-forward metrics

Important:
- this cockpit is evaluation-first and not an auto-trader
- Twelve Data is the operational v1 source for this profitability path
- the productive baseline uses Twelve-Data-compatible **core features**; legacy volume-proxy fields stay research-only context and are not the required model edge
- `83%` is not treated as a guaranteed headline target; the system is designed to show real out-of-sample evidence, calibration, and expectancy instead of hiding weak regimes
- if the neural stack is unavailable or disabled, the app and report stay on the LightGBM baseline and say so explicitly

## Full Retraining Path

Use this path only when you have a larger historical XAUUSD dataset available locally. The checked-in sample/demo CSV is not large enough to retrain the model end to end.

### 1. Install Python dependencies
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-onnx.txt  # only needed for ONNX export/runtime verification
```

### 2. Standardize and filter data
```bash
python src/validate_merged_data.py \
  --input data/xauusd_m1_2022_2025.csv \
  --output data/processed/xauusd_m1_standardized.csv

python src/filter_overlap.py \
  --input data/processed/xauusd_m1_standardized.csv \
  --output data/processed/xauusd_m1_overlap.csv
```

### 3. Engineer features and labels
```bash
python src/feature_engineering.py \
  --input data/processed/xauusd_m1_overlap.csv \
  --output data/processed/xauusd_features.csv \
  --feature-config python_training/config/features.yaml

python src/create_labels.py \
  --input data/processed/xauusd_features.csv \
  --output data/processed/xauusd_labeled.csv
```

If `create_labels.py` produces 0 labeled rows, the dataset is too small to support retraining with the current 68-feature contract and 15-minute forward-return labels.

### 4. Train and analyze
```bash
python src/train_lightgbm.py \
  --input data/processed/xauusd_labeled.csv \
  --feature-config python_training/config/features.yaml

python src/analyze_confidence.py \
  --input data/processed/xauusd_labeled.csv \
  --model python_training/models/lightgbm_xauusd_v1.txt

python src/backtest_simple.py \
  --input data/processed/xauusd_labeled.csv \
  --model python_training/models/lightgbm_xauusd_v1.txt
```

`src/backtest_simple.py` is deliberately labeled as an approximate signal simulation. It is not a trading-grade backtest.

### 5. Export MT5 artifacts
```bash
python src/export_to_onnx_simple.py \
  --model python_training/models/lightgbm_xauusd_v1.txt \
  --feature-list python_training/models/feature_list.json \
  --output mt5_expert_advisor/Files/models/xauusd_ai_v1.onnx

python src/export_mt5_validation_set.py \
  --input data/processed/xauusd_m1_overlap.csv \
  --model python_training/models/lightgbm_xauusd_v1.txt \
  --output mt5_expert_advisor/Files/config/validation_set.csv
```

## MT5 Validation Mode
- EA path: `mt5_expert_advisor/XAUUSD_AI_Bot.mq5`
- Default model path: `models\\xauusd_ai_v1.onnx`
- Validation fixture: `config\\validation_set.csv`
- Signal log: `logs\\xauusd_ai_signals.csv`
- Closed-bar inference only: the EA computes features from `shift=1`
- Trading is not implemented in this recovery pass

## Project Structure
```text
data/
  xauusd_m1_2022_2025.csv
  processed/
src/
  run_sample_demo.py
  validate_merged_data.py
  filter_overlap.py
  feature_engineering.py
  create_labels.py
  train_lightgbm.py
  export_to_onnx_simple.py
  export_mt5_validation_set.py
  pipeline_contract.py
python_training/
  config/features.yaml
  models/
mt5_expert_advisor/
  XAUUSD_AI_Bot.mq5
  FeatureEngine.mqh
  Features_*.mqh
  Files/config/
```

## Known Gaps
- Full retraining is not reproducible from the checked-in one-day sample dataset alone
- No verified MT5 Strategy Tester report in-repo yet
- No order execution or `RiskManager.mqh` yet
- Existing historical metrics in older artifacts should be treated as model-development outputs, not proof of MT5 live-readiness
