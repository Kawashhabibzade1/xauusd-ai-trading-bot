# XAUUSD AI Trading Bot

Validation-first XAUUSD pipeline for LightGBM training and MT5 ONNX inference.

## Current Status
- The checked-in `data/xauusd_m1_2022_2025.csv` is a one-day sample/demo dataset, not the full historical training set
- Fast local demo mode is available via `python src/run_sample_demo.py`
- Model contract is fixed to 68 ordered features
- MT5 integration is validation-first: feature computation, ONNX inference, fixture comparison, and signal logging
- Experimental MT5 demo-account auto-trading is available in the EA; broader live-risk validation and Strategy Tester proof are still pending

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

If `OANDA_API_TOKEN`, `TWELVEDATA_API_KEY`, or a local MT5 terminal is available, the dashboard can also show a live `XAU/USD` market snapshot. MT5 Local is the best path for this project because it can provide local broker bars with `tick_volume`; OANDA is next-best, and Twelve Data remains the fallback display source.

```bash
export TWELVEDATA_API_KEY=your_key_here
python src/serve_demo_ui.py
```

The market-data helpers also fall back to a repo-local `.env` or `.env.local` file containing:
```bash
OANDA_API_TOKEN=your_oanda_token_here
OANDA_ENV=practice
TWELVEDATA_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
MT5_TERMINAL_PATH=/path/to/terminal64.exe
```

You do not need to put MT5 credentials into the project if your MetaTrader terminal is already open and logged in locally. In that case, the MT5 live path can often attach to the running terminal directly.

Important:
- `Live Market Snapshot` can use MT5 Local, OANDA, or Twelve Data
- `Auto` in the Streamlit cockpit now tries `MT5 Local` first when your terminal is running on the same machine
- `MT5 Local` remains the preferred live provider when a local MetaTrader terminal is available
- `Sample Demo Prediction` still comes from the checked-in sample artifact and is not computed from the live market feed

## Live MT5 Mode

Use this path if you want the repo to run on current bars from your local MetaTrader 5 terminal.

```bash
python src/install_mt5_exporter.py
python src/run_live_mt5_pipeline.py
```

If your MT5 terminal is already open and logged in, no credentials are required in the command. The runner will try to connect to the local terminal first.

If the direct `MetaTrader5` Python bridge is not available on your machine, the runner automatically falls back to the MT5 exporter CSV written by the bundled exporter EA.

Optional environment variable:
```bash
MT5_TERMINAL_PATH=/path/to/terminal
```

Preferred macOS flow:
```bash
python src/install_mt5_exporter.py
python src/mt5_keychain_cli.py store --login 12345678 --server MetaQuotes-Demo
python src/mt5_keychain_cli.py status
python src/mt5_keychain_cli.py run-live-pipeline -- --source-mode export
```

The `store` command prompts for the MT5 password securely and saves the login, password, and server in macOS Keychain. The secure pipeline wrappers load those values only for the current Python process, so you do not need to put MT5 secrets into `.env`, `.env.local`, or shell history.

MT5 live outputs are written separately:
- `data/live/xauusd_mt5_raw.csv`
- `data/live/mt5_processed/xauusd_m1_standardized.csv`
- `data/live/mt5_processed/xauusd_m1_overlap.csv`
- `data/live/mt5_processed/xauusd_features.csv`
- `data/live/mt5_processed/xauusd_labeled.csv`
- `python_training/models/live_mt5/`
- `mt5_expert_advisor/Files/config/validation_set_mt5_live.csv`
- `mt5_expert_advisor/Files/models/xauusd_ai_mt5_live.onnx`
- `data/live/live_mt5_pipeline_report.json`

When the local MT5 `MQL5/Files` folder is available, this runner now also syncs the generated MT5 bot artifacts there automatically so the EA can load the newest live model without manual file copying.

Important caveat:
- this path is local-only and depends on a MetaTrader terminal on the same machine
- the Python `MetaTrader5` package is environment/platform dependent, so this path is intended for your local trading machine, not Streamlit Community Cloud
- on macOS, the `MT5_Live_Data_Exporter.mq5` fallback is often the more reliable path
- the exporter writes `xauusd_mt5_live.csv` into your local MT5 `MQL5/Files` folder, and the Python runner reads that file automatically
- by default the pipeline uses `tick_volume` as the live `volume` field; `real_volume` can be preferred only when your broker/feed actually provides it

## MT5 Demo Auto-Trading

Use this path if you want the MT5 Expert Advisor to place demo-account orders automatically from the ONNX model.

Safety defaults:
- demo-account trading only; broker execution is hard-locked to demo accounts in the EA
- validation mode stays on by default
- one managed position at a time per symbol/magic number
- ATR-based stop loss and fixed risk-reward take profit
- spread and confidence guards before order submission

### 1. Generate and sync the live MT5 model artifacts
```bash
python src/install_mt5_exporter.py
python src/mt5_keychain_cli.py store --login 12345678 --server MetaQuotes-Demo
python src/mt5_keychain_cli.py run-research-pipeline -- --source-mode export
```

One-command prep if your Keychain entry already exists:
```bash
./scripts/prepare_mt5_demo_rollout.sh --source-mode export
```

Optional manual resync if you want to push the repo artifacts into MT5 again:
```bash
python src/sync_mt5_bot_artifacts.py --variant live_mt5_research
```

### 2. Install and compile the trading EA
```bash
python src/install_mt5_bot.py
```

### 3. Attach the EA in MetaTrader 5
Attach `XAUUSD_AI_Bot` to a live `XAUUSD` or `XAUUSD-*` `M1` chart on your demo account.

Important chart setup:
- keep `MT5_Live_Data_Exporter` on its own separate `XAUUSD` `M1` chart if you want the local Streamlit live price, timeframe ribbon, or MT5 paper-research worker to stay fresh
- attach `XAUUSD_AI_Bot` to a different `XAUUSD` `M1` chart
- MT5 allows only one Expert Advisor per chart, so attaching `XAUUSD_AI_Bot` onto the same chart will remove the exporter and freeze `xauusd_mt5_live.csv`

Use these EA inputs:
- `InpValidationMode=false`
- `InpEnableDemoTrading=true`
- `InpDemoOnly=true`
- `InpRequireTradeDirective=true`
- `InpMaxDirectiveEntryDriftPoints=30`
- `InpSessionTradeLimit=1`
- `InpModelName=models\\xauusd_ai_mt5_live.onnx`

Recommended starting defaults:
- `InpFixedLotSize=0.01`
- `InpConfidenceThresh=0.55`
- `InpStopAtrMultiple=1.00`
- `InpTakeProfitRR=1.50`

Important:
- keep this on a demo account first
- enable `Algo Trading` in MT5
- the EA writes signal logs to `logs\\xauusd_ai_signals.csv` and trade logs to `logs\\xauusd_ai_trades.csv`
- the EA follows the paper/research trade directive, so MT5 demo trades use the same gated setup source as the paper path
- the EA blocks entries when the live market price drifts too far from the paper directive entry, so demo execution stays close to the mirrored paper setup
- the EA caps demo order size at `0.01` lots and, with `InpSessionTradeLimit=1`, will place only one successful demo trade for this proof run
- Streamlit and learning stay on the paper/research artifacts only; MT5 broker-side trade logs are not ingested there
- this path is still experimental and should be validated in MT5 Strategy Tester before any non-demo use

Verification checklist before first accepted order:
- EA initializes successfully on the `XAUUSD` `M1` chart
- non-demo accounts are rejected when `InpDemoOnly=true`
- `logs\\xauusd_ai_signals.csv` updates on eligible overlap bars
- `logs\\xauusd_ai_trades.csv` records `SKIP` or `BLOCK` reasons before any accepted order
- wide spreads, low confidence, and opposite-position flips are blocked according to the configured guards

## Local MT5 Paper Trading Stack

Use this path if you want the whole project to work locally on your PC with:
- live MT5 exporter data
- a continuously refreshed local paper-trading worker
- the Streamlit research cockpit

Prerequisites:
- desktop MetaTrader 5 is open on the same machine
- `MT5_Live_Data_Exporter` is attached to a live `XAUUSD` or `XAUUSD-*` `M1` chart
- if `XAUUSD_AI_Bot` is also running, keep it on a separate `XAUUSD` `M1` chart so the exporter is not removed
- `Algo Trading` is enabled

Start everything:
```bash
./scripts/start_mt5_local_stack.sh
```

Then open:
```text
http://127.0.0.1:8502
```

Stop everything:
```bash
./scripts/stop_mt5_local_stack.sh
```

The local MT5 paper-trading stack writes its own cockpit artifacts here:
- `data/live/research_mt5/xauusd_research_report.json`
- `data/live/research_mt5/xauusd_research_predictions.csv`
- `data/live/research_mt5/xauusd_paper_ledger.csv`
- `data/live/research_mt5/xauusd_learning_feedback.csv`
- `data/live/research_mt5/learning_status.json`
- `data/live/research_mt5/xauusd_research_overlays.json`

Important:
- this is paper trading only; it does not place broker orders
- the MT5 worker and the dashboard are local-runtime services, not Streamlit Cloud replacements
- the top cards in the cockpit now follow the MT5 local report when `Live provider` is `MT5 Local` or `Auto`

## Safe Learning Mode

The local MT5 stack now separates:
- `live scoring`
- `paper-trade feedback capture`
- `batch retraining readiness`
- `model promotion`

It does **not** retrain a neural net after every single paper trade. That would overfit very quickly.

Instead, the project now writes:
- `data/live/research_mt5/xauusd_learning_feedback.csv`
  This stores the current signal context plus any realized paper-trade outcome.
- `data/live/research_mt5/learning_status.json`
  This tells you whether there is enough diverse feedback for safe batch retraining.

Current learning safety checks include:
- minimum closed trades
- minimum number of trading days
- minimum long and short trade counts
- session concentration guard
- direction concentration guard
- recent-sample concentration guard

Future candidate models should be checked with:
```bash
python src/evaluate_mt5_model_promotion.py --candidate-metrics path/to/candidate_metrics.json
```

The promotion gate only approves a candidate when it clears the configured floors for:
- trade count
- precision
- profit factor
- expectancy
- drawdown

and does not regress too far versus the current MT5 local baseline.

## Live OANDA Mode

Use this path if you want the repo to run on current OANDA `XAU_USD` candles instead of the checked-in sample CSV or the Twelve Data feed.

```bash
export OANDA_API_TOKEN=your_oanda_token_here
export OANDA_ENV=practice
python src/run_live_oanda_pipeline.py
```

This live mode fetches recent one-minute OANDA candles, writes them to `data/live/`, then runs:
- standardization
- overlap filtering
- feature engineering
- label creation
- LightGBM training on the fetched window
- confidence analysis
- approximate backtest generation
- MT5 validation fixture export
- ONNX export

OANDA live outputs are written separately from the Twelve Data variant:
- `data/live/xauusd_oanda_raw.csv`
- `data/live/oanda_processed/xauusd_m1_standardized.csv`
- `data/live/oanda_processed/xauusd_m1_overlap.csv`
- `data/live/oanda_processed/xauusd_features.csv`
- `data/live/oanda_processed/xauusd_labeled.csv`
- `python_training/models/live_oanda/`
- `mt5_expert_advisor/Files/config/validation_set_oanda_live.csv`
- `mt5_expert_advisor/Files/models/xauusd_ai_oanda_live.onnx`
- `data/live/live_oanda_pipeline_report.json`

Important caveat:
- OANDA candle volume is the number of price updates in a candle, not centralized exchange gold volume
- it is still materially better for this project than a no-volume spot feed because the current feature contract expects a live volume input
- treat short-window retraining and backtest outputs as recent-window research, not as final proof of profitability

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

The dashboard can now serve the sample demo plus live market snapshots from MT5 Local, OANDA, or Twelve Data.

Auto mode:
```bash
python src/serve_demo_ui.py
```

- if `MT5 Local` is selected, the cockpit tries to pull directly from your running MetaTrader terminal
- if `OANDA_API_TOKEN` is set, the cockpit can use OANDA for the live snapshot
- otherwise it can still fall back to Twelve Data
- if neither is set, it serves the checked-in sample demo only

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
- MT5 Local is the preferred live source when available on the same machine; OANDA is the next-best fallback, and Twelve Data remains the display-only fallback
- the productive baseline uses API-compatible **core features**; legacy volume-proxy fields stay research-only context and are not the required model edge
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
