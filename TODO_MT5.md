# TODO: MT5 Validation-First Recovery

## Goal
Bring the MT5 side to deterministic validation mode before any live order execution work.

## Implemented In This Pass
- [x] Shared 68-feature contract established on the Python side
- [x] ONNX model path standardized to `models\\xauusd_ai_v1.onnx`
- [x] Validation-mode EA inputs added
- [x] Feature engine expanded to the 68-feature contract
- [x] CSV signal logging added
- [x] Validation fixture export script added

## Still Pending

### Validation Hardening
- [ ] Compile EA in MetaEditor and resolve any MQL5 compile errors
- [ ] Generate fresh ONNX artifact from the repaired Python pipeline
- [ ] Generate fresh `validation_set.csv`
- [ ] Run MT5 fixture-parity validation and record mismatches
- [ ] Confirm probability drift stays within the configured tolerances

### Trading Features Deferred On Purpose
- [ ] Add order execution logic
- [ ] Add `RiskManager.mqh`
- [ ] Add ATR stop loss / trailing stop
- [ ] Add daily trade caps and drawdown kill switch

### Proof Still Missing
- [ ] MT5 Strategy Tester report
- [ ] Python-vs-MT5 parity report
- [ ] Shadow-trading evidence
- [ ] Live deployment checklist

## Outputs Expected From The Next Validation Session
1. Compiled `XAUUSD_AI_Bot.ex5`
2. Fresh `mt5_expert_advisor/Files/models/xauusd_ai_v1.onnx`
3. Fresh `mt5_expert_advisor/Files/config/validation_set.csv`
4. `MQL5/Files/logs/xauusd_ai_signals.csv`
5. A short parity summary with mismatched features if any
