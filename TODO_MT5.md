# TODO: MT5 Demo Autopilot Hardening

## Goal
Keep MT5 demo execution validation-first while hardening the autopilot, telemetry, and live-readiness gates before any live-account rollout work.

## Implemented In This Pass
- [x] Shared 68-feature contract established on the Python side
- [x] ONNX model path standardized to `models\\xauusd_ai_v1.onnx`
- [x] Validation-mode EA inputs added
- [x] Feature engine expanded to the 68-feature contract
- [x] CSV signal logging added
- [x] Validation fixture export script added
- [x] Demo-account broker order execution exists in `XAUUSD_AI_Bot.mq5`
- [x] Trade-directive mirroring from the paper/research pipeline exists
- [x] Broker autopilot directive fields now include `directive_id`, `broker_trading_enabled`, and `broker_block_reason`
- [x] EA heartbeat/state file now writes `config\\mt5_execution_state.csv`
- [x] Broker-side `OPEN`, `CLOSE`, and `BREAKEVEN` events are ingested into the Python report and worker notifications
- [x] Demo autopilot can now disarm on stale runtime health, broker daily trade caps, consecutive losses, and daily realized loss limits
- [x] `live_ready` is now tracked separately from broker execution enablement

## Still Pending

### Validation And Compile Proof
- [ ] Compile EA in MetaEditor and resolve any MQL5 compile errors
- [ ] Generate fresh ONNX artifact from the repaired Python pipeline
- [ ] Generate fresh `validation_set.csv`
- [ ] Run MT5 fixture-parity validation and record mismatches
- [ ] Confirm probability drift stays within the configured tolerances

### Demo Autopilot Verification
- [ ] Confirm `mt5_execution_state.csv` updates every timer cycle on the live chart
- [ ] Confirm a healthy READY directive opens exactly one demo trade and is not retried after `OPEN` or `OPEN_REJECT`
- [ ] Confirm natural broker-side closes are logged as `CLOSE` with the correct inferred exit reason
- [ ] Confirm stale exporter, stale account snapshot, and stale EA heartbeat each disable new entries within the configured threshold
- [ ] Confirm `OPEN`, `CLOSE`, `BREAKEVEN`, `AUTOPILOT_DISABLED`, and `AUTOPILOT_RESTORED` notifications dedupe correctly

### Rollout Proof Still Missing
- [ ] MT5 Strategy Tester report
- [ ] Python-vs-MT5 parity report
- [ ] Shadow-trading evidence
- [ ] Live-account rollout checklist with manual promotion approval

## Outputs Expected From The Next Validation Session
1. Compiled `XAUUSD_AI_Bot.ex5`
2. Fresh `mt5_expert_advisor/Files/models/xauusd_ai_v1.onnx`
3. Fresh `mt5_expert_advisor/Files/config/validation_set.csv`
4. `MQL5/Files/logs/xauusd_ai_signals.csv`
5. `MQL5/Files/config/mt5_execution_state.csv`
6. A short parity summary with mismatched features if any
