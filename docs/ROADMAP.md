# Development Roadmap

## Phase 1: Contract And Data
- [x] Canonical raw dataset selected: `data/xauusd_m1_2022_2025.csv`
- [x] Standardization path moved to `data/processed/xauusd_m1_standardized.csv`
- [x] London-NY overlap filter standardized
- [x] 68-feature contract captured in `python_training/config/features.yaml`

## Phase 2: Python Training Pipeline
- [x] Feature engineering script aligned to the 68-feature contract
- [x] Label creation aligned to the shared contract
- [x] LightGBM training aligned to the shared contract
- [x] Confidence analysis no longer crashes on fallback
- [x] Approximate simulation explicitly marked as non-executable

## Phase 3: MT5 Validation Integration
- [x] Feature engine expanded to 68 features
- [x] Validation-mode EA inputs added
- [x] ONNX model path standardized
- [x] Signal logging added
- [x] Validation fixture export added
- [ ] Compile and run parity checks in MT5

## Phase 4: Deferred Until Parity Is Proven
- [ ] Order execution
- [ ] Risk management module
- [ ] Strategy Tester validation
- [ ] Shadow trading
- [ ] Live deployment
