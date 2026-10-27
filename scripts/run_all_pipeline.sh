#!/usr/bin/env bash
# End-to-end execution pipeline for Enveda CASMI 2026

set -e

echo "=========================================================="
echo "Enveda CASMI 2026: Molecules Full Execution Pipeline"
echo "Centre of Artificial Intelligence, IUST Kashmir"
echo "=========================================================="

# 1. Environment & Hardware Verification
echo "[Stage 1/5] Verifying environment & hardware compute..."
python3 scripts/verify_hardware.py

# 2. Run Test Suite
echo "[Stage 2/5] Running Pytest test suite..."
PYTHONPATH=src pytest tests/ -v

# 3. Spectral Preprocessing
echo "[Stage 3/5] Preprocessing spectra..."
PYTHONPATH=src python3 scripts/preprocess_spectra.py

# 4. Generate Splits & Baseline Benchmark
echo "[Stage 4/5] Running cross-validation splits and baseline benchmark..."
PYTHONPATH=src python3 scripts/create_splits.py
PYTHONPATH=src python3 scripts/train_baseline.py

# 5. Build and Verify Kaggle Submission
echo "[Stage 5/5] Generating and validating competition submission.csv..."
PYTHONPATH=src python3 scripts/generate_submission.py --input outputs/predictions/sample_submission.csv --output outputs/predictions/submission.csv

echo "=========================================================="
echo "[✓] Full Pipeline Completed Successfully."
echo "=========================================================="
