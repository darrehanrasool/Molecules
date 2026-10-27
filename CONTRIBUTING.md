# Contributing to Enveda CASMI 2026 - Molecules

Thank you for your interest in contributing to our research repository for the Enveda CASMI 2026 challenge!

## Development Guidelines

1. **Deterministic Reproducibility**:
   - Fix all random seeds (`SEED=42`).
   - Log all hyperparameters in `configs/` or `experiments/experiment_registry.csv`.

2. **RDKit Tautomer Canonicalization Compliance**:
   - All evaluation and candidate filtering must adhere to RDKit 2026.03.3 tautomer canonicalization.
   - Predictions are compared using InChIKey14 (first 14 characters).

3. **Code Quality**:
   - Format with `black` and lint with `flake8`.
   - Maintain unit tests for any new algorithms or data processing pipelines in `tests/`.
   - Ensure `pytest tests/ -v` passes before opening pull requests.

4. **Kaggle Notebook Compatibility**:
   - All final inference pipelines must be self-contained and run within the 9-hour execution limit without active internet connections.
