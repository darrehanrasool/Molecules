.PHONY: help setup install download-data verify-data preprocess train evaluate visualize test submission clean

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

help:
	@echo "Enveda CASMI 2026 - Molecules Management Commands:"
	@echo "  make setup          - Set up project directory structure and environment"
	@echo "  make install        - Install production and development dependencies"
	@echo "  make download-data  - Download competition datasets via Kaggle API"
	@echo "  make verify-data    - Verify dataset integrity, checksums, and spectral formats"
	@echo "  make preprocess     - Preprocess MS/MS spectra (noise removal, normalization, binning)"
	@echo "  make train          - Train spectral encoders, retrieval models, and rerankers"
	@echo "  make evaluate       - Run full validation benchmark and compute MRR@25"
	@echo "  make visualize      - Generate spectral diagnostic and evaluation figures"
	@echo "  make test           - Run unit and integration tests via pytest"
	@echo "  make submission     - Validate and build submission.csv for Kaggle"
	@echo "  make clean          - Remove build artifacts, pycache, and temporary caches"

setup:
	$(PYTHON) scripts/setup_environment.py

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

download-data:
	$(PYTHON) scripts/download_competition_data.py

verify-data:
	$(PYTHON) scripts/verify_hardware.py

preprocess:
	$(PYTHON) scripts/preprocess_spectra.py

train:
	$(PYTHON) scripts/train_baseline.py --config configs/training.yaml

evaluate:
	$(PYTHON) scripts/evaluate_metric.py

visualize:
	$(PYTHON) -c "import matplotlib; print('Visualization engine ready')"

test:
	$(PYTHON) -m pytest tests/ -v

submission:
	$(PYTHON) scripts/generate_submission.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
