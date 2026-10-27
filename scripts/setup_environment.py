#!/usr/bin/env python3
"""Environment setup and directory structure verification."""

import sys
from pathlib import Path

REQUIRED_DIRS = [
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/splits",
    "data/external",
    "data/manifests",
    "docs",
    "experiments/configs",
    "experiments/results",
    "models/classical_ml",
    "models/deep_learning",
    "models/retrieval",
    "models/generative",
    "models/ensemble",
    "models/explainability",
    "notebooks",
    "outputs/predictions",
    "outputs/trained_models",
    "outputs/inference_reports",
    "papers/pdf",
    "papers/metadata",
    "papers/literature_reviews",
    "papers/methodology_extraction",
    "papers/baseline_comparisons",
    "papers/research_gaps",
    "reports/evaluation",
    "reports/figures",
    "reports/model_cards",
    "reports/research",
    "scripts",
    "src/molecules",
    "tests/unit",
    "tests/integration",
    "visualizations",
    "logs/application",
    "logs/training",
    "logs/hardware",
    "checkpoints/best",
    "checkpoints/latest",
]


def main():
    root = Path(__file__).resolve().parent.parent
    print(f"[*] Initializing Enveda CASMI 2026 Environment at: {root}")

    for d in REQUIRED_DIRS:
        target = root / d
        target.mkdir(parents=True, exist_ok=True)

    print(f"[+] All {len(REQUIRED_DIRS)} research directories verified successfully.")


if __name__ == "__main__":
    main()
