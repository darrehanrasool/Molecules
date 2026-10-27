# Research Directory Architecture & Guide

This document outlines the detailed folder layout, responsibility domains, and lifecycle of assets within the **Enveda CASMI 2026 - Molecules** research platform.

```
Molecule/
├── .github/workflows/          # GitHub Actions CI/CD test automation
├── configs/                    # YAML configuration files for deterministic execution
│   ├── project.yaml            # High-level project, authorship, and path variables
│   ├── data.yaml               # Preprocessing, m/z ranges, and database paths
│   ├── models.yaml             # Model hyperparameters and embedding dimensions
│   ├── training.yaml           # Training schedules, learning rates, loss functions
│   ├── hardware.yaml           # Resource allocation and Kaggle constraints
│   └── visualization.yaml      # Matplotlib/Seaborn themes and color palettes
├── data/                       # Tiered data architecture
│   ├── raw/                    # Untouched competition datasets
│   ├── interim/                # Intermediate parsed and filtered spectra
│   ├── processed/              # Normalized, binned, and tokenized datasets
│   ├── splits/                 # K-Fold train/validation/test partitions
│   ├── external/               # Public reference spectral libraries (MassBank, GNPS, HMDB)
│   └── manifests/              # MD5/SHA256 checksums and dataset registries
├── docs/                       # Comprehensive scientific and software documentation
│   ├── architecture.md         # End-to-end ML pipeline and inference architecture
│   ├── methodology.md          # Mathematical formulas and algorithmic foundations
│   ├── metric_evaluation.md    # Mean Reciprocal Rank @ 25 and InChIKey14 protocol
│   ├── scientific_background.md# Mass spectrometry physics, ionization, and fragmentation
│   ├── dataset_documentation.md# LC-MS/MS data schemas and adduct transformations
│   ├── data_dictionary.md      # Detailed column definitions and data types
│   ├── model_card.md           # Model capabilities, limitations, and intended uses
│   ├── reproducibility.md      # Deterministic seed controls and hardware specifications
│   ├── installation.md         # Environment setup and dependency installation
│   └── troubleshooting.md      # Common failure modes and debugging steps
├── experiments/                # Controlled experiment tracking
│   ├── configs/                # Per-experiment parameter configurations
│   ├── results/                # Recorded evaluation metrics and logs
│   └── experiment_registry.csv # Unified audit log of all experimental runs
├── models/                     # Specialized model directories
│   ├── classical_ml/           # Precursor-filtered similarity baselines
│   ├── deep_learning/          # PyTorch neural spectral encoders (Spec2Vec, MassFormer)
│   ├── retrieval/              # Vector similarity search engines
│   ├── generative/             # Autoregressive SMILES generators
│   ├── ensemble/               # Consensus rankers and MRR optimizers
│   └── explainability/         # Substructure attribution and peak assignment
├── notebooks/                  # Numbered reproducible Jupyter research notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_spectral_preprocessing.ipynb
│   ├── 03_metric_and_canonicalization.ipynb
│   ├── 04_baseline_models.ipynb
│   └── 05_submission_pipeline.ipynb
├── outputs/                    # Output artifacts
│   ├── predictions/            # Generated submission files (`submission.csv`)
│   ├── trained_models/         # Serialized model weights and checkpoints
│   └── inference_reports/      # Latency, memory, and performance audits
├── papers/                     # Academic literature repository
│   ├── pdf/                    # Key reference papers
│   ├── metadata/               # `papers_metadata.json` catalog
│   ├── literature_reviews/     # Literature reviews and survey documents
│   ├── methodology_extraction/ # Extracted equations, algorithms, and loss functions
│   ├── baseline_comparisons/   # Published vs. local benchmark comparison matrix
│   └── research_gaps/          # Identified gaps in natural product MS/MS identification
├── reports/                    # Formal evaluation reports
│   ├── evaluation/             # Cross-validation performance evaluations
│   ├── figures/                # High-resolution vector figures and plots
│   ├── model_cards/            # Exported governance and model cards
│   ├── research/               # Formal research summaries
│   ├── model_comparison.md     # Markdown comparative benchmark table
│   └── model_comparison.csv    # Raw comparative performance metric records
├── scripts/                    # Automation and CLI utility scripts
│   ├── setup_environment.py    # Directory verification and dependency checking
│   ├── download_competition_data.py # Automated Kaggle dataset downloader
│   ├── verify_hardware.py      # Hardware profiling (CPU, RAM, GPU, Apple Silicon MPS)
│   ├── preprocess_spectra.py   # Spectrum cleaning, noise filtering, and binning
│   ├── create_splits.py        # Scaffold-based cross-validation splitter
│   ├── train_baseline.py       # Baseline model training and evaluation script
│   ├── evaluate_metric.py      # Standalone MRR@25 calculation CLI
│   ├── generate_submission.py  # Kaggle submission.csv generator and validator
│   └── run_all_pipeline.sh     # One-click end-to-end execution script
├── src/                        # Modular production Python package
│   └── molecules/
│       ├── __init__.py
│       ├── config.py           # Typed settings and environment paths
│       ├── data/               # Data loaders, preprocessors, spectral features
│       ├── evaluation/         # MRR@25 and InChIKey14 canonicalization
│       ├── models/             # PyTorch models, baselines, similarity metrics
│       ├── postprocessing/     # Deduplication, reranking, and CSV submission builder
│       └── utils/              # Chemical utilities, IO operations, logging
├── tests/                      # Automated test suite
│   ├── unit/                   # Unit tests for functions and classes
│   ├── integration/            # End-to-end integration tests
│   └── fixtures/               # Mock data files and test fixtures
├── visualizations/             # High-resolution generated plots
│   ├── data_exploration/
│   ├── spectral_analysis/
│   ├── training/
│   ├── validation/
│   └── comparisons/
├── logs/                       # Application, training, and hardware logs
├── checkpoints/                # Model checkpoint weights
├── CITATION.cff                # Formal academic citation file
├── Makefile                    # Developer CLI commands
├── pyproject.toml              # Build system and package metadata
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
└── pytest.ini                  # Pytest test configuration
```
