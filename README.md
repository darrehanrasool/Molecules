# Enveda CASMI 2026: Molecule ID From Mass Spectra

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg?logo=pytest)](https://github.com/darrehanrasool/Molecules/actions)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Enveda%20CASMI%202026-blue.svg?logo=kaggle)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![RDKit 2026.03.3](https://img.shields.io/badge/RDKit-2026.03.3-green.svg)](https://www.rdkit.org/)

An enterprise-grade, modular, research-driven machine learning repository for the **Critical Assessment of Small Molecule Identification (CASMI) 2026** challenge hosted by **Enveda Biosciences** and **Kaggle**. This project predicts 2D chemical structures (SMILES strings) of unknown small molecules and natural products directly from Liquid Chromatography Tandem Mass Spectrometry (LC-MS/MS) spectra.

---

## Institutional Affiliation & Research Team

This research project is designed, engineered, and maintained at the **Centre of Artificial Intelligence (CAI)** and the **Department of Computer Science & Engineering**, **Islamic University of Science and Technology (IUST), Kashmir**.

| Role                            | Name                       | Designation & Affiliation                                                               | Research & Project Contribution                                                                            |
| :------------------------------ | :------------------------- | :-------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| **Project Instructor & Mentor** | **Dr. Asif Assad**         | Faculty Supervisor & Project Mentor, Department of Computer Science & Engineering, IUST | Research Conceptualization, Algorithmic Guidance, Experimental Design & Academic Supervision               |
| **Contributing Researcher**     | **Dar Rehan Rasool**       | Graduate Researcher, M.Tech AI, Centre of Artificial Intelligence, IUST                 | Spectral Feature Engineering, Neural Retrieval Architecture, Metric Optimization & Pipeline Implementation |
| **Contributing Researcher**     | **Muneeb Syed Shah**       | Graduate Researcher, M.Tech AI, Centre of Artificial Intelligence, IUST                 | Spectral Preprocessing, Precursor Mass Tolerancing, Benchmark Evaluations & Validation                     |
| **Contributing Researcher**     | **Shafiya Bashir Pampori** | Graduate Researcher, M.Tech AI, Centre of Artificial Intelligence, IUST                 | RDKit Tautomer Canonicalization, InChIKey14 Verification, Data Integrity & Submission Auditing             |

- **Academic Unit**: Centre of Artificial Intelligence (CAI) & Department of Computer Science and Engineering
- **Institution**: Islamic University of Science and Technology (IUST), Awantipora, Jammu & Kashmir, India

---

## Executive Challenge Overview

Modern mass spectrometry detects tens of thousands of chemical signals in biological and natural product extracts. While reference spectral libraries catalog known synthetic compounds, the vast majority of natural products remain **dark chemical matter**—novel chemical structures never previously synthesized or cataloged.

The goal of this competition is to build machine learning models capable of predicting 2D chemical structures represented as **SMILES** (Simplified Molecular Input Line Entry System) strings from tandem mass spectra (`LC-MS/MS`).

### Tandem Mass Spectrometry (MS/MS) Fundamentals

1. **Ionization**: Mass spectrometers measure charged particles. Molecules are ionized in:
   - **Positive Ion Mode ($[M+H]^+$, $[M+NH_4]^+$, $[M+Na]^+$)**: The analyte picks up a proton or cation adduct.
   - **Negative Ion Mode ($[M-H]^-$)**: The analyte loses a proton or acquires an anionic adduct.
2. **Precursor Mass Detection ($m/z$)**: The precursor mass-to-charge ratio ($m/z$) is measured with high accuracy ($<10\,\text{ppm}$ tolerance), strongly constraining the elemental composition / molecular formula.
3. **Collision-Induced Dissociation (CID) Fragmentation**: The precursor ion is selected in the quadrupole and collided with neutral inert gas (e.g., $N_2$ or $Ar$) at specific collision energies (e.g., $10 - 50\,\text{eV}$). This cleaves covalent bonds, generating fragment ions whose intensities and $m/z$ values form the signature MS/MS fragmentation fingerprint.
4. **Base Peak & Precursor Peak**: The base peak represents the most abundant fragment ion, while the precursor peak corresponds to the unfragmented intact ion.

---

## Evaluation Metric: Mean Reciprocal Rank @ 25 (MRR@25)

Submissions are evaluated on **Mean Reciprocal Rank @ 25 (MRR@25)**:

$$\text{MRR@25} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

Where:

- $N$ is the total number of test molecules.
- $\text{rank}_i$ is the 1-based position ($1 \le \text{rank}_i \le 25$) of the **first correct structure** in your candidate ranked list for molecule $i$.
- If none of the 25 predictions is correct, molecule $i$ receives a score of $0.0$.
- Rank scoring schedule:
  - Rank 1: **1.000**
  - Rank 2: **0.500**
  - Rank 3: **0.333**
  - ...
  - Rank 25: **0.040**

### Matching Standard: InChIKey14 via RDKit Tautomer Canonicalization

Predictions are compared using **atom connectivity matching**:

1. Both the predicted SMILES and ground truth SMILES are processed through RDKit's `TautomerEnumerator` canonicalization (pinned to version `2026.03.3`).
2. The canonicalized molecule is converted to an **InChIKey**, and truncated to the **first 14-character block** (`InChIKey14`), representing the molecular skeleton and connectivity.
3. **Stereochemical and tautomeric variations are not penalized**: For example, D-glucose and L-glucose isomers reduce to the identical InChIKey14: `WQZGKKKJIJFFOK`.

---

## Submission Format & Kaggle Notebook Constraints

Each `molecule_id` must have up to 25 candidate SMILES strings, ordered by highest confidence first, joined by semicolons:

```csv
molecule_id,smiles
m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;CN1C=NC2=C1C(=O)N(C)C(=O)N2C
m_004d06,NCCc1ccc(O)cc1;CC(=O)Nc1ccc(O)cc1;OCC(O)CO
```

### Kaggle Environment Guardrails

- **Execution Budget**: $\le 9$ hours runtime (CPU or GPU).
- **Network Access**: **Disabled** during notebook submission evaluation.
- **External Data**: Freely and publicly available external data and pre-trained models are permitted.
- **Artifact Name**: `submission.csv` strictly required in the root directory.

---

## Repository Directory Architecture

Following our research group's standard layout:

```
Molecule/
├── configs/                     # YAML configuration matrices
│   ├── project.yaml             # Competition & metadata definitions
│   ├── data.yaml                # Preprocessing & path configurations
│   ├── models.yaml              # Architecture & hyperparameter specifications
│   ├── training.yaml            # Optimizer, scheduler & batch configs
│   ├── hardware.yaml            # GPU/CPU/Memory allocations
│   └── visualization.yaml       # Plotting themes and color palettes
├── data/                        # Tiered data storage architecture
│   ├── raw/                     # Original competition files (train/test/sample)
│   ├── interim/                 # Filtered and normalized spectra
│   ├── processed/               # Tokenized and binned feature representations
│   ├── splits/                  # K-fold cross-validation partitions
│   ├── external/                # External reference libraries (MassBank, GNPS, HMDB)
│   ├── manifests/               # Dataset hash registries and integrity audits
│   └── README.md
├── docs/                        # Formal research documentation
│   ├── FOLDER_STRUCTURE.md      # Detailed repository guide
│   ├── architecture.md          # Multi-stage ML pipeline design
│   ├── methodology.md           # Mathematical and algorithmic formulation
│   ├── metric_evaluation.md     # MRR@25 and InChIKey14 specification
│   ├── scientific_background.md # Mass spectrometry physics and chemistry
│   ├── dataset_documentation.md # Feature schemas and adduct definitions
│   ├── data_dictionary.md       # Column definitions and types
│   ├── model_card.md            # Model card and performance boundaries
│   ├── reproducibility.md       # Deterministic seeds and environment specs
│   ├── installation.md          # Setup instructions
│   └── troubleshooting.md       # Common debugging recipes
├── experiments/                 # Tracked experimental artifacts
│   ├── configs/                 # Per-experiment configs
│   ├── results/                 # Metrics and evaluation outputs
│   └── experiment_registry.csv  # Global experiment audit log
├── models/                      # Model weights and specialized architectures
│   ├── classical_ml/            # Precursor matching & fingerprint models
│   ├── deep_learning/           # Neural spectral encoders (Spec2Vec, MassFormer)
│   ├── retrieval/               # Vector similarity search engines
│   ├── generative/              # SMILES autoregressive generators
│   ├── ensemble/                # Rank fusion and consensus models
│   └── explainability/          # Substructure and fragment attribution
├── notebooks/                   # Numbered, reproducible Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_spectral_preprocessing.ipynb
│   ├── 03_metric_and_canonicalization.ipynb
│   ├── 04_baseline_models.ipynb
│   └── 05_submission_pipeline.ipynb
├── outputs/                     # Final generated artifacts
│   ├── predictions/             # submission.csv and prediction dumps
│   ├── trained_models/          # Exported model weights
│   └── inference_reports/       # Latency, coverage, and MRR metrics
├── papers/                      # Research literature & academic analysis
│   ├── pdf/                     # Foundational literature references
│   ├── metadata/                # papers_metadata.json catalog
│   ├── literature_reviews/      # State-of-the-art literature review
│   ├── methodology_extraction/  # Mathematical derivations and formulas
│   ├── baseline_comparisons/    # Benchmark matrices against published papers
│   ├── research_gaps/           # Open challenges in natural product mass spec
│   └── README.md
├── reports/                     # Formal academic and validation reports
│   ├── evaluation/              # Fold-by-fold validation reports
│   ├── figures/                 # High-resolution vector figures
│   ├── model_cards/             # Model governance documentation
│   ├── research/                # In-depth scientific findings
│   ├── model_comparison.md      # Markdown comparative benchmark table
│   └── model_comparison.csv     # Raw metric comparative records
├── scripts/                     # Standalone CLI execution utilities
│   ├── setup_environment.py     # Automated environment verification
│   ├── download_competition_data.py # Kaggle API fetcher
│   ├── verify_hardware.py       # Compute and GPU resource profiler
│   ├── preprocess_spectra.py    # Peak cleaning and binning pipeline
│   ├── create_splits.py         # Scaffold / stratified split generator
│   ├── train_baseline.py        # Baseline model training script
│   ├── evaluate_metric.py       # MRR@25 evaluation runner
│   ├── generate_submission.py   # Final submission validator & bundler
│   └── run_all_pipeline.sh      # End-to-end automated runner
├── src/                         # Production Python source package
│   └── molecules/
│       ├── __init__.py
│       ├── config.py            # Typed settings and environment paths
│       ├── data/                # Loaders, preprocessors, feature extractors
│       ├── evaluation/          # MRR@25 and InChIKey14 canonicalization
│       ├── models/              # Neural and classical model definitions
│       ├── postprocessing/      # Deduplication, reranking, and formatting
│       └── utils/               # Chemistry helpers, IO, and logging
├── tests/                       # Pytest test suite
│   ├── unit/                    # Unit tests for metric, data, canonicalizer
│   ├── integration/             # End-to-end submission generation tests
│   └── fixtures/                # Mock spectra and gold-standard SMILES
├── visualizations/              # Generated high-resolution diagnostic plots
│   ├── data_exploration/
│   ├── spectral_analysis/
│   ├── training/
│   ├── validation/
│   └── comparisons/
├── logs/                        # Application and training log files
├── checkpoints/                 # Model checkpoints (best, latest, periodic)
├── CITATION.cff                 # Citation metadata
├── Makefile                     # Standard developer commands
├── pyproject.toml               # Package configuration
├── requirements.txt             # Primary production dependencies
├── requirements-dev.txt         # Development & testing dependencies
└── pytest.ini                   # Pytest configuration
```

---

## Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/darrehanrasool/Molecules.git
cd Molecules
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Hardware & System

```bash
python scripts/verify_hardware.py
```

### 3. Run Test Suite

```bash
pytest tests/ -v
```

### 4. Build Benchmark & Run Full Pipeline

```bash
bash scripts/run_all_pipeline.sh
```

---

## Citations & References

If referencing this challenge or codebase, please cite:

```bibtex
@misc{healey2026casmi,
  title={Enveda CASMI 2026 - Molecule ID From Mass Spectra},
  author={David Healey and Christoph Krettler and Tobias Kind and Marie Killian and Seth Drake and Erik DeBloois and Ivy Lightheart and Yojana Gadiya and Pelle Simpson and Daniel Domingo-Fernandez and Viswa Colluru and August Allen and Walter Reade and Ashley Oldacre},
  year={2026},
  publisher={Kaggle},
  howpublished={\url{https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra}}
}
```

```bibtex
@software{molecules_casmi2026,
  author = {Assad, Dr. Asif and Rasool, Dar Rehan and Shah, Muneeb Syed and Pampori, Shafiya Bashir},
  title = {Enveda CASMI 2026 - Molecule ID From Mass Spectra: Deep Spectral Representation & Candidate Retrieval Platform},
  year = {2026},
  url = {https://github.com/darrehanrasool/Molecules}
}
```
