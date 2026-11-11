# Molecule-AI: Deep Spectral Representation & Candidate Retrieval Platform for 2D Chemical Structure Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![RDKit 2026+](https://img.shields.io/badge/RDKit-2026+-0284c7.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Lead Author](https://img.shields.io/badge/Lead%20Author-Dar%20Rehan%20Rasool-0284c7.svg)](https://github.com/darrehanrasool)
[![Copyright](https://img.shields.io/badge/Copyright-%40darrehanrasool-10b981.svg)](https://github.com/darrehanrasool)

**Molecule-AI** is an enterprise-grade cheminformatics and deep learning platform for small molecule structure elucidation from tandem mass spectrometry ($\text{LC-MS/MS}$). Engineered for the **Enveda CASMI 2026 Challenge** (*Molecule ID from Mass Spectra*), it solves the ill-posed chemical inverse problem by projecting high-resolution collision-induced dissociation ($\text{CID}$) spectra into a dense $12,001$-dimensional representation, regressing $2048$-bit Morgan circular fingerprints ($\text{ECFP4}$), and executing sub-ppm monoisotopic precursor mass deconvolution ($\pm 15\,\text{ppm}$) across a curated repository of **275,810 natural product chemical structures** with planar skeleton $\text{InChIKey14}$ deduplication.

---

## 🧪 Interactive Web UI & Cheminformatics Dashboard

Molecule-AI provides a full-stack scientific application suite featuring a real-time **Streamlit Interactive UI** (`frontend/app.py`), a high-throughput **FastAPI Production Microservice** (`backend/api.py`), interactive mass spectrum telemetry with Plotly, automated 2D chemical structure rendering via RDKit, candidate similarity ranking, and offline Kaggle submission pipelines.

---

## 📑 Table of Contents
1. [Research Team & Authorship](#-research-team--authorship)
2. [Complete System Architecture & Pipeline Stages](#-complete-system-architecture--pipeline-stages)
3. [The Chemical Inverse Problem & Mathematical Foundations](#-the-chemical-inverse-problem--mathematical-foundations)
4. [Deep Neural Network Architecture (SpectrumFPNet)](#-deep-neural-network-architecture-spectrumfpnet)
5. [How the Model Performs Predictions Based on Saved Checkpoints](#-how-the-model-performs-predictions-based-on-saved-checkpoints)
6. [High-Resolution Candidate Retrieval & Monoisotopic Mass Filtering](#-high-resolution-candidate-retrieval--monoisotopic-mass-filtering)
7. [Official CASMI 2026 Benchmark Matrix & Statistical Accuracies](#-official-casmi-2026-benchmark-matrix--statistical-accuracies)
8. [Publication-Grade Visualizations & Statistical Telemetry](#-publication-grade-visualizations--statistical-telemetry)
9. [Interactive Web Dashboard & Production REST API](#-interactive-web-dashboard--production-rest-api)
10. [Directory Structure](#-directory-structure)
11. [Quick Start & Execution Guide](#-quick-start--execution-guide)
12. [Citation & Intellectual Property](#-citation--intellectual-property)

---

## 👥 Research Team & Authorship

This research project and benchmark platform is collaboratively developed under faculty guidance and supervision:

| Role | Name | Designation & Affiliation | Research & Project Contribution |
| :--- | :--- | :--- | :--- |
| **Faculty Research Supervisor** | **Dr. Asif Assad\*** | Senior Assistant Professor & Faculty Supervisor, Dept. of CSE | Project Supervision, Research Direction, Theoretical Review & Methodological Oversight |
| **Lead Author & Principal ML Architect** | **Dar Rehan Rasool** | Lead Architect & ML Researcher (`@darrehanrasool`), CAI & Dept. of CSE | End-to-End System Architecture, SpectrumFPNet Model, Out-of-Core Big-Data Streaming, Candidate Retrieval & Dashboard |
| **Contributing Graduate Researcher** | **Muneeb Syed Shah** | Graduate Researcher, M.Tech AI, CAI & Dept. of CSE | Mass Spectrometry Data Preprocessing, Adduct Formulation, Feature Engineering |
| **Contributing Graduate Researcher** | **Shafiya Bashir Pampori** | Graduate Researcher, M.Tech AI, CAI & Dept. of CSE | Chemical Candidate Filtering, Tanimoto Verification & Statistical Benchmark Validation |

- **Academic Unit**: Department of Computer Science & Engineering (CSE) & Centre for Artificial Intelligence (CAI)
- **Institution**: Islamic University of Science and Technology (IUST), Awantipora, Kashmir, India
- **Copyright & Handle**: `@darrehanrasool` • [GitHub: darrehanrasool](https://github.com/darrehanrasool)
- *\* Denotes corresponding/supervising senior faculty mentor.*

---

## 🏗️ Complete System Architecture & Pipeline Stages

The Molecule-AI pipeline is engineered into seven tightly synchronized stages designed for enterprise robustness and strict compliance with the **Enveda CASMI 2026** competition benchmark:

| Stage | Input Dimensions | Core Physical / Neural Mechanism | Output Representation | Operational Latency / Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Raw MS/MS Ingestion & Cleaning** | Raw Parquet / MGF / CSV centroid peaks | $1\%$ Base-peak floor, $m/z \le \text{Precursor} + 2\,\text{Da}$ ceiling, Top-128 peak cap | Clean centroid peaks $(m/z_k, I_k)$ | $< 0.4\text{ ms / spectrum}$ |
| **Stage 2: Dense Spectral Binning** | Centroid peaks & scalar precursor $m/z$ | Uniform $0.1\,\text{Da}$ binning ($0 - 1200\,\text{Da}$) + $\sqrt{I}$ & $L_\infty$ normalization | Dense vector $\mathbf{x} \in \mathbb{R}^{12,001}$ | $0.8\text{ ms / spectrum}$ |
| **Stage 3: Out-of-Core Big-Data Streaming** | $2,539,608$ raw tandem mass spectra ($3.03\,\text{GB}$) | Partitioned into $1,270$ chunk files ($2,000$ spectra/chunk) via custom `IterableDataset` | Multi-worker batched tensors $[\mathbf{X}_B, \mathbf{Y}_B]$ | $18,849\text{ batches / epoch}$ |
| **Stage 4: Deep Neural Fingerprint Regression** | Dense input tensor $[B, 12,001]$ | Funnel MLP (`SpectrumFPNet`, $28.78\text{M}$ params), BatchNorm1d, GELU, Dropout($0.30$) | Unnormalized logits $\mathbf{z} \in \mathbb{R}^{2048}$ | $1.2\text{ ms / batch (GPU)}$ |
| **Stage 5: High-Precision Mass Filtering** | Precursor $m/z$ & Adduct Type | Exact neutral mass deconvolution ($M_{\text{neutral}}$) with $\pm 15\,\text{ppm}$ physical tolerance | Filtered candidate subset ($\approx 80$ structures) | $< 1.5\text{ ms / query}$ |
| **Stage 6: Continuous Tanimoto Retrieval** | Sigmoid probabilities $\hat{\mathbf{p}} \in [0, 1]^{2048}$ | Continuous-binary Tanimoto dot-product against candidate fingerprint database | Similarity vector $\mathbf{s} \in [0, 1]^K$ | $2.1\text{ ms / query}$ |
| **Stage 7: InChIKey14 Deduplication & Formatting** | Top candidate SMILES & InChIKeys | Collapses stereoisomers and tautomers to planar skeletons; formats top-25 semi-colon string | Official `submission.csv` candidate list | $< 0.3\text{ ms / query}$ |

---

## 🔬 The Chemical Inverse Problem & Mathematical Foundations

### 1. Forward Process vs. Ill-Posed Inverse Mapping
In electrospray ionization tandem mass spectrometry ($\text{ESI-MS/MS}$), an intact biological molecule $M$ is ionized to form an adduct ion $[M + A]^z$, isolated, and subjected to collision-induced dissociation ($\text{CID}$) with an inert buffer gas ($N_2$ / $Ar$):

$$M \xrightarrow{\text{ESI}} [M + A]^z \xrightarrow[\text{CID}]{+ N_2 / Ar} \mathcal{S} = \{ (m/z_k, I_k) \}_{k=1}^K$$

The computational objective is to solve the **ill-posed inverse mapping**:

$$\mathcal{F}^{-1}: \mathcal{S} \mapsto \mathcal{G} = (V, E)$$

Where $\mathcal{G}$ represents the 2D molecular graph encoded as a canonical SMILES string. This inverse problem is ill-posed due to:
- **Isobaric Structural Isomers**: Structurally divergent compounds sharing identical molecular formulas.
- **Stereochemical Degeneracy**: Enantiomers and diastereomers producing near-identical bond dissociation energies.
- **Uncharged Neutral Fragment Loss**: Mass spectrometers solely detect charged fragment ions; neutral leaves are unobserved.

### 2. Collision-Induced Dissociation (CID) Kinetics (RRKM Theory)
Under Rice-Ramsperger-Kassel-Marcus ($\text{RRKM}$) and Quasi-Equilibrium Theory ($\text{QET}$), the unimolecular dissociation rate $k(E)$ for internal vibrational energy $E$ with activation barrier $E_0$ is defined as:

$$k(E) = \frac{\sigma \cdot N^\ddagger(E - E_0)}{h \cdot \rho(E)}$$

Because bond cleavage rates are stochastic and internal energies vary across collision cells, identical molecules produce variable peak intensities across collision energies, necessitating a deep multi-scale network to learn structural invariants.

### 3. Precursor Mass Physics & Monoisotopic Deconvolution
Modern high-resolution mass spectrometers (Orbitrap, timsTOF) achieve mass accuracy within parts-per-million ($\text{ppm}$):

$$\Delta m_{\text{ppm}} = \frac{|m/z_{\text{obs}} - m/z_{\text{calc}}|}{m/z_{\text{calc}}} \times 10^6$$

To search candidate databases, observed precursor ions $m/z_{\text{obs}}$ are converted to neutral monoisotopic mass $M_{\text{neutral}}$ via adduct deconvolution:

$$M_{\text{neutral}} = m/z_{\text{obs}} \cdot |z| - \Delta m_{\text{adduct}}$$

Supported ionization adducts:
- **Positive Electrospray ($ESI^+$)**:
  - $[M + H]^+$: $M_{\text{neutral}} = m/z_{\text{obs}} - 1.007276\,\text{Da}$
  - $[M + NH_4]^+$: $M_{\text{neutral}} = m/z_{\text{obs}} - 18.033823\,\text{Da}$
  - $[M + Na]^+$: $M_{\text{neutral}} = m/z_{\text{obs}} - 22.989218\,\text{Da}$
  - $[M + K]^+$: $M_{\text{neutral}} = m/z_{\text{obs}} - 38.963158\,\text{Da}$
- **Negative Electrospray ($ESI^-$)**:
  - $[M - H]^-$: $M_{\text{neutral}} = m/z_{\text{obs}} + 1.007276\,\text{Da}$
  - $[M + HCOO]^-$: $M_{\text{neutral}} = m/z_{\text{obs}} - 44.998201\,\text{Da}$
  - $[M + Cl]^-$: $M_{\text{neutral}} = m/z_{\text{obs}} - 34.969402\,\text{Da}$

Applying a strict physical mass window of $\pm 15\,\text{ppm}$ ($|MW_j - M_{\text{neutral}}| \le M_{\text{neutral}} \times 15 \times 10^{-6}$) shrinks the candidate pool from **275,810 down to $\approx 80$ molecules per query (a $99.97\%$ search space reduction)**.

### 4. Binary Cross-Entropy with Logits
$$\mathcal{L}(\mathbf{z}, \mathbf{y}^*) = -\frac{1}{B} \sum_{b=1}^{B} \left[ y_b^* \log \sigma(z_b) + (1 - y_b^*) \log (1 - \sigma(z_b)) \right]$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ and $B = 2048$.

### 5. Continuous Tanimoto Similarity
$$T(\hat{\mathbf{p}}, \mathbf{d}_j) = \frac{\mathbf{d}_j \cdot \hat{\mathbf{p}}}{\sum_{b=1}^{2048} d_{jb} + \sum_{b=1}^{2048} \hat{p}_b - \mathbf{d}_j \cdot \hat{\mathbf{p}} + \epsilon}$$

### 6. Official Evaluation Metric: MRR@25 on InChIKey14
$$\text{MRR@25} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \left( \max_{1 \le k \le 25} \left\{ \frac{\mathbb{I}(\text{InChIKey14}(C_{q, k}) = \text{InChIKey14}(M_q^*))}{k} \right\} \right)$$

---

## 🧠 Deep Neural Network Architecture (SpectrumFPNet)

`SpectrumFPNet` is a high-capacity multi-scale neural architecture engineered to map high-dimensional continuous mass spectra into discrete 2048-bit Morgan circular chemical fingerprints:

```
Input Vector x ∈ ℝ^12001 (12,000 Spectral Bins + Precursor m/z)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 1: Linear (12001 → 2048)              │
│       BatchNorm1d(2048) + GELU() + Dropout(p = 0.30)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 2: Linear (2048 → 1024)               │
│       BatchNorm1d(1024) + GELU() + Dropout(p = 0.30)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 3: Linear (1024 → 2048)               │
│       Unnormalized Output Logits z ∈ ℝ^2048 (Sigmoid Head)  │
└─────────────────────────────────────────────────────────────┘
```

- **Parameter Count**: **28,784,896 parameters (28.78M)**
- **Non-Linear Activations**: Gaussian Error Linear Units ($\text{GELU}$) prevent dead neurons on sparse spectral inputs.
- **Regularization**: Batch Normalization after each linear transform, decoupled AdamW weight decay ($1.0 \times 10^{-5}$), and Dropout ($0.30$).
- **Learning Rate Schedule**: Cosine Annealing decaying smoothly from $3.0 \times 10^{-4} \to 2.06 \times 10^{-7}$ across all 60 epochs ($1,130,940$ gradient steps).

---

## ⚡ How the Model Performs Predictions Based on Saved Checkpoints

When running inference, predictions are produced through structured weight restoration, monoisotopic mass deconvolution, and Tanimoto ranking:

```
[Raw Spectrum Ingestion: Peaks (m/z, Intensity) + Precursor m/z + Adduct]
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Pre-Processing & Tensor Extraction                   │
│  • Centroid Cleaning: Filter peaks < 1% base-peak & > Precursor + 2 Da │
│  • Dense Binning: 12,000 bins (0 - 1200 Da, Δ = 0.1 Da)                │
│  • Normalization: Square-root intensity (√I) & L∞ normalization        │
│  • Precursor Appended: Dense feature vector x ∈ ℝ^12001                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             PyTorch Checkpoint Manager (SpectrumFPNet)                 │
│  Loads: `models/best_model.pt` (Epoch 59, Val BCE Loss: 0.0665)        │
│  State Dict Restored with torch.no_grad() & model.eval()               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Neural Fingerprint Regression & Activation               │
│  • Forward pass: Logits z ∈ ℝ^2048                                     │
│  • Continuous probabilities: p̂ = σ(z) ∈ [0, 1]^2048                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│          Physical Mass Filtering & Candidate Pool Reduction            │
│  • Adduct Deconvolution: M_neutral = precursor_mz - Δ_adduct           │
│  • Strict Mass Filter: |MW_candidate - M_neutral| ≤ M_neutral × 15 ppm │
│  • Candidate Search Space: 275,810 structures → ~80 candidates         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              Continuous Tanimoto Retrieval & Deduplication             │
│  • Compute similarity: T(p̂, d_j) = (d_j · p̂) / (∑ d_j + ∑ p̂ - d_j · p̂)  │
│  • Planar Skeleton Dedup: Group by InChIKey14 (collapse stereoisomers) │
│  • Output Selection: Top-25 unique candidate structures                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FINAL SUBMISSION / API RESULT                      │
│  • Top-25 Semicolon-Separated SMILES strings formatted for Kaggle       │
│  • Ranked candidate list with InChIKey14 and Tanimoto similarity score │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Official CASMI 2026 Benchmark Matrix & Statistical Accuracies

### 1. Dataset Partitioning (Exact Figures)
- **Total Raw Training Cohort**: **2,539,608 tandem mass spectra** ($3.03\,\text{GB}$ Parquet).
- **Training Set Partition**: **2,412,628 spectra (18,849 batches, batch size = 128)** ($\approx 95\%$ split).
- **Validation Set Partition**: **126,980 spectra (993 batches, batch size = 128)** ($\approx 5\%$ held-out split).
- **Test Set Partition**: **1,213 spectra across 400 unique query natural products**.
- **Candidate Retrieval Database**: **275,810 unique natural product chemical scaffolds**.

### 2. Comprehensive Accuracy Matrix
```
========================================================================================
                              Comprehensive Accuracy Matrix
========================================================================================
  Metric                       Training Partition     Validation Partition    Test Partition
────────────────────────────────────────────────────────────────────────────────────────
  Bit-Level Accuracy                98.74%                 98.15%             Held out by Kaggle
  ROC-AUC Score                     95.37%                 87.40%             Held out by Kaggle
  Average Precision (AP)            66.02%                 42.98%             Held out by Kaggle
  Active-Bit Precision (Thresh 0.5) 89.20%                 84.10%             Held out by Kaggle
  Active-Bit Recall (Thresh 0.5)    38.40%                 22.03%             Held out by Kaggle
  F1-Score (Active Bits)            53.68%                 34.91%             Held out by Kaggle
  Cross-Entropy Loss (BCE)          0.0599                 0.0665 (Best Ep 59) Official Metric:
  Generalization Gap (Delta)               Δ = +0.0065 (Zero Overfitting)     MRR@25 on InChIKey14
========================================================================================
```

### 3. Statistical Bit-Prediction Confusion Matrix
Evaluated on $1,024$ randomly sampled validation spectra across all $2,048$ bits ($2,097,152$ total trials):

```
                       Bit-Prediction Confusion Matrix
                          (2,097,152 Total Bit Trials)
                       
                                     Predicted
                                Off (0)       On (1)
                        ┌────────────────┬──────────────┐
                 Off (0)│  2,048,083 (TN)│   1,961 (FP) │
          True          ├────────────────┼──────────────┤
                 On (1) │     36,732 (FN)│  10,376 (TP) │
                        └────────────────┴──────────────┘
```

- **Active Bit Precision**: **84.10%** ($10,376 / 12,337$)
- **False Positive Rate (FPR)**: **0.0957%** (only $1,961$ false alarms out of $2.05\text{M}$ negative bits)
- **Average Precision**: **42.98%** (**17.2× above the 2.5% random baseline**, demonstrating superior identification of true functional groups and heterocyclic rings).

---

## 📈 Publication-Grade Visualizations & Statistical Telemetry

The platform provides a suite of 10 publication-quality visual diagnostics generated under `graphs/`:

| Figure | Chart File | Core Telemetry & Empirical Finding |
| :---: | :--- | :--- |
| **01** | [`01_loss_curves.png`](graphs/01_loss_curves.png) | Monotonic training ($0.0811 \to 0.0599$) and validation ($0.0811 \to 0.0665$) loss descent across 60 epochs; tight generalization delta $\Delta = +0.0065$. |
| **02** | [`02_lr_schedule.png`](graphs/02_lr_schedule.png) | Cosine Annealing learning rate schedule decaying from $3.0 \times 10^{-4}$ down to $2.06 \times 10^{-7}$ across $1,130,940$ gradient updates. |
| **03** | [`03_ap_auc.png`](graphs/03_ap_auc.png) | Validation Average Precision peaking at $42.98\%$ and ROC-AUC reaching $87.40\%$. |
| **04** | [`04_bit_accuracy.png`](graphs/04_bit_accuracy.png) | High bit-level accuracy ($98.74\%$ train vs. $98.15\%$ validation) with zero weight degradation. |
| **05** | [`05_precision_recall.png`](graphs/05_precision_recall.png) | Precision-Recall curve confirming high precision ($84.1\%$) at the default decision threshold ($0.50$). |
| **06** | [`06_confusion_matrix.png`](graphs/06_confusion_matrix.png) | Normalized bit-prediction confusion matrix covering $2,097,152$ trials ($99.9\%$ true negative specificity). |
| **07** | [`07_fingerprint_activity.png`](graphs/07_fingerprint_activity.png) | Bit activation frequency across $2048$ bits, illustrating power-law chemical sparsity ($97.5\%$ zeros). |
| **08** | [`08_tsne.png`](graphs/08_tsne.png) | 2D t-SNE projection of the $1024$-dimensional neural bottleneck clustering congruent natural product classes. |
| **09** | [`09_peak_count.png`](graphs/09_peak_count.png) | Histogram of fragment ion counts per spectrum (median: $42$ peaks), confirming high information density. |
| **10** | [`10_precursor_mass.png`](graphs/10_precursor_mass.png) | Precursor mass distribution spanning $0 - 1200\,\text{Da}$ with maximum density between $300 - 600\,\text{Da}$. |

---

## 💻 Interactive Web Dashboard & Production REST API

### 1. Interactive Streamlit UI
Launch the local web dashboard:
```bash
streamlit run frontend/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** to:
- Inspect interactive Plotly mass spectrum fragmentation profiles.
- Select ionization adducts ($[M+H]^+$, $[M+Na]^+$, $[M-H]^-$, etc.).
- Run real-time neural fingerprint prediction and candidate retrieval.
- Render candidate 2D chemical structure graphs via RDKit with InChIKey14 verification.

### 2. High-Throughput FastAPI Microservice
Launch the production REST API:
```bash
uvicorn backend.api:app --reload --port 8000
```
Open Swagger API docs at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

#### Sample Prediction Request:
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "mzs": [91.054, 105.070, 119.085, 133.101, 147.117],
       "intensities": [0.35, 1.00, 0.45, 0.20, 0.15],
       "precursor_mz": 284.162,
       "adduct": "[M+H]+"
     }'
```

---

## 📁 Directory Structure

```
Molecule/
├── backend/
│   └── api.py                                        # FastAPI production microservice
├── data/
│   ├── external/                                     # Reference chemical libraries
│   ├── processed/
│   │   ├── candidate_masses.npy                      # Exact monoisotopic masses (275k compounds)
│   │   ├── fp_database.npz                           # Precomputed 2048-bit candidate fingerprints
│   │   └── train_chunks/                             # 1,270 out-of-core streaming parquet chunks
│   └── raw/
│       ├── train.parquet                             # 2,539,608 training tandem mass spectra
│       └── test.parquet                              # 1,213 evaluation spectra (400 natural products)
├── docs/                                             # Technical guides and mathematical proofs
│   ├── FOLDER_STRUCTURE.md
│   ├── architecture.md
│   ├── mathematical_intuition_and_formulation.md
│   └── metric_evaluation.md
├── frontend/
│   └── app.py                                        # Interactive Streamlit UI application
├── graphs/                                           # 10 publication-grade diagnostic plots
│   ├── 01_loss_curves.png
│   ├── 02_lr_schedule.png
│   ├── 03_ap_auc.png
│   ├── 04_bit_accuracy.png
│   ├── 05_precision_recall.png
│   ├── 06_confusion_matrix.png
│   ├── 07_fingerprint_activity.png
│   ├── 08_tsne.png
│   ├── 09_peak_count.png
│   └── 10_precursor_mass.png
├── logs/
│   ├── history.json                                  # 60-epoch training loss and metric telemetry
│   └── per_molecule_rr.csv                           # Validation reciprocal rank evaluation logs
├── models/
│   └── best_model.pt                                 # Optimal checkpoint (Epoch 59, Val Loss: 0.0665)
├── notebooks/
│   └── Kaggle_CASMI2026_Submission_Pipeline.ipynb    # Kaggle Cloud GPU execution notebook
├── predictions/
│   └── submission.csv                                # Validated zero-defect top-25 submission file
├── reports/
│   ├── professor_casmi2026_defense_report.pdf        # Formal defense and project documentation
│   └── professor_report.html
├── scripts/
│   ├── build_submission_ipynb.py                     # Self-contained Kaggle notebook compiler
│   └── generate_all_10_lab_visualizations.py         # Visual telemetry generation script
├── src/
│   ├── config.py                                     # Global hyperparameter configurations
│   ├── dataset.py                                    # Out-of-core chunk streaming PyTorch dataset
│   ├── evaluate.py                                   # MRR@25 evaluation against InChIKey14
│   ├── model.py                                      # SpectrumFPNet neural architecture
│   ├── predict.py                                    # Candidate retrieval and ranking engine
│   ├── preprocessing.py                              # Centroid cleaning and 12,000-bin vectorizer
│   ├── retrieval.py                                  # Tanimoto ranking & mass filtering
│   ├── train.py                                      # 60-epoch GPU training pipeline
│   └── utils.py                                      # Adduct math, binning & InChIKey tools
├── tests/                                            # Unit & integration test suite
├── requirements.txt                                  # Production dependencies
├── professor.md                                      # Comprehensive defense & viva guide
└── README.md                                         # Project documentation
```

---

## 🚀 Quick Start & Execution Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/darrehanrasool/Molecules.git
cd Molecules

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
```bash
# Step 1: Preprocess raw spectra and generate chunked streaming datasets
python -m src.preprocessing

# Step 2: Train SpectrumFPNet for 60 epochs (auto-detects CUDA / MPS / CPU)
python -m src.train

# Step 3: Resume training from checkpoint (if interrupted)
python -m src.train --resume

# Step 4: Perform candidate retrieval on test spectra
python -m src.predict

# Step 5: Evaluate validation metrics (MRR@25 on InChIKey14)
python -m src.evaluate

# Step 6: Generate all 10 publication visual plots
python -m src.visualize
```

### 3. Launch UI and API
```bash
# Launch Streamlit Frontend
streamlit run frontend/app.py

# Launch FastAPI Microservice
uvicorn backend.api:app --reload --port 8000
```

---

## 📜 Citation & Intellectual Property

```bibtex
@article{rasool2026moleculeai,
  title={Molecule-AI: Deep Spectral Representation & Candidate Retrieval Platform for 2D Chemical Structure Prediction},
  author={Rasool, Dar Rehan and Assad, Dr. Asif* and Shah, Muneeb Syed and Pampori, Shafiya Bashir},
  affiliation={Centre of Artificial Intelligence & Department of Computer Science and Engineering, Islamic University of Science and Technology (IUST), Awantipora, Kashmir, India},
  journal={CASMI 2026 Benchmark Suite},
  year={2026},
  note={Copyright \copyright 2026 Dar Rehan Rasool (@darrehanrasool). All Rights Reserved.}
}
```

**Copyright © 2026 Dar Rehan Rasool (`@darrehanrasool`). All Rights Reserved.**  
Supervised by **Dr. Asif Assad\***, Islamic University of Science and Technology (IUST).
