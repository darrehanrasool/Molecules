# CASMI 2026 From-Zero Kaggle Runbook

## Purpose

`notebooks/CASMI2026_From_Zero_End_To_End.ipynb` is the clean experimental
pipeline. It is separate from the historical submission notebook so previous
results are preserved and cannot silently mix with the new experiment.

The notebook creates one audit directory:

```text
/kaggle/working/casmi_run/
├── eda/          schema and distribution reports
├── figures/      EDA figures
├── manifests/    data contract, split, environment, final run manifest
├── metrics/      baseline, training, coverage, and final metrics
├── models/       selected checkpoint
├── predictions/  final submission copy
├── logs/         run logs
└── submission.csv
```

## Exact execution order

1. Create a Kaggle notebook with GPU enabled.
2. Attach the competition dataset containing `train.parquet`,
   `test.parquet`, and `sample_submission.csv`.
3. Add the notebook from this repository:
   `notebooks/CASMI2026_From_Zero_End_To_End.ipynb`.
4. Run all cells from top to bottom without skipping cells.
5. Download `/kaggle/working/casmi_run/` as the experiment artifact bundle.
6. Upload `/kaggle/working/submission.csv` to the competition.
7. Record the official Kaggle score separately from the hidden-test metrics.

## What each phase proves

| Phase | Artifact | Evidence |
|---|---|---|
| Data audit | `eda/*`, `data_contract.json` | Columns, dtypes, missingness, cardinality |
| EDA | `eda/summary.csv`, `figures/eda_overview.png` | Peak counts, precursor distribution, adduct distribution |
| Chemistry | notebook output and label counts | Valid SMILES, canonical structures, InChIKey14 |
| Split | `manifests/split.json` | Zero train/validation group overlap |
| Preprocessing | `final_run_manifest.json` | Peak filter, bin width, intensity transform |
| Candidate library | notebook output | Candidate count and exact-mass representation |
| Baseline | `metrics/mass_baseline.json` | Mass-only retrieval lower bound |
| Neural training | `metrics/training_history.csv` | Epoch losses and molecule-level validation metrics |
| Coverage | `metrics/validation_mass_coverage.csv` | Whether the correct target survives the mass filter |
| Submission | `predictions/submission.csv` | One row per molecule and 25 candidates |

## Metrics to report to the professor

Report these values exactly as generated; do not copy values from the old
reports:

- number of raw train/test rows;
- number of unique molecule groups;
- valid-SMILES fraction;
- train/validation rows and groups;
- overlap count (must be zero);
- candidate-library size;
- mass-filter candidate count distribution;
- target survival rate at 15 ppm and 30 ppm;
- training BCE loss;
- fingerprint AP and ROC-AUC if added to the run;
- validation Top-1, Top-5, Top-25;
- validation MRR@25;
- final official Kaggle MRR@25.

The hidden test labels are not available in the notebook. Any statement about
test accuracy before the leaderboard result is an estimate based on validation,
not a measured test result.

## Mathematical and ML choices

- **Chemical representation:** canonical SMILES and InChIKey14 for connectivity
  evaluation; Morgan/ECFP4 radius 2 with 2048 bits as a multi-label target.
- **Spectrum representation:** cleaned centroid peaks, square-root intensity,
  per-spectrum max normalization, and 0.1 Da dense bins.
- **Split design:** deterministic molecule-group split to prevent multiple
  collision-energy spectra from crossing train and validation.
- **Learning objective:** BCE-with-logits because each fingerprint bit is a
  correlated but non-mutually-exclusive binary target.
- **Optimizer:** AdamW with decoupled weight decay.
- **Retrieval:** neutral-mass filtering followed by continuous Tanimoto ranking
  and InChIKey14 deduplication.
- **Selection criterion:** molecule-level validation MRR@25, not bit accuracy.

## Important resource note

The dense matrix for the complete multi-million-row training set can exceed
Kaggle notebook RAM if materialized at once. The notebook is designed as a
fully auditable reference run and should first be executed on a bounded
development sample. The production full-data run should use the repository's
out-of-core chunk writer and `IterableDataset`, with the same preprocessing
contract and split manifest. Never increase the data volume by changing the
preprocessing rules.
