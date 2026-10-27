# Tiered Data Architecture

This directory houses the raw, interim, processed, partitioned, and external reference datasets for the Enveda CASMI 2026 challenge.

## Directory Substructure

- `raw/`: Untouched primary competition data downloaded from Kaggle (`train.parquet`, `test.parquet`, `sample_submission.csv`).
- `interim/`: Pre-filtered and noise-reduced intermediate spectra.
- `processed/`: Normalized, peak-binned, and tokenized representations.
- `splits/`: Stratified 5-fold cross-validation splits partitioned by InChIKey skeleton to prevent data leakage.
- `external/`: Curated public reference libraries:
  - MassBank (North America & Europe)
  - GNPS (Global Natural Products Social Molecular Networking)
  - HMDB (Human Metabolome Database)
  - PubChem Compound monoisotopic catalog
- `manifests/`: Cryptographic checksum registries (SHA-256) verifying dataset integrity.
