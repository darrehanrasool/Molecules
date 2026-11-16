# Reproducibility Guide

## Deterministic Controls

To guarantee bit-exact reproducibility across training runs and cross-validation splits:

1. **Random Seed Initialization**:
   All random number generators (`numpy`, `random`, `torch`, `scikit-learn`) are initialized with `SEED=42`:
   ```python
   import random, os, numpy as np, torch

   def set_seed(seed: int = 42):
       random.seed(seed)
       os.environ["PYTHONHASHSEED"] = str(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(seed)
           torch.backends.cudnn.deterministic = True
           torch.backends.cudnn.benchmark = False
   ```

2. **RDKit Version Locking**:
   The competition pins RDKit tautomer canonicalization at version `2026.03.3`. Verification of local environment:
   ```bash
   python -c "import rdkit; print('RDKit Version:', rdkit.__version__)"
   ```

3. **Data Splitting**:
   Training and validation rows are assigned by a deterministic hash of
   `group_id`, which is the InChIKey14 skeleton when available and otherwise
   `molecule_id`, not by individual spectrum rows. This keeps all
   collision-energy spectra for one chemical group in the same split and
   prevents row-level leakage. The split fraction and hash seed are
   controlled by `val_frac` and `CFG.SEED` in `src/dataset.py`.

## Preprocessing Contract

The training and Kaggle inference paths must use the same peak contract:

- discard non-finite and non-positive peaks;
- discard peaks above `precursor_mz + 2 Da`;
- discard peaks below 1% of the spectrum base peak;
- retain the 128 most intense remaining peaks;
- bin with the shared `clean_peaks` and `bin_spectrum` helpers.

After changing preprocessing, regenerate processed chunks before retraining:

```bash
rm -rf data/processed/train_chunks data/processed/test_chunks
python -m src.preprocessing
python -m src.train
```

The Kaggle notebook contains the same cleaning logic inline because Kaggle
executes it independently of the repository package.
