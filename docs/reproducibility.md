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
   Splits are saved directly to disk (`data/splits/cv_fold_*.csv`) based on Murcko scaffold / InChIKey skeleton partitioning to eliminate cross-fold structural leakage.
