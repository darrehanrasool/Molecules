#!/usr/bin/env python3
"""Cross-validation splitter partitioned by InChIKey skeleton to prevent data leakage."""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold
from molecules.evaluation.canonicalization import smiles_to_inchikey14
from molecules.utils.io import load_dataset, save_dataset
from molecules.utils.logging import get_logger

logger = get_logger("create_splits")


def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "processed" / "train_clean.parquet"
    splits_dir = root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists() and not data_path.with_suffix(".csv").exists():
        logger.warning(f"Clean data not found. Running preprocess_spectra first.")
        import subprocess
        subprocess.run([sys.executable, str(root / "scripts" / "preprocess_spectra.py")])

    df = load_dataset(str(data_path))
    logger.info(f"Loaded {len(df)} records for CV splitting.")

    skeletons = [smiles_to_inchikey14(s) or f"UNK_{i}" for i, s in enumerate(df["smiles"])]
    df["skeleton"] = skeletons

    unique_skeletons = df["skeleton"].unique()
    n_splits = min(5, len(unique_skeletons))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_skeletons)):
        val_skels = set(unique_skeletons[val_idx])
        val_mask = df["skeleton"].isin(val_skels)
        train_df = df[~val_mask]
        val_df = df[val_mask]

        train_file = splits_dir / f"train_fold_{fold}.csv"
        val_file = splits_dir / f"val_fold_{fold}.csv"

        save_dataset(train_df, str(train_file))
        save_dataset(val_df, str(val_file))
        logger.info(f"Fold {fold}: {len(train_df)} train, {len(val_df)} validation")

    logger.info("Splits created successfully with zero skeleton overlap.")


if __name__ == "__main__":
    main()
