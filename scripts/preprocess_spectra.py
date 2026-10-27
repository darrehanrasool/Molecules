#!/usr/bin/env python3
"""Spectral preprocessing pipeline CLI: Noise filtering and peak normalization."""

import sys
from pathlib import Path
import pandas as pd
from molecules.data.loader import parse_peak_string
from molecules.data.preprocessor import SpectralPreprocessor
from molecules.utils.io import load_dataset, save_dataset
from molecules.utils.logging import get_logger

logger = get_logger("preprocess_spectra")


def main():
    root = Path(__file__).resolve().parent.parent
    raw_path = root / "data" / "raw" / "train.parquet"
    out_path = root / "data" / "processed" / "train_clean.parquet"

    if not raw_path.exists() and not raw_path.with_suffix(".csv").exists():
        logger.warning(f"Raw data not found at {raw_path}. Creating sample demonstration data.")
        # Create demo dataset for testing
        df = pd.DataFrame([
            {
                "molecule_id": "m_0014ef",
                "precursor_mz": 255.0652,
                "adduct": "[M-H]-",
                "peaks": "105.03:0.42 153.01:0.85 254.05:1.0",
                "smiles": "O=C1C=C(c2ccccc2)Oc3cc(O)cc(O)c13"
            },
            {
                "molecule_id": "m_004d06",
                "precursor_mz": 181.0707,
                "adduct": "[M+H]+",
                "peaks": "60.02:0.5 73.03:1.0 145.05:0.65",
                "smiles": "OCC1OC(O)C(O)C(O)C1O"
            }
        ])
        save_dataset(df, str(raw_path.with_suffix(".csv")))
    else:
        df = load_dataset(str(raw_path))

    logger.info(f"Loaded {len(df)} spectra. Preprocessing...")
    preprocessor = SpectralPreprocessor(min_intensity_rel=0.005, normalize=True)

    cleaned_peaks_list = []
    for _, row in df.iterrows():
        raw_peaks = parse_peak_string(row["peaks"])
        cleaned = preprocessor.process(raw_peaks, row.get("precursor_mz"))
        cleaned_str = " ".join([f"{mz:.4f}:{it:.4f}" for mz, it in cleaned])
        cleaned_peaks_list.append(cleaned_str)

    df["peaks_clean"] = cleaned_peaks_list
    save_dataset(df, str(out_path))
    logger.info(f"Preprocessed dataset saved successfully.")


if __name__ == "__main__":
    main()
