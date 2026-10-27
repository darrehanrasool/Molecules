#!/usr/bin/env python3
"""Baseline retrieval training and validation runner."""

import argparse
import sys
from pathlib import Path
import pandas as pd
from molecules.data.loader import parse_peak_string, MassSpectrumRecord
from molecules.models.retrieval import PrecursorRetrievalPredictor
from molecules.postprocessing.ranker import CandidateRanker
from molecules.evaluation.metric import compute_mrr_25
from molecules.utils.io import load_dataset
from molecules.utils.chemistry import precursor_mz_to_neutral_mass
from molecules.utils.logging import get_logger

logger = get_logger("train_baseline")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline retrieval model.")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    logger.info(f"Starting baseline retrieval benchmark using config: {args.config}")

    train_path = root / "data" / "processed" / "train_clean.parquet"
    if not train_path.exists() and not train_path.with_suffix(".csv").exists():
        import subprocess
        subprocess.run([sys.executable, str(root / "scripts" / "preprocess_spectra.py")])

    train_df = load_dataset(str(train_path))

    records = []
    for _, row in train_df.iterrows():
        peaks = parse_peak_string(row.get("peaks_clean", row.get("peaks", "")))
        rec = MassSpectrumRecord(
            molecule_id=str(row["molecule_id"]),
            precursor_mz=float(row["precursor_mz"]),
            adduct=str(row.get("adduct", "[M+H]+")),
            peaks=peaks,
            smiles=str(row.get("smiles")),
        )
        records.append(rec)

    # Use training set as candidate catalog for benchmark
    catalog = train_df.copy()
    catalog["neutral_mass"] = [
        precursor_mz_to_neutral_mass(row["precursor_mz"], row.get("adduct", "[M+H]+"))
        for _, row in catalog.iterrows()
    ]

    retriever = PrecursorRetrievalPredictor(candidate_catalog=catalog, ppm_tolerance=15.0)
    ranker = CandidateRanker(max_candidates=25)

    predictions = {}
    ground_truth = {}

    for rec in records:
        scored = retriever.predict_candidates(rec, top_k=25)
        top_smiles = ranker.rank_and_deduplicate(scored)
        predictions[rec.molecule_id] = top_smiles
        ground_truth[rec.molecule_id] = rec.smiles

    metrics = compute_mrr_25(predictions, ground_truth)
    logger.info("=" * 50)
    logger.info(f"Baseline Benchmark MRR@25: {metrics['mrr_25']:.4f}")
    logger.info(f"Top-1 Accuracy          : {metrics['top_1_accuracy']:.4f}")
    logger.info(f"Top-5 Accuracy          : {metrics['top_5_accuracy']:.4f}")
    logger.info(f"Top-25 Accuracy         : {metrics['top_25_accuracy']:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
