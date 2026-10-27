"""
Mean Reciprocal Rank @ 25 (MRR@25) Competition Metric Engine
Computes CASMI 2026 official metric with InChIKey14 atom connectivity matching.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from molecules.evaluation.canonicalization import smiles_to_inchikey14


def compute_molecule_reciprocal_rank(
    predicted_smiles_list: List[str],
    ground_truth_smiles_or_key: str,
    max_k: int = 25
) -> Tuple[float, Optional[int]]:
    """
    Computes reciprocal rank for a single query molecule against up to max_k candidates.

    Returns:
        (reciprocal_rank, first_hit_rank_1_indexed)
        If not found in top max_k, returns (0.0, None).
    """
    # Canonicalize ground truth key
    if len(ground_truth_smiles_or_key) == 14 and ground_truth_smiles_or_key.isupper() and "-" not in ground_truth_smiles_or_key:
        truth_key = ground_truth_smiles_or_key
    else:
        truth_key = smiles_to_inchikey14(ground_truth_smiles_or_key)

    if truth_key is None:
        return 0.0, None

    for rank_idx, candidate_smiles in enumerate(predicted_smiles_list[:max_k], start=1):
        cand_key = smiles_to_inchikey14(candidate_smiles)
        if cand_key == truth_key:
            return 1.0 / rank_idx, rank_idx

    return 0.0, None


def compute_mrr_25(
    predictions: Dict[str, Union[List[str], str]],
    ground_truth: Dict[str, str],
    max_k: int = 25
) -> Dict[str, Any]:
    """
    Computes Mean Reciprocal Rank @ 25 across all molecules in ground_truth.

    Args:
        predictions: Dict mapping molecule_id to list of candidate SMILES,
                     or semicolon-joined string of candidate SMILES.
        ground_truth: Dict mapping molecule_id to true SMILES or InChIKey14.
        max_k: Maximum candidate cutoff (default 25).

    Returns:
        Dict containing mrr_25, top_1_acc, top_5_acc, top_25_acc, and per-item scores.
    """
    if not ground_truth:
        return {"mrr_25": 0.0, "count": 0}

    total_rr = 0.0
    top_1_hits = 0
    top_5_hits = 0
    top_25_hits = 0
    reciprocal_ranks: Dict[str, float] = {}
    first_hit_ranks: Dict[str, Optional[int]] = {}

    for mol_id, truth_val in ground_truth.items():
        candidates = predictions.get(mol_id, [])
        if isinstance(candidates, str):
            candidates = [s.strip() for s in candidates.split(";") if s.strip()]

        rr, hit_rank = compute_molecule_reciprocal_rank(candidates, truth_val, max_k=max_k)
        total_rr += rr
        reciprocal_ranks[mol_id] = rr
        first_hit_ranks[mol_id] = hit_rank

        if hit_rank == 1:
            top_1_hits += 1
        if hit_rank is not None and hit_rank <= 5:
            top_5_hits += 1
        if hit_rank is not None and hit_rank <= max_k:
            top_25_hits += 1

    n = len(ground_truth)
    return {
        "mrr_25": total_rr / n,
        "top_1_accuracy": top_1_hits / n,
        "top_5_accuracy": top_5_hits / n,
        "top_25_accuracy": top_25_hits / n,
        "num_molecules": n,
        "reciprocal_ranks": reciprocal_ranks,
        "first_hit_ranks": first_hit_ranks,
    }


def score_submission_file(submission_csv_path: str, ground_truth_csv_path: str) -> Dict[str, Any]:
    """Scores a generated submission.csv file against ground_truth.csv."""
    sub_df = pd.read_csv(submission_csv_path)
    gt_df = pd.read_csv(ground_truth_csv_path)

    preds = dict(zip(sub_df["molecule_id"], sub_df["smiles"]))
    gt = dict(zip(gt_df["molecule_id"], gt_df["smiles"]))

    return compute_mrr_25(preds, gt)
