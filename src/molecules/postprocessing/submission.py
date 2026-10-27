"""
Kaggle Submission Generator and Quality Auditor for CASMI 2026.
Ensures zero-defect compliance with competition submission constraints.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd


def format_candidate_smiles(candidates: List[str], max_candidates: int = 25) -> str:
    """Joins up to max_candidates SMILES strings using semicolons."""
    selected = [s.strip() for s in candidates[:max_candidates] if s.strip()]
    return ";".join(selected)


def build_submission_dataframe(
    predictions_map: Dict[str, Union[List[str], str]],
    expected_molecule_ids: Optional[List[str]] = None,
    max_candidates: int = 25,
) -> pd.DataFrame:
    """
    Constructs a validated submission DataFrame with columns [molecule_id, smiles].
    """
    rows = []
    mol_ids = expected_molecule_ids if expected_molecule_ids is not None else list(predictions_map.keys())

    for mid in mol_ids:
        raw_val = predictions_map.get(mid, [])
        if isinstance(raw_val, list):
            smiles_str = format_candidate_smiles(raw_val, max_candidates=max_candidates)
        elif isinstance(raw_val, str):
            cands = [s.strip() for s in raw_val.split(";") if s.strip()]
            smiles_str = format_candidate_smiles(cands, max_candidates=max_candidates)
        else:
            smiles_str = ""

        rows.append({"molecule_id": str(mid), "smiles": smiles_str})

    return pd.DataFrame(rows)


def validate_submission_dataframe(
    df: pd.DataFrame, expected_molecule_ids: Optional[Set[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validates submission against all Kaggle competition rules:
    - Columns must be exactly ['molecule_id', 'smiles'].
    - No null values.
    - No empty rows.
    - Every molecule_id must appear exactly once.
    - At most 25 candidates per entry.
    """
    errors: List[str] = []

    # Check columns
    if list(df.columns) != ["molecule_id", "smiles"]:
        errors.append(f"Invalid columns: expected ['molecule_id', 'smiles'], got {list(df.columns)}")

    # Check nulls
    if df["molecule_id"].isnull().any():
        errors.append("Detected null values in molecule_id column")
    if df["smiles"].isnull().any():
        errors.append("Detected null values in smiles column")

    # Check duplicates
    if df["molecule_id"].duplicated().any():
        num_dups = df["molecule_id"].duplicated().sum()
        errors.append(f"Detected {num_dups} duplicate molecule_id entries")

    # Check empty smiles
    empty_mask = df["smiles"].astype(str).str.strip() == ""
    if empty_mask.any():
        errors.append(f"Detected {empty_mask.sum()} empty predictions in smiles column")

    # Check max candidates count
    counts = df["smiles"].astype(str).apply(lambda x: len(x.split(";")))
    excessive = counts > 25
    if excessive.any():
        errors.append(f"Detected rows exceeding 25 candidates: {excessive.sum()} rows")

    # Check coverage of expected IDs
    if expected_molecule_ids is not None:
        present_ids = set(df["molecule_id"].astype(str))
        missing = expected_molecule_ids - present_ids
        if missing:
            errors.append(f"Missing {len(missing)} expected molecule_ids from test set")

    is_valid = len(errors) == 0
    return is_valid, errors


def save_and_verify_submission(
    df: pd.DataFrame, output_path: str, expected_molecule_ids: Optional[Set[str]] = None
) -> None:
    """Validates and writes submission.csv to disk."""
    is_valid, errors = validate_submission_dataframe(df, expected_molecule_ids)
    if not is_valid:
        error_msg = "\n".join(errors)
        raise ValueError(f"Submission failed validation:\n{error_msg}")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
