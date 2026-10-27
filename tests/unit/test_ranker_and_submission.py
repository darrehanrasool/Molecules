"""Unit tests for CandidateRanker and Submission validation."""

import pandas as pd
from molecules.postprocessing.ranker import CandidateRanker
from molecules.postprocessing.submission import (
    build_submission_dataframe,
    format_candidate_smiles,
    validate_submission_dataframe,
)


def test_candidate_ranker_deduplication(glucose_isomers):
    # Two stereoisomers of glucose with different confidence scores
    scored = [
        (glucose_isomers["achiral"], 0.95),
        (glucose_isomers["chiral"], 0.80), # Same InChIKey14, should be collapsed!
        ("CC(=O)Oc1ccccc1C(=O)O", 0.60),   # Aspirin
    ]

    ranker = CandidateRanker(max_candidates=25, deduplicate_by_inchikey14=True)
    deduped = ranker.rank_and_deduplicate(scored)

    # Should retain glucose only once (the higher-scoring one) and aspirin
    assert len(deduped) == 2
    assert deduped[0] == glucose_isomers["achiral"]
    assert deduped[1] == "CC(=O)Oc1ccccc1C(=O)O"


def test_format_candidate_smiles():
    cands = ["CC", "CCC", "CCCC"]
    formatted = format_candidate_smiles(cands, max_candidates=2)
    assert formatted == "CC;CCC"


def test_submission_validation_success():
    df = pd.DataFrame([
        {"molecule_id": "m_01", "smiles": "CC;CCC"},
        {"molecule_id": "m_02", "smiles": "CCO;CCN"},
    ])
    valid, errors = validate_submission_dataframe(df)
    assert valid
    assert len(errors) == 0


def test_submission_validation_failures():
    # Duplicate molecule_id and excessive candidates (> 25)
    too_many = ";".join([f"C{i}" for i in range(30)])
    df = pd.DataFrame([
        {"molecule_id": "m_01", "smiles": "CC"},
        {"molecule_id": "m_01", "smiles": too_many},
    ])
    valid, errors = validate_submission_dataframe(df)
    assert not valid
    assert any("duplicate" in e.lower() for e in errors)
    assert any("exceeding 25" in e.lower() for e in errors)
