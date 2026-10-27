"""Integration test running end-to-end spectrum prediction and scoring pipeline."""

import pandas as pd
from molecules.models.retrieval import PrecursorRetrievalPredictor
from molecules.postprocessing.ranker import CandidateRanker
from molecules.postprocessing.submission import (
    build_submission_dataframe,
    validate_submission_dataframe,
)
from molecules.evaluation.metric import compute_mrr_25


def test_end_to_end_pipeline(sample_spectrum):
    # Setup candidate database containing true compound and distractors
    catalog = pd.DataFrame([
        {
            "smiles": sample_spectrum.smiles, # Chrysin
            "neutral_mass": 254.0579,
            "peaks": "105.03:0.42 153.01:0.85 254.05:1.0",
        },
        {
            "smiles": "OCC1OC(O)C(O)C(O)C1O", # Glucose
            "neutral_mass": 180.0634,
            "peaks": "60.02:0.5 73.03:1.0",
        },
        {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O", # Aspirin
            "neutral_mass": 180.0423,
            "peaks": "120.02:0.8 138.03:1.0",
        },
    ])

    # 1. Retrieval
    retriever = PrecursorRetrievalPredictor(candidate_catalog=catalog, ppm_tolerance=20.0)
    scored_candidates = retriever.predict_candidates(sample_spectrum, top_k=25)
    assert len(scored_candidates) > 0

    # 2. Reranker & Deduplication
    ranker = CandidateRanker(max_candidates=25)
    top_smiles = ranker.rank_and_deduplicate(scored_candidates)
    assert len(top_smiles) > 0
    # True Chrysin should be retrieved at rank 1 due to neutral mass match (254.0579)
    assert top_smiles[0] == sample_spectrum.smiles

    # 3. Build Submission DataFrame
    predictions_map = {sample_spectrum.molecule_id: top_smiles}
    sub_df = build_submission_dataframe(predictions_map)

    # 4. Validate Submission
    valid, errors = validate_submission_dataframe(sub_df)
    assert valid, f"Validation errors: {errors}"

    # 5. Compute Metric
    ground_truth = {sample_spectrum.molecule_id: sample_spectrum.smiles}
    eval_results = compute_mrr_25(predictions_map, ground_truth)
    assert eval_results["mrr_25"] == 1.0
    assert eval_results["top_1_accuracy"] == 1.0
