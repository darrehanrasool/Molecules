"""Unit tests for MRR@25 competition evaluation metric."""

import pytest
from molecules.evaluation.metric import (
    compute_molecule_reciprocal_rank,
    compute_mrr_25,
)


def test_mrr_at_rank_1(glucose_isomers):
    truth = glucose_isomers["chiral"]
    candidates = [glucose_isomers["achiral"], "CC(=O)Oc1ccccc1C(=O)O"]
    rr, rank = compute_molecule_reciprocal_rank(candidates, truth, max_k=25)
    assert rank == 1
    assert pytest.approx(rr, 1e-5) == 1.0


def test_mrr_at_rank_2(glucose_isomers):
    truth = glucose_isomers["chiral"]
    candidates = ["CC(=O)Oc1ccccc1C(=O)O", glucose_isomers["achiral"]]
    rr, rank = compute_molecule_reciprocal_rank(candidates, truth, max_k=25)
    assert rank == 2
    assert pytest.approx(rr, 1e-5) == 0.5


def test_mrr_at_rank_25(glucose_isomers):
    truth = glucose_isomers["chiral"]
    # 24 incorrect candidates followed by 1 correct
    candidates = [f"C{i}H{2*i+2}" for i in range(1, 25)] + [glucose_isomers["achiral"]]
    rr, rank = compute_molecule_reciprocal_rank(candidates, truth, max_k=25)
    assert rank == 25
    assert pytest.approx(rr, 1e-5) == 0.04


def test_mrr_beyond_rank_25(glucose_isomers):
    truth = glucose_isomers["chiral"]
    # 25 incorrect candidates followed by 1 correct at position 26
    candidates = [f"C{i}H{2*i+2}" for i in range(1, 26)] + [glucose_isomers["achiral"]]
    rr, rank = compute_molecule_reciprocal_rank(candidates, truth, max_k=25)
    assert rank is None
    assert rr == 0.0


def test_mrr_aggregate_calculation(glucose_isomers):
    ground_truth = {
        "m_01": glucose_isomers["chiral"], # will hit at rank 1 -> 1.0
        "m_02": glucose_isomers["chiral"], # will hit at rank 2 -> 0.5
        "m_03": glucose_isomers["chiral"], # will miss -> 0.0
    }
    predictions = {
        "m_01": [glucose_isomers["achiral"]],
        "m_02": ["CC", glucose_isomers["achiral"]],
        "m_03": ["CCO", "CCN"],
    }

    metrics = compute_mrr_25(predictions, ground_truth, max_k=25)
    expected_mrr = (1.0 + 0.5 + 0.0) / 3.0
    assert pytest.approx(metrics["mrr_25"], 1e-5) == expected_mrr
    assert pytest.approx(metrics["top_1_accuracy"], 1e-5) == 1.0 / 3.0
    assert pytest.approx(metrics["top_5_accuracy"], 1e-5) == 2.0 / 3.0
    assert pytest.approx(metrics["top_25_accuracy"], 1e-5) == 2.0 / 3.0
