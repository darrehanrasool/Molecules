#!/usr/bin/env python3
"""CLI utility to compute MRR@25 on predictions or submission files."""

import argparse
from pathlib import Path
from molecules.evaluation.metric import score_submission_file
from molecules.utils.logging import get_logger

logger = get_logger("evaluate_metric")


def main():
    parser = argparse.ArgumentParser(description="Evaluate submission against ground truth using MRR@25.")
    parser.add_argument("--submission", type=str, default="outputs/predictions/sample_submission.csv")
    parser.add_argument("--ground-truth", type=str, default="tests/fixtures/sample_submission.csv")
    args = parser.parse_args()

    sub_p = Path(args.submission)
    gt_p = Path(args.ground_truth)

    if not sub_p.exists():
        logger.error(f"Submission file not found: {sub_p}")
        return
    if not gt_p.exists():
        logger.error(f"Ground truth file not found: {gt_p}")
        return

    logger.info(f"Evaluating {sub_p.name} against {gt_p.name} on MRR@25...")
    results = score_submission_file(str(sub_p), str(gt_p))

    print("=" * 50)
    print(f"Evaluated Molecules: {results['num_molecules']}")
    print(f"MRR@25 Score       : {results['mrr_25']:.4f}")
    print(f"Top-1 Accuracy     : {results['top_1_accuracy']:.4f}")
    print(f"Top-5 Accuracy     : {results['top_5_accuracy']:.4f}")
    print(f"Top-25 Accuracy    : {results['top_25_accuracy']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
