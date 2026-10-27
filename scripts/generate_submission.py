#!/usr/bin/env python3
"""Submission generator and quality auditor for Enveda CASMI 2026."""

import argparse
from pathlib import Path
import pandas as pd
from molecules.postprocessing.submission import (
    validate_submission_dataframe,
    save_and_verify_submission,
)
from molecules.utils.logging import get_logger

logger = get_logger("generate_submission")


def main():
    parser = argparse.ArgumentParser(description="Generate and validate Kaggle submission.csv.")
    parser.add_argument("--input", type=str, default="outputs/predictions/sample_submission.csv")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--verify", action="store_true", help="Run strict zero-defect audit only")
    args = parser.parse_args()

    input_p = Path(args.input)
    if not input_p.exists():
        logger.error(f"Input predictions file not found: {input_p}")
        return

    df = pd.read_csv(input_p)
    logger.info(f"Loaded {len(df)} predictions from {input_p}.")

    is_valid, errors = validate_submission_dataframe(df)
    if not is_valid:
        logger.error("Submission verification FAILED:")
        for err in errors:
            logger.error(f"  - {err}")
        return

    logger.info("[✓] Submission passed all Kaggle formatting constraints.")
    if not args.verify:
        save_and_verify_submission(df, args.output)
        logger.info(f"[+] Verified submission saved to: {args.output}")


if __name__ == "__main__":
    main()
