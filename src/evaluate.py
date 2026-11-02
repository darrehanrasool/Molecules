"""Evaluate submission against ground truth (works when labels are known)."""
import numpy as np, pandas as pd
from collections import defaultdict
from src.config import CFG
from src.utils import smiles_to_inchikey14, get_logger

log = get_logger("evaluate")

def mrr_at_25(pred_lists, true_keys, k=25):
    rr = []
    for preds, tkey in zip(pred_lists, true_keys):
        score = 0.0
        for i, p in enumerate(preds[:k]):
            if smiles_to_inchikey14(p) == tkey:
                score = 1.0 / (i + 1); break
        rr.append(score)
    return float(np.mean(rr)), rr

def main():
    """Only run when test labels are available (local sanity check)."""
    sub = pd.read_csv(CFG.ROOT / "submission.csv")
    sub["preds"] = sub["smiles"].str.split(";")

    chunk_dir = CFG.PROC_DIR / "test_chunks"
    test_files = sorted(chunk_dir.glob("test_*.npz"))
    if not test_files and (CFG.PROC_DIR / "test.npz").is_file():
        test_files = [CFG.PROC_DIR / "test.npz"]
    if not test_files:
        raise FileNotFoundError(
            f"No processed test data found in {chunk_dir}. "
            "Run preprocessing first."
        )

    truth = {}
    for test_file in test_files:
        with np.load(test_file, allow_pickle=True) as te:
            if "inchikey14" not in te:
                log.warning(
                    "Test data has no inchikey14 labels; "
                    "MRR cannot be computed for unlabeled test data."
                )
                return
            truth.update(zip(te["molecule_id"], te["inchikey14"]))

    if not truth:
        log.warning("No test labels found — cannot compute MRR.")
        return

    molecule_ids, pred_lists, truth_keys = [], [], []
    for _, r in sub.iterrows():
        if r["molecule_id"] in truth:
            molecule_ids.append(r["molecule_id"])
            pred_lists.append(r["preds"]); truth_keys.append(truth[r["molecule_id"]])
    mrr, per = mrr_at_25(pred_lists, truth_keys)
    log.info(f"MRR@25 = {mrr:.4f}  (n={len(per)})")
    pd.DataFrame({"molecule_id": molecule_ids,
                  "reciprocal_rank": per}).to_csv(CFG.LOG_DIR / "per_molecule_rr.csv", index=False)

if __name__ == "__main__":
    main()