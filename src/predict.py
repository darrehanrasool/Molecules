"""Predict top-25 SMILES per test molecule → submission.csv."""
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from collections import defaultdict
from src.config import pick_device
from src.config import CFG
from src.model import SpectrumFPNet
from src.retrieval import retrieve_topk
from src.utils import get_logger, set_seed

log = get_logger("predict")

def aggregate_fp(preds):
    """Median across spectra of one molecule."""
    return np.median(np.stack(preds), axis=0)

def main():
    set_seed()
    from src.config import pick_device
    device = pick_device()
    ckpt = torch.load(CFG.MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    model = SpectrumFPNet(ckpt["in_dim"], CFG.FP_BITS).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    log.info(f"Loaded model AP={ckpt.get('ap', 0):.4f}")

    groups = defaultdict(list)
    meta = {}
    chunk_dir = CFG.PROC_DIR / "test_chunks"
    test_files = sorted(chunk_dir.glob("test_*.npz"))
    if not test_files:
        test_file = CFG.PROC_DIR / "test.npz"
        if test_file.is_file():
            test_files = [test_file]
        else:
            raise FileNotFoundError(
                f"No processed test data found in {chunk_dir}. "
                "Run preprocessing first: "
                "python -m src.preprocessing"
            )

    for test_file in test_files:
        log.info(f"Predicting spectra from {test_file.name}")
        with np.load(test_file, allow_pickle=True) as te:
            spectra = te["spectra"].astype(np.float32, copy=False)
            pmzs = te["precursor_mz"]
            adducts = te["adduct"]
            mol_ids = te["molecule_id"]
            X = np.concatenate([
                spectra,
                (pmzs / 1000.0).reshape(-1, 1).astype(np.float32),
            ], axis=1)

            preds = []
            with torch.no_grad():
                for i in tqdm(range(0, len(X), 256), desc="infer"):
                    xb = torch.from_numpy(X[i:i + 256]).to(device)
                    preds.append(torch.sigmoid(model(xb)).cpu().numpy())
            preds = np.concatenate(preds, axis=0)

        for i, mid in enumerate(mol_ids):
            groups[mid].append(preds[i])
            meta[mid] = (pmzs[i], adducts[i])

    rows = []
    for mid, plist in tqdm(groups.items(), desc="retrieve"):
        fp = aggregate_fp(plist)
        pm, ad = meta[mid]
        smis, _, _ = retrieve_topk(fp, pm, ad, top_k=CFG.TOP_K)
        if len(smis) < CFG.TOP_K:
            while len(smis) < CFG.TOP_K: smis.append(smis[-1] if smis else "C")
        rows.append({"molecule_id": mid, "smiles": ";".join(smis)})

    sub = pd.DataFrame(rows)
    sub.to_csv(CFG.ROOT / "submission.csv", index=False)
    log.info(f"Wrote submission.csv ({len(sub)} rows).")

if __name__ == "__main__":
    main()