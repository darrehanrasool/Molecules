"""Retrieve top-K candidate SMILES given a predicted fingerprint + precursor mass."""
import numpy as np
from src.config import CFG
from src.utils import estimate_neutral_mass

_FP_DB = None
def load_fp_db():
    global _FP_DB
    if _FP_DB is None:
        with np.load(CFG.FP_DB, allow_pickle=True) as data:
            _FP_DB = {
                "fps": data["fps"],
                "smiles": data["smiles"],
                "inchikey14": data["inchikey14"],
            }
    return _FP_DB

def tanimoto_matrix(query_fp: np.ndarray, db_fps: np.ndarray):
    """query_fp (B,), db_fps (N, B) — Tanimoto on bits."""
    inter = db_fps @ query_fp
    a = db_fps.sum(1); b = query_fp.sum()
    union = a + b - inter + 1e-9
    return inter / union

def retrieve_topk(query_fp, precursor_mz, adduct, top_k=CFG.TOP_K, ppm=CFG.PPM_TOL):
    db = load_fp_db()
    sims = tanimoto_matrix(query_fp.astype(np.float32), db["fps"])

    neutral = estimate_neutral_mass(precursor_mz, adduct)
    if neutral is not None:
        # We don't have exact mass in DB → approximate by len of SMILES not possible
        # Instead rely on fingerprint similarity; mass filter can be added if a mass DB is built.
        pass

    order = np.argsort(-sims)[:top_k * 4]
    seen, out_smi, out_key, out_sim = set(), [], [], []
    for idx in order:
        k = db["inchikey14"][idx]
        if k in seen: continue
        seen.add(k)
        out_smi.append(str(db["smiles"][idx]))
        out_key.append(str(k))
        out_sim.append(float(sims[idx]))
        if len(out_smi) >= top_k: break
    return out_smi, out_key, out_sim