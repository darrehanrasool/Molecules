"""Retrieve top-K candidate SMILES given a predicted fingerprint + precursor mass."""
import numpy as np
from pathlib import Path
from src.config import CFG
from src.utils import estimate_neutral_mass

_FP_DB = None
_CANDIDATE_MASSES = None

def load_fp_db():
    global _FP_DB, _CANDIDATE_MASSES
    if _FP_DB is None:
        with np.load(CFG.FP_DB, allow_pickle=True) as data:
            _FP_DB = {
                "fps": data["fps"],
                "smiles": data["smiles"],
                "inchikey14": data["inchikey14"],
            }
        
        # Load precomputed exact monoisotopic masses if available
        mass_file = CFG.PROC_DIR / "candidate_masses.npy"
        if mass_file.is_file():
            _CANDIDATE_MASSES = np.load(mass_file)
        else:
            _CANDIDATE_MASSES = None
    return _FP_DB

def tanimoto_matrix(query_fp: np.ndarray, db_fps: np.ndarray):
    """query_fp (B,), db_fps (N, B) — Tanimoto on bits or continuous probabilities."""
    inter = db_fps @ query_fp
    a = db_fps.sum(1)
    b = query_fp.sum()
    union = a + b - inter + 1e-9
    return inter / union

def retrieve_topk(query_fp, precursor_mz, adduct, top_k=CFG.TOP_K, ppm=CFG.PPM_TOL):
    """
    Retrieve top-K candidates using precursor mass filtering (±ppm) and Tanimoto ranking.
    Deduplicates candidates by InChIKey14 to conserve top-25 budget.
    """
    db = load_fp_db()
    neutral = estimate_neutral_mass(precursor_mz, adduct)
    
    candidate_indices = None
    if neutral is not None and _CANDIDATE_MASSES is not None:
        # Physical mass filter: |candidate_mass - neutral| <= neutral * ppm * 1e-6
        tol = neutral * (ppm * 1e-6)
        mask = np.abs(_CANDIDATE_MASSES - neutral) <= tol
        
        # If strict tolerance has at least top_k candidates, restrict search space
        if mask.sum() >= top_k:
            candidate_indices = np.where(mask)[0]
        else:
            # Relax to 30 ppm
            relaxed_tol = neutral * (30.0 * 1e-6)
            mask_relaxed = np.abs(_CANDIDATE_MASSES - neutral) <= relaxed_tol
            if mask_relaxed.sum() >= top_k:
                candidate_indices = np.where(mask_relaxed)[0]

    if candidate_indices is not None and len(candidate_indices) > 0:
        sub_fps = db["fps"][candidate_indices]
        sims = tanimoto_matrix(query_fp.astype(np.float32), sub_fps)
        order = np.argsort(-sims)
        selected_db_indices = candidate_indices[order]
        selected_sims = sims[order]
    else:
        # Fallback to global search across all 275k candidates
        sims = tanimoto_matrix(query_fp.astype(np.float32), db["fps"])
        order = np.argsort(-sims)[:top_k * 6]
        selected_db_indices = order
        selected_sims = sims[order]

    seen = set()
    out_smi, out_key, out_sim = [], [], []
    for idx, sim in zip(selected_db_indices, selected_sims):
        k = db["inchikey14"][idx]
        if k in seen:
            continue
        seen.add(k)
        out_smi.append(str(db["smiles"][idx]))
        out_key.append(str(k))
        out_sim.append(float(sim))
        if len(out_smi) >= top_k:
            break

    # If deduplication dropped below top_k, fill up to top_k
    if len(out_smi) < top_k:
        for idx, sim in zip(selected_db_indices, selected_sims):
            smi = str(db["smiles"][idx])
            if smi not in out_smi:
                out_smi.append(smi)
                out_key.append(str(db["inchikey14"][idx]))
                out_sim.append(float(sim))
            if len(out_smi) >= top_k:
                break

    return out_smi, out_key, out_sim
