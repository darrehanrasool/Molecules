"""FastAPI service: upload a spectrum → top-25 SMILES."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from src.config import pick_device
import numpy as np, torch
from src.config import CFG
from src.model import SpectrumFPNet
from src.retrieval import retrieve_topk
from src.utils import bin_spectrum

app = FastAPI(title="CASMI 2026 API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = pick_device()
ckpt = torch.load(CFG.MODEL_DIR / "best_model.pt", map_location=DEVICE)
MODEL = SpectrumFPNet(ckpt["in_dim"], CFG.FP_BITS).to(DEVICE)
MODEL.load_state_dict(ckpt["model"]); MODEL.eval()

class Spectrum(BaseModel):
    mzs: List[float]
    intensities: List[float]
    precursor_mz: float
    adduct: str = "[M+H]+"

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict")
def predict(s: Spectrum):
    if not s.mzs or len(s.mzs) != len(s.intensities):
        raise HTTPException(
            status_code=422,
            detail="mzs and intensities must be non-empty arrays of equal length.",
        )
    if not np.isfinite(s.precursor_mz) or s.precursor_mz <= 0:
        raise HTTPException(
            status_code=422,
            detail="precursor_mz must be a finite positive number.",
        )
    if not np.isfinite(s.mzs).all() or not np.isfinite(s.intensities).all():
        raise HTTPException(
            status_code=422,
            detail="Peak m/z and intensity values must all be finite numbers.",
        )
    if s.adduct not in CFG.ADDUCT_DELTA:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported adduct: {s.adduct}",
        )

    vec = bin_spectrum(s.mzs, s.intensities)
    x = np.concatenate([vec, [s.precursor_mz / 1000.0]]).astype(np.float32)
    with torch.no_grad():
        fp = torch.sigmoid(MODEL(torch.from_numpy(x).unsqueeze(0).to(DEVICE))).cpu().numpy()[0]
    smis, keys, sims = retrieve_topk(fp, s.precursor_mz, s.adduct, top_k=25)
    return {"candidates": [{"smiles": sm, "inchikey14": k, "tanimoto": v}
                           for sm, k, v in zip(smis, keys, sims)]}