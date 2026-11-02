# CASMI 2026 — Molecule ID from MS/MS Spectra

Predict 2D chemical structures (SMILES) from LC-MS/MS spectra of natural products.

## Pipeline

1. **Preprocess** spectra → binned vectors
2. **Neural network** predicts Morgan fingerprint (2048 bits) from binned spectrum
3. **Retrieval** in training fingerprint DB (Tanimoto) → top-25 candidates
4. **Re-rank** by precursor-mass agreement + fingerprint similarity
5. **Evaluate** with MRR@25

Training refreshes charts 01–07 after each completed epoch. After training,
`python -m src.visualize` generates the full set of 10 charts under `graphs/`;
the data-distribution charts use a reproducible sample of the processed chunks.

## Quick Start

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.preprocessing
python -m src.train
python -m src.predict
python -m src.evaluate
python -m src.visualize
streamlit run frontend/app.py
uvicorn backend.api:app --reload --port 8000
```

### NVIDIA GPU training on Windows

To keep the default environment unchanged, create a separate CUDA-enabled
environment and install the project requirements from the PyTorch CUDA 11.8
wheel index:

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
.\.venv-gpu\Scripts\python.exe -m src.train
```

Run training only after stopping any existing training process. The training
code selects CUDA automatically when the installed PyTorch build can access it.
To continue a stopped run, use
`.\.venv-gpu\Scripts\python.exe -m src.train --resume`. New runs save a
full resume checkpoint after each completed epoch. For older runs that only
have `best_model.pt` and `history.json`, resume is allowed only when the
checkpoint matches the latest completed epoch; model weights resume, but
optimizer state starts fresh because it was not saved in the old checkpoint.
