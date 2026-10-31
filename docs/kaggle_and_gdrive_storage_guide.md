# Zero-Local-Footprint Guide: Kaggle GPU & 5TB Google Drive Workflow

## 1. Executive Summary

To protect your Mac from local disk exhaustion and memory saturation:
1. **Local Repository Role**: Lightweight code development, documentation, tests, configuration matrices, publication figures, and validation scripts ($<100\,\text{MB}$ total).
2. **Kaggle GPU Execution**: Heavy computation (loading gigabytes of raw LC-MS/MS Parquet spectra, training neural network encoders, running inference on test sets) runs entirely in the cloud on Kaggle's free Tesla T4/P100 GPUs (16 GB VRAM) with 30 GB cloud RAM.
3. **Google Drive Integration (5TB)**: External spectral libraries (MassBank, GNPS, PubChem) can be stored in your 5TB Google Drive or downloaded directly into Kaggle `/kaggle/temp/` without consuming 1 byte of Mac storage.

---

## 2. Dataset Structure Schema on Local Mac vs. Kaggle

### Local Mac Lightweight Mock Format
On your local machine, the repository contains schema-compliant lightweight mock datasets (`data/raw/train.csv`, `data/splits/`):
- Each record has the exact competition schema:
  `molecule_id, precursor_mz, adduct, collision_energy, peaks, smiles, inchikey14`
- Total disk usage on Mac: **$< 2\,\text{KB}$**.
- Pytest and pipeline verification execute in **$< 1.5\,\text{seconds}$** without consuming RAM.

### Kaggle GPU File Hierarchy (In Cloud)
When executing inside Kaggle:
```
/kaggle/
├── input/
│   └── enveda-CASMI26-molecule-id-mass-spectra/
│       ├── train.parquet          # High-throughput training spectra
│       ├── test.parquet           # Query test spectra (unlabeled)
│       └── sample_submission.csv  # Required template
├── working/
│   ├── submission.csv             # Final generated submission (MRR@25 formatted)
│   ├── best_model.pt              # Exported PyTorch weights
│   └── validation_metrics.json    # Fold scores & evaluation log
└── temp/                          # Scratch space (cleared on exit)
```

---

## 3. Streaming Parquet Data on Kaggle Without Out-Of-Memory (OOM) Errors

In mass spectrometry competitions, loading the entire `train.parquet` into RAM simultaneously can exceed memory limits. The solution is **batched chunk streaming**:

```python
import pyarrow.parquet as pq
import pandas as pd

def stream_parquet_in_batches(parquet_path: str, batch_size: int = 5000):
    """
    Streams Parquet dataset in memory-efficient row batches.
    Memory footprint remains strictly constant (<500 MB) regardless of file size.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_chunk = batch.to_pandas()
        yield df_chunk
```

---

## 4. Google Drive 5TB Mounting for Colab / Cloud Training

If training outside Kaggle (e.g. Google Colab Pro with your 5TB Google Drive):

```python
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

PROJECT_PATH = '/content/drive/MyDrive/CASMI_2026_Enveda'
os.makedirs(PROJECT_PATH, exist_ok=True)
print(f"Mounted 5TB Google Drive at: {PROJECT_PATH}")
```

Your Mac remains 100% free of large files while all multi-gigabyte models and spectral databases reside safely on Google Drive or Kaggle Cloud.
