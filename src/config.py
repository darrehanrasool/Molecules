"""Global configuration for CASMI 2026."""
import os
from pathlib import Path
# ---------------- Device helper (cross-platform) ----------------
import torch

class CFG:
    # Paths
    ROOT       = Path(__file__).resolve().parent.parent
    RUN_DIR    = Path(os.environ.get("CASMI_RUN_DIR", ROOT))
    DATA_DIR   = ROOT / "data"
    RAW_DIR    = DATA_DIR / "raw"
    PROC_DIR   = DATA_DIR / "processed"
    EXT_DIR    = DATA_DIR / "external"
    MODEL_DIR  = RUN_DIR / "models"
    LOG_DIR    = RUN_DIR / "logs"
    GRAPH_DIR  = RUN_DIR / "graphs"
    DOC_DIR    = ROOT / "docs"

    for p in [DATA_DIR, RAW_DIR, PROC_DIR, EXT_DIR, MODEL_DIR, LOG_DIR, GRAPH_DIR, DOC_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    # Data files
    TRAIN_PARQUET = RAW_DIR / "train.parquet"
    TEST_PARQUET  = RAW_DIR / "test.parquet"
    TRAIN_PROC    = PROC_DIR / "train_processed.parquet"
    TEST_PROC     = PROC_DIR / "test_processed.parquet"
    FP_DB         = PROC_DIR / "fp_database.npz"

    # Spectrum binarization
    MAX_MZ   = 1200.0
    BIN_SIZE = 0.1
    N_BINS   = int(MAX_MZ / BIN_SIZE)   # 12000

    # Fingerprint
    FP_BITS  = 2048
    FP_RADIUS = 2

    # Model
    HIDDEN_1 = 2048
    HIDDEN_2 = 1024
    DROPOUT  = 0.3
    LR       = 3e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS   = 60
    BATCH    = 128
    NUM_WORKERS = 8        # 20-core machine → safe
    SEED     = 42

    # Retrieval
    TOP_K    = 25
    PPM_TOL  = 15.0        # precursor mass tolerance for candidate filter

    # Adduct proton masses for neutral-mass estimation
    PROTON = 1.007276
    ADDUCT_DELTA = {
        "[M+H]+":    PROTON,
        "[M+NH4]+":  PROTON + 17.026549,
        "[M-H2O+H]+": PROTON - 18.010565,
        "[M-2H2O+H]+": PROTON - 2 * 18.010565,
        "[M+Na]+":   22.989218,
        "[M+K]+":    38.963158,
        "[M-H]-":    -PROTON,
        "[M-H2O-H]-": -PROTON - 18.010565,
        "[M+CH2O2-H]-": -PROTON + 46.00548,
        "[M+Cl]-":   34.968853,
    }
def pick_device():
    """cuda → mps (Apple Silicon) → cpu, first available wins."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")