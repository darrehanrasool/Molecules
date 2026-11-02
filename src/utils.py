"""Utilities: seeding, logging, and spectrum helpers."""

import os
import random
import logging
import sys

import numpy as np
import torch

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    Chem = None
    AllChem = None
    RDLogger = None

from src.config import CFG


def set_seed(seed=CFG.SEED):
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name="casmi"):
    """Create console + file logger."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        "%H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(
        CFG.LOG_DIR / f"{name}.log",
        encoding="utf-8"
    )
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


def smiles_to_morgan(
    smiles: str,
    n_bits=CFG.FP_BITS,
    radius=CFG.FP_RADIUS
):
    """Return Morgan fingerprint as uint8 vector, or None if invalid."""

    if smiles is None:
        return None

    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits
    )

    arr = np.zeros(n_bits, dtype=np.uint8)

    for i in fp.GetOnBits():
        arr[i] = 1

    return arr


def smiles_to_inchikey14(smiles: str):
    """Convert SMILES to the first 14 characters of InChIKey."""

    if smiles is None:
        return None

    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        mol = Chem.MolFromSmiles(
            Chem.MolToSmiles(mol)
        )

        key = Chem.MolToInchiKey(mol)

        if key is None:
            return None

        return key.split("-")[0]

    except Exception:
        return None


def estimate_neutral_mass(
    precursor_mz: float,
    adduct: str
):
    """Estimate neutral mass from precursor m/z and adduct."""

    if precursor_mz is None or adduct is None:
        return None

    try:
        precursor_mz = float(precursor_mz)
    except (TypeError, ValueError):
        return None

    d = CFG.ADDUCT_DELTA.get(adduct)

    if d is None:
        return None

    z = 1

    return (precursor_mz - d) * z


def bin_spectrum(mzs, intensities):
    """
    Convert a peak list into a dense binned spectrum.

    Returns:
        np.ndarray with shape (CFG.N_BINS,)
        dtype=float32
    """

    # Allocate exactly one spectrum vector.
    out = np.zeros(
        CFG.N_BINS,
        dtype=np.float32
    )

    if mzs is None or intensities is None:
        return out

    if len(mzs) == 0 or len(intensities) == 0:
        return out

    mzs = np.asarray(
        mzs,
        dtype=np.float32
    )

    intensities = np.asarray(
        intensities,
        dtype=np.float32
    )

    # Make sure both arrays have the same usable length.
    n = min(len(mzs), len(intensities))

    if n == 0:
        return out

    mzs = mzs[:n]
    intensities = intensities[:n]

    # Remove invalid values.
    valid = (
        np.isfinite(mzs)
        & np.isfinite(intensities)
    )

    if not np.any(valid):
        return out

    mzs = mzs[valid]
    intensities = intensities[valid]

    # Remove negative intensities.
    intensities = np.clip(
        intensities,
        0.0,
        None
    )

    # sqrt intensity transform.
    intensities = np.sqrt(
        intensities
    ).astype(
        np.float32,
        copy=False
    )

    # Convert m/z values to bin indices.
    idx = np.floor(
        mzs / CFG.BIN_SIZE
    ).astype(np.int64)

    # Keep only valid bins.
    valid_bins = (
        (idx >= 0)
        & (idx < CFG.N_BINS)
    )

    if not np.any(valid_bins):
        return out

    idx = idx[valid_bins]
    intensities = intensities[valid_bins]

    # Add intensities into bins.
    np.add.at(
        out,
        idx,
        intensities
    )

    # Normalize safely.
    max_value = np.max(out)

    if np.isfinite(max_value) and max_value > 0:
        out /= max_value

    return out