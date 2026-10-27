"""Data loader for tandem mass spectrometry (MS/MS) datasets."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def parse_peak_string(peak_str: Union[str, List[Tuple[float, float]], np.ndarray]) -> np.ndarray:
    """
    Parses various serialized peak representations into an (N, 2) numpy array of [mz, intensity].

    Accepts:
        - "mz1:int1 mz2:int2 ..."
        - "mz1,int1;mz2,int2..."
        - List of (mz, intensity) tuples
    """
    if isinstance(peak_str, np.ndarray):
        return peak_str
    if isinstance(peak_str, list):
        return np.array(peak_str, dtype=np.float32)

    if not isinstance(peak_str, str) or not peak_str.strip():
        return np.empty((0, 2), dtype=np.float32)

    pairs = []
    # Try space-separated pairs with colon e.g. "105.03:0.42 153.01:1.0"
    if ":" in peak_str:
        tokens = peak_str.strip().split()
        for tok in tokens:
            if ":" in tok:
                parts = tok.split(":")
                try:
                    pairs.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    elif ";" in peak_str:
        # Semicolon separated e.g. "105.03,0.42;153.01,1.0"
        tokens = peak_str.strip().split(";")
        for tok in tokens:
            if "," in tok:
                parts = tok.split(",")
                try:
                    pairs.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue

    if not pairs:
        return np.empty((0, 2), dtype=np.float32)

    arr = np.array(pairs, dtype=np.float32)
    # Sort peaks by m/z ascending
    return arr[arr[:, 0].argsort()]


class MassSpectrumRecord:
    """Represents a single query or reference LC-MS/MS spectrum."""

    def __init__(
        self,
        molecule_id: str,
        precursor_mz: float,
        adduct: str = "[M+H]+",
        peaks: Optional[np.ndarray] = None,
        collision_energy: Optional[float] = None,
        smiles: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.molecule_id = molecule_id
        self.precursor_mz = float(precursor_mz)
        self.adduct = adduct
        self.peaks = peaks if peaks is not None else np.empty((0, 2), dtype=np.float32)
        self.collision_energy = collision_energy
        self.smiles = smiles
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"MassSpectrumRecord(id={self.molecule_id}, precursor={self.precursor_mz:.4f}, "
            f"adduct={self.adduct}, peaks={len(self.peaks)})"
        )


def load_spectra_dataframe(path: Union[str, Path]) -> List[MassSpectrumRecord]:
    """Loads a collection of spectra from Parquet or CSV."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    records: List[MassSpectrumRecord] = []
    for _, row in df.iterrows():
        peaks_raw = row.get("peaks", "")
        parsed_peaks = parse_peak_string(peaks_raw)
        rec = MassSpectrumRecord(
            molecule_id=str(row["molecule_id"]),
            precursor_mz=float(row.get("precursor_mz", 0.0)),
            adduct=str(row.get("adduct", "[M+H]+")),
            peaks=parsed_peaks,
            collision_energy=row.get("collision_energy"),
            smiles=str(row["smiles"]) if "smiles" in row and pd.notna(row["smiles"]) else None,
        )
        records.append(rec)
    return records
