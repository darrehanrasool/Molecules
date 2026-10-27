"""Precursor-Constrained Candidate Retrieval Engine."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from molecules.data.loader import MassSpectrumRecord
from molecules.data.preprocessor import SpectralPreprocessor
from molecules.models.base import BaseMoleculePredictor
from molecules.models.similarity import spectral_cosine_similarity
from molecules.utils.chemistry import (
    precursor_mz_to_neutral_mass,
    ppm_error,
    within_ppm_tolerance,
)


class PrecursorRetrievalPredictor(BaseMoleculePredictor):
    """
    Retrieval model that:
    1. Filters candidate molecules by neutral mass within a PPM tolerance window.
    2. Scores matching candidates using spectral cosine or precursor mass proximity.
    3. Ranks and returns top-K candidates.
    """

    def __init__(
        self,
        candidate_catalog: Optional[pd.DataFrame] = None,
        ppm_tolerance: float = 15.0,
        preprocessor: Optional[SpectralPreprocessor] = None,
    ):
        self.ppm_tolerance = ppm_tolerance
        self.preprocessor = preprocessor or SpectralPreprocessor()
        self.catalog = candidate_catalog

    def set_catalog(self, catalog_df: pd.DataFrame) -> None:
        """Sets or updates the candidate molecule database."""
        self.catalog = catalog_df

    def predict_candidates(
        self, spectrum: MassSpectrumRecord, top_k: int = 25
    ) -> List[Tuple[str, float]]:
        """
        Retrieves top_k candidate SMILES for the query spectrum.
        """
        if self.catalog is None or len(self.catalog) == 0:
            return []

        query_neutral_mass = precursor_mz_to_neutral_mass(spectrum.precursor_mz, spectrum.adduct)
        clean_query_peaks = self.preprocessor.process(spectrum.peaks, spectrum.precursor_mz)

        # Filter candidates within PPM tolerance
        cat = self.catalog
        if "neutral_mass" in cat.columns:
            masses = cat["neutral_mass"].values
            ppm_errors = np.abs(masses - query_neutral_mass) / query_neutral_mass * 1e6
            match_mask = ppm_errors <= self.ppm_tolerance
            candidates = cat[match_mask].copy()
            candidates["ppm_error"] = ppm_errors[match_mask]
        else:
            candidates = cat.copy()
            candidates["ppm_error"] = 0.0

        if len(candidates) == 0:
            # Fallback to closest 50 candidates by mass if no candidate is within ppm
            if "neutral_mass" in cat.columns:
                mass_diffs = np.abs(cat["neutral_mass"].values - query_neutral_mass)
                closest_indices = np.argsort(mass_diffs)[:50]
                candidates = cat.iloc[closest_indices].copy()
                candidates["ppm_error"] = mass_diffs[closest_indices] / query_neutral_mass * 1e6
            else:
                candidates = cat.head(50).copy()
                candidates["ppm_error"] = 0.0

        # Score candidates
        scored_candidates: List[Tuple[str, float]] = []
        for _, row in candidates.iterrows():
            cand_smiles = str(row["smiles"])
            score = 1.0 / (1.0 + float(row.get("ppm_error", 0.0)) * 0.1)

            # If reference peaks are available in catalog, add spectral cosine
            if "peaks" in row and pd.notna(row["peaks"]):
                from molecules.data.loader import parse_peak_string
                cand_peaks = parse_peak_string(row["peaks"])
                cand_clean = self.preprocessor.process(cand_peaks)
                cos_sim = spectral_cosine_similarity(clean_query_peaks, cand_clean)
                score = 0.5 * score + 0.5 * cos_sim

            scored_candidates.append((cand_smiles, score))

        # Sort descending by confidence score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]
