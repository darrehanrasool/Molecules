"""Abstract Base Class for Molecule Identification Predictors."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from molecules.data.loader import MassSpectrumRecord


class BaseMoleculePredictor(ABC):
    """Abstract interface for spectral molecule candidate predictors."""

    @abstractmethod
    def predict_candidates(
        self, spectrum: MassSpectrumRecord, top_k: int = 25
    ) -> List[Tuple[str, float]]:
        """
        Given a query spectrum, returns a ranked list of (smiles, confidence_score) tuples,
        ordered by highest confidence first, with at most top_k items.
        """
        pass

    def batch_predict(
        self, spectra: List[MassSpectrumRecord], top_k: int = 25
    ) -> Dict[str, List[str]]:
        """
        Predicts top_k candidate SMILES for a collection of spectra.
        Returns a dict mapping molecule_id -> list of SMILES strings.
        """
        results: Dict[str, List[str]] = {}
        for spec in spectra:
            ranked_pairs = self.predict_candidates(spec, top_k=top_k)
            results[spec.molecule_id] = [smiles for smiles, _ in ranked_pairs]
        return results
