"""Spectral models: Baselines, Similarity metrics, Retrieval engines, and Deep Encoders."""

from molecules.models.base import BaseMoleculePredictor
from molecules.models.similarity import (
    spectral_cosine_similarity,
    modified_cosine_similarity,
)
from molecules.models.retrieval import PrecursorRetrievalPredictor
from molecules.models.neural_encoder import (
    Spectral1DCNNEncoder,
    NeuralFingerprintPredictor,
)

__all__ = [
    "BaseMoleculePredictor",
    "spectral_cosine_similarity",
    "modified_cosine_similarity",
    "PrecursorRetrievalPredictor",
    "Spectral1DCNNEncoder",
    "NeuralFingerprintPredictor",
]
