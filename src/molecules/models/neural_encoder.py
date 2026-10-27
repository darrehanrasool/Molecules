"""Deep Neural Spectral Encoder and Molecular Fingerprint Predictor."""

from typing import List, Optional
import torch
import torch.nn as nn


class Spectral1DCNNEncoder(nn.Module):
    """
    1D Convolutional Neural Network backbone for raw and binned mass spectra.
    Extracts multi-scale peak cluster patterns and neutral loss signatures.
    """

    def __init__(
        self,
        input_dim: int = 14900,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv_net(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class NeuralFingerprintPredictor(nn.Module):
    """
    End-to-end model predicting molecular substructure fingerprint bits
    (e.g., Morgan / ECFP6 2048-dim) directly from tandem mass spectra.
    """

    def __init__(
        self,
        spectral_dim: int = 14900,
        fingerprint_dim: int = 2048,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [1024, 512]
        self.encoder = Spectral1DCNNEncoder(
            input_dim=spectral_dim, embedding_dim=hidden_dims[-1], dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, fingerprint_dim),
        )

    def forward(self, spectral_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs logits for each molecular fingerprint bit.
        """
        emb = self.encoder(spectral_tensor)
        return self.head(emb)
