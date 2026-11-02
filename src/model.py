"""Spectrum → Morgan fingerprint predictor."""
import torch, torch.nn as nn
from src.config import CFG

class SpectrumFPNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, CFG.HIDDEN_1),
            nn.BatchNorm1d(CFG.HIDDEN_1),
            nn.GELU(),
            nn.Dropout(CFG.DROPOUT),
            nn.Linear(CFG.HIDDEN_1, CFG.HIDDEN_2),
            nn.BatchNorm1d(CFG.HIDDEN_2),
            nn.GELU(),
            nn.Dropout(CFG.DROPOUT),
            nn.Linear(CFG.HIDDEN_2, out_dim),
        )
    def forward(self, x): return self.net(x)