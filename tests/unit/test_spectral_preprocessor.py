"""Unit tests for spectral preprocessing and noise filtering."""

import numpy as np
from molecules.data.preprocessor import SpectralPreprocessor, compute_neutral_losses
from molecules.data.loader import parse_peak_string


def test_parse_peak_string():
    raw_str = "105.03:0.42 153.01:1.00 254.05:0.85"
    peaks = parse_peak_string(raw_str)
    assert peaks.shape == (3, 2)
    assert peaks[0, 0] == 105.03
    assert peaks[1, 1] == 1.00


def test_spectral_preprocessor_base_peak_normalization():
    peaks = np.array([
        [100.0, 10.0],
        [200.0, 50.0], # Base peak
        [300.0, 0.05], # Low intensity noise (0.1% of base peak)
    ], dtype=np.float32)

    preprocessor = SpectralPreprocessor(min_intensity_rel=0.005, normalize=True)
    cleaned = preprocessor.process(peaks)

    # Low noise peak at 300.0 should be filtered out
    assert len(cleaned) == 2
    # Base peak should be normalized to 1.0
    assert cleaned[1, 0] == 200.0
    assert cleaned[1, 1] == 1.0
    assert cleaned[0, 1] == 0.2


def test_neutral_loss_computation():
    peaks = np.array([
        [100.0, 0.5],
        [150.0, 1.0],
    ], dtype=np.float32)
    precursor_mz = 200.0

    losses = compute_neutral_losses(peaks, precursor_mz)
    assert len(losses) == 2
    # 200 - 150 = 50, 200 - 100 = 100
    assert losses[0, 0] == 50.0
    assert losses[1, 0] == 100.0
