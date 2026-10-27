"""Feature engineering for mass spectra: dense binning and spectral vectors."""

from typing import Tuple
import numpy as np


def spectrum_to_dense_bins(
    peaks: np.ndarray,
    min_mz: float = 10.0,
    max_mz: float = 1500.0,
    bin_size: float = 0.1,
) -> np.ndarray:
    """
    Transforms sparse (m/z, intensity) peak array into a dense 1D fixed-dimension vector.

    Args:
        peaks: (N, 2) array of [m/z, intensity]
        min_mz: Lower bound of m/z range
        max_mz: Upper bound of m/z range
        bin_size: Width of each m/z bin in Daltons

    Returns:
        1D float32 numpy array of length floor((max_mz - min_mz) / bin_size)
    """
    num_bins = int(np.floor((max_mz - min_mz) / bin_size))
    dense_vector = np.zeros(num_bins, dtype=np.float32)

    if peaks is None or len(peaks) == 0:
        return dense_vector

    mz = peaks[:, 0]
    intensities = peaks[:, 1]

    # Mask valid bounds
    mask = (mz >= min_mz) & (mz < max_mz)
    if not np.any(mask):
        return dense_vector

    valid_mz = mz[mask]
    valid_int = intensities[mask]

    bin_indices = np.floor((valid_mz - min_mz) / bin_size).astype(int)
    # Clamp to boundaries
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Accumulate intensities into bins
    np.maximum.at(dense_vector, bin_indices, valid_int)
    return dense_vector


def compute_spectral_binned_pair(
    peaks: np.ndarray,
    precursor_mz: float,
    min_mz: float = 10.0,
    max_mz: float = 1000.0,
    bin_size: float = 0.5,
) -> np.ndarray:
    """
    Concatenates direct fragment bins and neutral loss bins into a joint feature vector.
    """
    frag_bins = spectrum_to_dense_bins(peaks, min_mz=min_mz, max_mz=max_mz, bin_size=bin_size)

    # Compute neutral losses
    from molecules.data.preprocessor import compute_neutral_losses
    loss_peaks = compute_neutral_losses(peaks, precursor_mz)
    loss_bins = spectrum_to_dense_bins(loss_peaks, min_mz=min_mz, max_mz=max_mz, bin_size=bin_size)

    return np.concatenate([frag_bins, loss_bins], axis=0)
