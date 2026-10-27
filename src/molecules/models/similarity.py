"""Mass spectral similarity functions: Cosine, Modified Cosine, and Entropy."""

from typing import Tuple
import numpy as np


def spectral_cosine_similarity(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    mz_tolerance_da: float = 0.05,
    intensity_power: float = 0.5,
    mz_power: float = 1.0,
) -> float:
    """
    Computes standard peak-matched cosine similarity between two spectra.
    """
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return 0.0

    # Calculate weighted intensities
    w_a = (peaks_a[:, 1] ** intensity_power) * (peaks_a[:, 0] ** mz_power)
    w_b = (peaks_b[:, 1] ** intensity_power) * (peaks_b[:, 0] ** mz_power)

    norm_a = np.sqrt(np.sum(w_a ** 2))
    norm_b = np.sqrt(np.sum(w_b ** 2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Match peaks within mz_tolerance_da
    dot_product = 0.0
    j_start = 0
    len_b = len(peaks_b)

    for i in range(len(peaks_a)):
        mz_a = peaks_a[i, 0]
        while j_start < len_b and peaks_b[j_start, 0] < mz_a - mz_tolerance_da:
            j_start += 1

        best_diff = mz_tolerance_da
        best_match_idx = -1
        for j in range(j_start, len_b):
            mz_b = peaks_b[j, 0]
            diff = abs(mz_a - mz_b)
            if diff <= mz_tolerance_da and diff < best_diff:
                best_diff = diff
                best_match_idx = j
            elif mz_b > mz_a + mz_tolerance_da:
                break

        if best_match_idx >= 0:
            dot_product += w_a[i] * w_b[best_match_idx]

    return float(dot_product / (norm_a * norm_b))


def modified_cosine_similarity(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    precursor_mz_a: float,
    precursor_mz_b: float,
    mz_tolerance_da: float = 0.05,
) -> float:
    """
    Computes modified cosine similarity incorporating both direct fragment matches
    and precursor mass-shift matches (accounting for delta mass shifts).
    """
    delta_mass = precursor_mz_a - precursor_mz_b
    direct_score = spectral_cosine_similarity(peaks_a, peaks_b, mz_tolerance_da)

    if abs(delta_mass) < mz_tolerance_da:
        return direct_score

    # Shift spectrum B fragments by delta_mass
    shifted_peaks_b = peaks_b.copy()
    shifted_peaks_b[:, 0] += delta_mass
    valid_mask = shifted_peaks_b[:, 0] > 0
    shifted_peaks_b = shifted_peaks_b[valid_mask]

    shifted_score = spectral_cosine_similarity(peaks_a, shifted_peaks_b, mz_tolerance_da)
    return float(max(direct_score, shifted_score))
