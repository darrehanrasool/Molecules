"""Spectral Preprocessor: Noise filtering, normalization, and peak curation."""

from typing import Optional, Tuple
import numpy as np


class SpectralPreprocessor:
    """
    Cleans and standardizes experimental tandem mass spectra.
    """

    def __init__(
        self,
        min_intensity_rel: float = 0.005,
        max_peaks: int = 300,
        normalize: bool = True,
        norm_method: str = "base_peak",
        remove_precursor_peak: bool = False,
        precursor_tolerance_da: float = 0.05,
    ):
        self.min_intensity_rel = min_intensity_rel
        self.max_peaks = max_peaks
        self.normalize = normalize
        self.norm_method = norm_method
        self.remove_precursor_peak = remove_precursor_peak
        self.precursor_tolerance_da = precursor_tolerance_da

    def process(
        self, peaks: np.ndarray, precursor_mz: Optional[float] = None
    ) -> np.ndarray:
        """
        Cleans and normalizes an input peak array (N, 2) -> (M, 2).
        """
        if peaks is None or len(peaks) == 0:
            return np.empty((0, 2), dtype=np.float32)

        p = peaks.copy()

        # Remove negative or NaN intensities
        valid_mask = np.isfinite(p[:, 0]) & np.isfinite(p[:, 1]) & (p[:, 1] > 0)
        p = p[valid_mask]
        if len(p) == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Base peak normalization
        base_peak_intensity = np.max(p[:, 1])
        if base_peak_intensity > 0:
            rel_intensity = p[:, 1] / base_peak_intensity
            # Filter low intensity noise
            keep_mask = rel_intensity >= self.min_intensity_rel
            p = p[keep_mask]
            if len(p) == 0:
                return np.empty((0, 2), dtype=np.float32)

        # Optionally remove unfragmented precursor peak
        if self.remove_precursor_peak and precursor_mz is not None:
            non_precursor_mask = np.abs(p[:, 0] - precursor_mz) > self.precursor_tolerance_da
            p = p[non_precursor_mask]
            if len(p) == 0:
                return np.empty((0, 2), dtype=np.float32)

        # Cap at max_peaks (keep top by intensity)
        if len(p) > self.max_peaks:
            top_indices = np.argsort(p[:, 1])[-self.max_peaks:]
            p = p[top_indices]

        # Final normalization
        if self.normalize:
            if self.norm_method == "base_peak":
                max_int = np.max(p[:, 1])
                if max_int > 0:
                    p[:, 1] = p[:, 1] / max_int
            elif self.norm_method == "tic":
                total_int = np.sum(p[:, 1])
                if total_int > 0:
                    p[:, 1] = p[:, 1] / total_int

        # Re-sort by m/z ascending
        p = p[p[:, 0].argsort()]
        return p.astype(np.float32)


def compute_neutral_losses(peaks: np.ndarray, precursor_mz: float) -> np.ndarray:
    """
    Computes neutral loss spectrum where loss_mz = precursor_mz - fragment_mz.
    """
    if len(peaks) == 0:
        return np.empty((0, 2), dtype=np.float32)

    loss_mz = precursor_mz - peaks[:, 0]
    valid_mask = loss_mz >= 0
    losses = np.column_stack([loss_mz[valid_mask], peaks[valid_mask, 1]])
    return losses[losses[:, 0].argsort()].astype(np.float32)
