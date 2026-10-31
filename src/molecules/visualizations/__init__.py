"""Visualizations package: Advanced scientific plots for mass spectrometry & ML."""

from molecules.visualizations.plots import (
    plot_learning_dynamics,
    plot_top_k_accuracy_schedule,
    plot_precision_recall_curves,
    plot_roc_f1_tradeoff,
    plot_spectral_mirror,
    plot_neutral_loss_distribution,
    plot_ppm_error_distribution,
    plot_fingerprint_bit_attribution,
    plot_reciprocal_rank_histogram,
    plot_model_zoo_comparison,
)

__all__ = [
    "plot_learning_dynamics",
    "plot_top_k_accuracy_schedule",
    "plot_precision_recall_curves",
    "plot_roc_f1_tradeoff",
    "plot_spectral_mirror",
    "plot_neutral_loss_distribution",
    "plot_ppm_error_distribution",
    "plot_fingerprint_bit_attribution",
    "plot_reciprocal_rank_histogram",
    "plot_model_zoo_comparison",
]
