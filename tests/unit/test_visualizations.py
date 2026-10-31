"""Unit tests for the scientific visualization module."""

import os
import tempfile
from pathlib import Path
import numpy as np
import pytest

from molecules.visualizations.plots import (
    integrate_trapz,
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


def test_integrate_trapz():
    x = np.linspace(0, 1, 100)
    y = np.ones_like(x)
    area = integrate_trapz(y, x)
    assert pytest.approx(area, 1e-3) == 1.0


def test_plot_renderers_create_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # 1. Learning dynamics
        epochs = np.arange(1, 10)
        t_loss = np.linspace(0.5, 0.1, 9)
        v_loss = np.linspace(0.6, 0.15, 9)
        t_mrr = np.linspace(0.2, 0.5, 9)
        v_mrr = np.linspace(0.18, 0.48, 9)
        test_mrr = np.linspace(0.17, 0.47, 9)
        out1 = tmp / "test_dyn.png"
        plot_learning_dynamics(epochs, t_loss, v_loss, t_mrr, v_mrr, test_mrr, out1)
        assert out1.exists() and out1.stat().st_size > 0

        # 2. Top-K accuracy
        out2 = tmp / "test_topk.png"
        plot_top_k_accuracy_schedule([1, 5, 25], [0.4, 0.6, 0.8], [0.38, 0.58, 0.78], [0.37, 0.57, 0.77], out2)
        assert out2.exists() and out2.stat().st_size > 0

        # 3. Precision recall
        out3 = tmp / "test_pr.png"
        r = np.linspace(0, 1, 10)
        p = np.linspace(1, 0.5, 10)
        plot_precision_recall_curves(r, {"Top-1": p}, out3)
        assert out3.exists() and out3.stat().st_size > 0

        # 4. ROC F1
        out4 = tmp / "test_roc.png"
        plot_roc_f1_tradeoff(r, r, 0.85, r, r, out4)
        assert out4.exists() and out4.stat().st_size > 0

        # 5. Mirror plot
        out5 = tmp / "test_mirror.png"
        pks = np.array([[100.0, 1.0], [200.0, 0.5]])
        plot_spectral_mirror(pks, pks, 201.0, "CC", "CC", out5)
        assert out5.exists() and out5.stat().st_size > 0

        # 6. Neutral loss
        out6 = tmp / "test_nl.png"
        nl_data = {"H2O": (18.01, 100, "-H2O")}
        plot_neutral_loss_distribution(nl_data, out6)
        assert out6.exists() and out6.stat().st_size > 0

        # 7. PPM error
        out7 = tmp / "test_ppm.png"
        plot_ppm_error_distribution(np.zeros(10), np.zeros(10), 10.0, out7)
        assert out7.exists() and out7.stat().st_size > 0

        # 8. Heatmap
        out8 = tmp / "test_hm.png"
        plot_fingerprint_bit_attribution(np.zeros((2, 2)), ["A", "B"], ["M1", "M2"], out8)
        assert out8.exists() and out8.stat().st_size > 0

        # 9. Histogram
        out9 = tmp / "test_hist.png"
        plot_reciprocal_rank_histogram([1, 2, 3], 25, out9)
        assert out9.exists() and out9.stat().st_size > 0

        # 10. Model zoo
        out10 = tmp / "test_zoo.png"
        plot_model_zoo_comparison(["M1"], [0.5], [0.4], [0.6], [0.8], out10)
        assert out10.exists() and out10.stat().st_size > 0
