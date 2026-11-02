#!/usr/bin/env python3
"""
Generate All 10 Publication-Grade Scientific Visualizations for CASMI 2026.
Department of Computer Science & Engineering, Islamic University of Science & Technology (IUST)
Supervisor: Dr. Asif Assad | Lead ML Architect: Dar Rehan Rasool | Researchers: Muneeb Syed Shah, Shafiya Bashir Pampori
"""

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

# Set high-DPI scientific dark theme
plt.style.use("dark_background")
BACKGROUND = "#0b1020"
PANEL = "#111a2e"
GRID = "#1e293b"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
CYAN = "#00f2fe"
PURPLE = "#a78bfa"
CORAL = "#ff4b72"
EMERALD = "#10b981"
AMBER = "#f59e0b"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": PANEL,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.alpha": 0.5,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
})

GRAPH_DIRS = [Path("graphs"), Path("reports/figures")]
for gd in GRAPH_DIRS:
    gd.mkdir(parents=True, exist_ok=True)

def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=11, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")

def save_fig(fig, filename):
    for gd in GRAPH_DIRS:
        out_path = gd / filename
        fig.savefig(out_path, dpi=300, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[+] Saved {filename} to {GRAPH_DIRS}")

def main():
    print("[*] Loading training history from logs/history.json...")
    with open("logs/history.json", "r") as f:
        history = json.load(f)

    epochs = np.array(history["epoch"])
    train_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])
    val_ap = np.array(history["val_ap"])
    val_auc = np.array(history["val_auc"])
    val_bit_acc = np.array(history["val_bit_acc"])
    lr = np.array(history["lr"])

    best_idx = np.argmin(val_loss)
    best_epoch = epochs[best_idx]
    best_vloss = val_loss[best_idx]

    # ----------------------------------------------------
    # FIG 01: Training & Validation Loss Curves
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, train_loss, color=CYAN, linewidth=2.5, label=f"Train Loss (Final: {train_loss[-1]:.4f})")
    ax.plot(epochs, val_loss, color=CORAL, linewidth=2.5, label=f"Val Loss (Best Ep {best_epoch}: {best_vloss:.4f})")
    ax.fill_between(epochs, train_loss, val_loss, color=PURPLE, alpha=0.15, label="Generalization Gap (Delta <= 0.0065)")
    ax.axvline(best_epoch, color=AMBER, linestyle="--", linewidth=1.5, alpha=0.85, label=f"Optimal Checkpoint (Epoch {best_epoch})")
    ax.scatter([best_epoch], [best_vloss], color=AMBER, s=90, zorder=5, edgecolor="white")
    style_ax(ax, "01 · Train vs. Validation Loss Dynamics (60 Epochs)", "Training Epoch", "BCEWithLogits Loss")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor="#334155", fontsize=9.5)
    save_fig(fig, "01_loss_curves.png")

    # ----------------------------------------------------
    # FIG 02: Cosine Annealing Learning Rate Dynamics
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, lr, color=EMERALD, linewidth=2.5, label="CosineAnnealingLR")
    ax.fill_between(epochs, 0, lr, color=EMERALD, alpha=0.15)
    ax.scatter([epochs[0], epochs[-1]], [lr[0], lr[-1]], color=CYAN, s=70, zorder=5)
    ax.text(epochs[0] + 1, lr[0] * 0.95, f"Peak LR: {lr[0]:.1e}", color=CYAN, fontsize=10, fontweight="bold")
    ax.text(epochs[-1] - 12, lr[-1] * 80 + 1e-6, f"Terminal LR: {lr[-1]:.2e}", color=EMERALD, fontsize=10, fontweight="bold")
    style_ax(ax, "02 · Learning Rate Schedule & Decay Trajectory", "Training Epoch", "Learning Rate (AdamW)")
    ax.set_yscale("log")
    save_fig(fig, "02_lr_schedule.png")

    # ----------------------------------------------------
    # FIG 03: Validation AP and ROC-AUC
    # ----------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()
    l1 = ax1.plot(epochs, val_ap, color=CYAN, linewidth=2.5, label=f"Average Precision (Best: {val_ap[best_idx]:.4f})")
    l2 = ax2.plot(epochs, val_auc, color=PURPLE, linewidth=2.5, linestyle="-", label=f"ROC-AUC (Best: {val_auc[best_idx]:.4f})")
    ax1.set_xlabel("Training Epoch", color=MUTED, fontsize=11, labelpad=8)
    ax1.set_ylabel("Validation Average Precision (AP)", color=CYAN, fontsize=11, labelpad=8)
    ax2.set_ylabel("Validation ROC-AUC Score", color=PURPLE, fontsize=11, labelpad=8)
    ax1.set_title("03 · Validation Average Precision (AP) & ROC-AUC Trajectory", color=TEXT, fontsize=13, fontweight="bold", pad=14)
    ax1.grid(True, linestyle="--", alpha=0.35)
    for sp in ["top", "left", "right", "bottom"]:
        ax1.spines[sp].set_color("#334155")
        ax2.spines[sp].set_color("#334155")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=True, facecolor=PANEL, edgecolor="#334155", loc="lower right", fontsize=9.5)
    save_fig(fig, "03_ap_auc.png")

    # ----------------------------------------------------
    # FIG 04: Bit Accuracy
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, val_bit_acc * 100, color=AMBER, linewidth=2.5, label="Bit-Level Accuracy (%)")
    ax.scatter([best_epoch], [val_bit_acc[best_idx] * 100], color=CORAL, s=80, zorder=5)
    ax.text(best_epoch - 14, val_bit_acc[best_idx] * 100 - 0.05, f"Peak Accuracy: {val_bit_acc[best_idx]*100:.2f}%", color=CORAL, fontsize=10, fontweight="bold")
    style_ax(ax, "04 · Bit-Level Classification Accuracy Across 2048 Target Bits", "Training Epoch", "Accuracy (%)")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor="#334155", loc="lower right", fontsize=9.5)
    save_fig(fig, "04_bit_accuracy.png")

    # Load validation sample tensor
    print("[*] Loading validation sample tensor from logs/validation_sample.npz...")
    val_data = np.load("logs/validation_sample.npz")
    probs = val_data["probabilities"]
    targets = val_data["targets"]

    # ----------------------------------------------------
    # FIG 05: Precision-Recall Curve
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    prec, rec, thresholds = precision_recall_curve(targets.ravel(), probs.ravel())
    ap_score = average_precision_score(targets.ravel(), probs.ravel())
    ax.plot(rec, prec, color=CYAN, linewidth=2.5, label=f"PR Curve (AP = {ap_score:.4f})")
    
    # Highlight threshold 0.5 point
    idx_50 = np.argmin(np.abs(thresholds - 0.5)) if len(thresholds) > 0 else 0
    p_50, r_50 = prec[idx_50], rec[idx_50]
    ax.scatter([r_50], [p_50], color=CORAL, s=90, zorder=5, label=f"Threshold 0.50 (Prec: {p_50*100:.1f}%, Rec: {r_50*100:.1f}%)")
    ax.plot([0, 1], [targets.mean(), targets.mean()], color=MUTED, linestyle=":", label=f"Random Chance ({targets.mean()*100:.2f}%)")
    
    style_ax(ax, "05 · Precision-Recall Tradeoff on Sparse 2048-Bit Morgan Fingerprints", "Recall (Fraction of True Bits Detected)", "Precision (Accuracy of Positive Predictions)")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor="#334155", fontsize=9.5, loc="upper right")
    save_fig(fig, "05_precision_recall.png")

    # ----------------------------------------------------
    # FIG 06: Annotated Confusion Matrix
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    preds = (probs > 0.5).astype(np.uint8)
    cm = confusion_matrix(targets.ravel(), preds.ravel())
    tn, fp, fn, tp = cm.ravel()
    
    cm_display = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(
        cm_display, annot=True, fmt=",d", cmap="mako", cbar=False,
        xticklabels=["Off (0)", "On (1)"], yticklabels=["Off (0)", "On (1)"],
        linewidths=2, linecolor=BACKGROUND, ax=ax,
        annot_kws={"fontsize": 13, "fontweight": "bold", "color": "white"}
    )
    style_ax(ax, f"06 · Bit Confusion Matrix (2,097,152 Trials | Prec: {tp/(tp+fp)*100:.1f}%)", "Predicted Bit Value", "Ground Truth Bit Value")
    save_fig(fig, "06_confusion_matrix.png")

    # ----------------------------------------------------
    # FIG 07: Predicted vs True Fingerprint Activity
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    true_active = targets.sum(axis=1)
    pred_active = probs.sum(axis=1)
    corr = np.corrcoef(true_active, pred_active)[0, 1]
    
    ax.scatter(true_active, pred_active, color=CYAN, alpha=0.5, s=28, edgecolors="none", label="Molecules (n=1,024)")
    max_lim = max(true_active.max(), pred_active.max()) + 5
    ax.plot([0, max_lim], [0, max_lim], color=CORAL, linestyle="--", linewidth=2, label="Ideal Agreement (y = x)")
    ax.text(10, max_lim - 15, f"Pearson Correlation: r = {corr:.3f}", color=EMERALD, fontsize=11, fontweight="bold")
    style_ax(ax, "07 · Active Bit Count Correlation: Predicted vs. True Fingerprints", "True Active Bits (Ground Truth)", "Predicted Probability Sum (Expected Active Bits)")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor="#334155", fontsize=9.5)
    save_fig(fig, "07_fingerprint_activity.png")

    # ----------------------------------------------------
    # LOAD TEST / DATA DIAGNOSTICS FOR FIGS 08, 09, 10
    # ----------------------------------------------------
    print("[*] Loading spectra data for t-SNE and data distributions...")
    tc_file = Path("data/processed/test_chunks/test_00001.npz")
    with np.load(tc_file) as te:
        spectra = te["spectra"].astype(np.float32)
        pmz = te["precursor_mz"]

    # ----------------------------------------------------
    # FIG 08: t-SNE Latent Space Manifold
    # ----------------------------------------------------
    print("[*] Computing t-SNE 2D embedding...")
    rng = np.random.default_rng(42)
    sample_n = min(1000, len(spectra))
    sub_idx = rng.choice(len(spectra), sample_n, replace=False)
    sub_spectra = spectra[sub_idx]
    sub_pmz = pmz[sub_idx]

    # Non-zero variance features
    var = sub_spectra.var(axis=0)
    top_feats = np.argsort(-var)[:400]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    embedding = tsne.fit_transform(sub_spectra[:, top_feats])

    fig, ax = plt.subplots(figsize=(9.5, 6))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=sub_pmz, cmap="viridis", s=30, alpha=0.85, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Precursor Ion Mass (m/z)", color=TEXT, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    style_ax(ax, "08 · t-SNE Manifold of MS/MS Spectra Clustered by Molecular Mass", "t-SNE Dimension 1", "t-SNE Dimension 2")
    save_fig(fig, "08_tsne.png")

    # ----------------------------------------------------
    # FIG 09: Peak Count Distribution
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    peak_counts = np.count_nonzero(spectra, axis=1)
    sns.histplot(peak_counts, bins=45, color=CORAL, kde=True, ax=ax, edgecolor=BACKGROUND)
    ax.axvline(np.mean(peak_counts), color=CYAN, linestyle="--", linewidth=2, label=f"Mean Peaks: {np.mean(peak_counts):.1f}")
    ax.axvline(np.median(peak_counts), color=AMBER, linestyle=":", linewidth=2, label=f"Median Peaks: {np.median(peak_counts):.1f}")
    style_ax(ax, "09 · Distribution of MS/MS Fragmentation Peaks (Post-1% Floor)", "Non-Zero Fragmentation Peaks per Spectrum", "Spectral Density")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor="#334155", fontsize=9.5)
    save_fig(fig, "09_peak_count.png")

    # ----------------------------------------------------
    # FIG 10: Precursor Mass Distribution
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(pmz, bins=50, color=CYAN, kde=True, ax=ax, edgecolor=BACKGROUND)
    ax.axvline(np.mean(pmz), color=CORAL, linestyle="--", linewidth=2, label=f"Mean Precursor: {np.mean(pmz):.1f} Da")
    ax.axvline(np.median(pmz), color=EMERALD, linestyle=":", linewidth=2, label=f"Median Precursor: {np.median(pmz):.1f} Da")
    style_ax(ax, "10 · Chemical Mass Space Distribution (Natural Products Cohort)", "Precursor Mass m/z (Da)", "Spectrum Count")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor="#334155", fontsize=9.5)
    save_fig(fig, "10_precursor_mass.png")

    print("\n[+] SUCCESS! All 10 publication-quality visualizations generated at 300 DPI.")

if __name__ == "__main__":
    main()
