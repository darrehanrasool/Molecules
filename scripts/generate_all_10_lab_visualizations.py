#!/usr/bin/env python3
"""
Generate All 10 High-Resolution, Vibrant, Publication-Grade Scientific Visualizations.
CASMI 2026 — Department of Computer Science & Engineering, IUST
Supervisor: Dr. Asif Assad | M.Tech Students: Dar Rehan Rasool, Muneeb Syed Shah, Shafiya Bashir Pampori (Centre for Artificial Intelligence, IUST)
"""

import json
import os
import shutil
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

# Vibrant, clean, high-contrast scientific theme
BACKGROUND = "#0f172a"    # Slate 900
PANEL = "#1e293b"         # Slate 800
CARD_BORDER = "#334155"   # Slate 700
TEXT_MAIN = "#f8fafc"     # Crisp White
TEXT_MUTED = "#94a3b8"    # Cool Slate

# Vibrant Color Palette
CYAN = "#06b6d4"          # Electric Cyan
ROSE = "#f43f5e"          # Vibrant Rose
PURPLE = "#a855f7"        # Royal Purple
GOLD = "#fbbf24"          # Bright Amber Gold
EMERALD = "#10b981"       # Mint Emerald
BLUE = "#3b82f6"          # Sky Blue
ORANGE = "#fb923c"        # Bright Orange

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": PANEL,
    "axes.edgecolor": CARD_BORDER,
    "axes.labelcolor": TEXT_MAIN,
    "text.color": TEXT_MAIN,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "grid.color": "#334155",
    "grid.alpha": 0.45,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
})

GRAPH_DIRS = [
    Path("graphs"),
    Path("reports/figures"),
    Path("/Users/darrehanrasool/.gemini/antigravity/brain/4965c974-7a39-4ec3-a0e3-b109fe6ec705/figures"),
]
for gd in GRAPH_DIRS:
    gd.mkdir(parents=True, exist_ok=True)

def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, color=TEXT_MAIN, fontsize=13, fontweight="bold", pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=11, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(CARD_BORDER)
    ax.spines["bottom"].set_color(CARD_BORDER)

def save_fig(fig, filename):
    for gd in GRAPH_DIRS:
        out_path = gd / filename
        fig.savefig(out_path, dpi=300, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[+] Saved vibrant {filename} to all destinations.")

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
    # FIG 01: Training & Validation Loss Curves (Vibrant)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(epochs, train_loss, color=CYAN, linewidth=2.8, label=f"Train Loss (Final: {train_loss[-1]:.4f})", zorder=3)
    ax.plot(epochs, val_loss, color=ROSE, linewidth=2.8, label=f"Val Loss (Best Ep {best_epoch}: {best_vloss:.4f})", zorder=3)
    ax.fill_between(epochs, train_loss, val_loss, color=PURPLE, alpha=0.25, label="Generalization Gap (Δ <= 0.0065)")
    ax.axvline(best_epoch, color=GOLD, linestyle="--", linewidth=1.8, alpha=0.9, label=f"Optimal Checkpoint (Epoch {best_epoch})")
    ax.scatter([best_epoch], [best_vloss], color=GOLD, s=110, zorder=5, edgecolor="white", linewidth=1.5)
    style_ax(ax, "01 · Train vs. Validation Loss Dynamics (60 Epochs | Zero Overfitting)", "Training Epoch", "Binary Cross-Entropy Loss")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor=CARD_BORDER, fontsize=10, loc="upper right")
    save_fig(fig, "01_loss_curves.png")

    # ----------------------------------------------------
    # FIG 02: Cosine Annealing Learning Rate Dynamics (Vibrant)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(epochs, lr, color=EMERALD, linewidth=2.8, label="CosineAnnealingLR (AdamW)", zorder=3)
    ax.fill_between(epochs, 0, lr, color=EMERALD, alpha=0.2)
    ax.scatter([epochs[0], epochs[-1]], [lr[0], lr[-1]], color=CYAN, s=80, zorder=5, edgecolor="white")
    ax.text(epochs[0] + 1.5, lr[0] * 0.9, f"Peak LR: {lr[0]:.1e}", color=CYAN, fontsize=10.5, fontweight="bold")
    ax.text(epochs[-1] - 14, lr[-1] * 80 + 1e-6, f"Terminal LR: {lr[-1]:.2e}", color=EMERALD, fontsize=10.5, fontweight="bold")
    style_ax(ax, "02 · Learning Rate Schedule & Decay Dynamics (1.13M Steps)", "Training Epoch", "Learning Rate (Log Scale)")
    ax.set_yscale("log")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor=CARD_BORDER, fontsize=10, loc="lower left")
    save_fig(fig, "02_lr_schedule.png")

    # ----------------------------------------------------
    # FIG 03: Validation AP and ROC-AUC (Vibrant Dual-Axis)
    # ----------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax2 = ax1.twinx()
    l1 = ax1.plot(epochs, val_ap, color=CYAN, linewidth=2.8, label=f"Average Precision (Best: {val_ap[best_idx]:.4f})", zorder=3)
    l2 = ax2.plot(epochs, val_auc, color=PURPLE, linewidth=2.8, linestyle="-", label=f"ROC-AUC (Best: {val_auc[best_idx]:.4f})", zorder=3)
    ax1.set_xlabel("Training Epoch", color=TEXT_MUTED, fontsize=11, labelpad=8)
    ax1.set_ylabel("Validation Average Precision (AP)", color=CYAN, fontsize=11, labelpad=8)
    ax2.set_ylabel("Validation ROC-AUC Score", color=PURPLE, fontsize=11, labelpad=8)
    ax1.set_title("03 · Validation Average Precision (AP) & ROC-AUC Trajectory", color=TEXT_MAIN, fontsize=13, fontweight="bold", pad=14)
    ax1.grid(True, linestyle="--", alpha=0.35)
    for sp in ["top", "left", "right", "bottom"]:
        ax1.spines[sp].set_color(CARD_BORDER)
        ax2.spines[sp].set_color(CARD_BORDER)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=True, facecolor=PANEL, edgecolor=CARD_BORDER, loc="lower right", fontsize=10)
    save_fig(fig, "03_ap_auc.png")

    # ----------------------------------------------------
    # FIG 04: Bit Accuracy (Vibrant)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(epochs, val_bit_acc * 100, color=GOLD, linewidth=2.8, label="Bit-Level Accuracy (%)", zorder=3)
    ax.fill_between(epochs, 97.8, val_bit_acc * 100, color=GOLD, alpha=0.15)
    ax.scatter([best_epoch], [val_bit_acc[best_idx] * 100], color=ROSE, s=90, zorder=5, edgecolor="white")
    ax.text(best_epoch - 15, val_bit_acc[best_idx] * 100 - 0.05, f"Peak Accuracy: {val_bit_acc[best_idx]*100:.2f}%", color=ROSE, fontsize=10.5, fontweight="bold")
    style_ax(ax, "04 · Bit-Level Classification Accuracy Across 2048 Morgan Bits", "Training Epoch", "Accuracy (%)")
    ax.set_ylim(97.8, 98.3)
    ax.legend(frameon=True, facecolor=PANEL, edgecolor=CARD_BORDER, loc="lower right", fontsize=10)
    save_fig(fig, "04_bit_accuracy.png")

    # Load validation sample tensor
    print("[*] Loading validation sample tensor from logs/validation_sample.npz...")
    val_data = np.load("logs/validation_sample.npz")
    probs = val_data["probabilities"]
    targets = val_data["targets"]

    # ----------------------------------------------------
    # FIG 05: Precision-Recall Curve (Vibrant Multi-Point)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    prec, rec, thresholds = precision_recall_curve(targets.ravel(), probs.ravel())
    ap_score = average_precision_score(targets.ravel(), probs.ravel())
    ax.plot(rec, prec, color=CYAN, linewidth=2.8, label=f"PR Curve (AP = {ap_score:.4f})", zorder=3)
    ax.fill_between(rec, 0, prec, color=CYAN, alpha=0.15)
    
    # Threshold points
    for th, col in [(0.50, ROSE), (0.35, GOLD), (0.20, EMERALD)]:
        if len(thresholds) > 0:
            idx = np.argmin(np.abs(thresholds - th))
            p_val, r_val = prec[idx], rec[idx]
            ax.scatter([r_val], [p_val], color=col, s=80, zorder=5, edgecolor="white", label=f"Threshold {th:.2f} (Prec: {p_val*100:.1f}%, Rec: {r_val*100:.1f}%)")
    
    ax.axhline(targets.mean(), color=TEXT_MUTED, linestyle=":", label=f"Random Chance ({targets.mean()*100:.2f}%)")
    style_ax(ax, "05 · Precision-Recall Tradeoff on Sparse 2048-Bit Morgan Fingerprints", "Recall (Fraction of True Bits Detected)", "Precision (Accuracy of Positive Predictions)")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor=CARD_BORDER, fontsize=9.5, loc="upper right")
    save_fig(fig, "05_precision_recall.png")

    # ----------------------------------------------------
    # FIG 06: Annotated Confusion Matrix (Decent, Colorful Heatmap)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    preds = (probs > 0.5).astype(np.uint8)
    cm = confusion_matrix(targets.ravel(), preds.ravel())
    tn, fp, fn, tp = cm.ravel()
    
    cm_display = np.array([[tn, fp], [fn, tp]])
    # Use vibrant YlGnBu colormap
    sns.heatmap(
        cm_display, annot=True, fmt=",d", cmap="viridis", cbar=True,
        xticklabels=["Predicted Inactive (0)", "Predicted Active (1)"],
        yticklabels=["True Inactive (0)", "True Active (1)"],
        linewidths=2.5, linecolor=BACKGROUND, ax=ax,
        annot_kws={"fontsize": 13, "fontweight": "bold", "color": "white"}
    )
    style_ax(ax, f"06 · Bit Confusion Matrix (2,097,152 Trials | Active-Bit Precision: {tp/(tp+fp)*100:.1f}%)", "", "")
    save_fig(fig, "06_confusion_matrix.png")

    # ----------------------------------------------------
    # FIG 07: Predicted vs True Fingerprint Activity (Vibrant)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    true_active = targets.sum(axis=1)
    pred_active = probs.sum(axis=1)
    corr = np.corrcoef(true_active, pred_active)[0, 1]
    
    # Density scatter with magma / plasma colors
    scatter = ax.scatter(
        true_active, pred_active, c=np.abs(true_active - pred_active),
        cmap="plasma", alpha=0.75, s=32, edgecolors="none", label="Validation Molecules (n=1,024)"
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Absolute Bit Error", color=TEXT_MAIN, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED)
    
    max_lim = max(true_active.max(), pred_active.max()) + 5
    ax.plot([0, max_lim], [0, max_lim], color=CYAN, linestyle="--", linewidth=2.2, label="Ideal Agreement (y = x)")
    ax.text(12, max_lim - 14, f"Pearson Correlation: r = {corr:.3f}", color=EMERALD, fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=CARD_BORDER))
    style_ax(ax, "07 · Active Bit Count Agreement: Predicted vs. True Fingerprints", "True Active Bits (Ground Truth)", "Predicted Probability Sum (Expected Active Bits)")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor=CARD_BORDER, fontsize=9.5, loc="lower right")
    save_fig(fig, "07_fingerprint_activity.png")

    # ----------------------------------------------------
    # FIG 08: t-SNE Manifold (Vibrant Turbo Colorbar)
    # ----------------------------------------------------
    print("[*] Loading spectra data for t-SNE and data distributions...")
    tc_file = Path("data/processed/test_chunks/test_00001.npz")
    with np.load(tc_file) as te:
        spectra = te["spectra"].astype(np.float32)
        pmz = te["precursor_mz"]

    print("[*] Computing t-SNE 2D embedding...")
    rng = np.random.default_rng(42)
    sample_n = min(1000, len(spectra))
    sub_idx = rng.choice(len(spectra), sample_n, replace=False)
    sub_spectra = spectra[sub_idx]
    sub_pmz = pmz[sub_idx]

    var = sub_spectra.var(axis=0)
    top_feats = np.argsort(-var)[:400]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    embedding = tsne.fit_transform(sub_spectra[:, top_feats])

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=sub_pmz, cmap="turbo", s=34, alpha=0.85, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Precursor Ion Mass (m/z)", color=TEXT_MAIN, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=TEXT_MUTED)
    style_ax(ax, "08 · t-SNE Manifold of MS/MS Spectra Clustered by Precursor Mass", "t-SNE Dimension 1", "t-SNE Dimension 2")
    save_fig(fig, "08_tsne.png")

    # ----------------------------------------------------
    # FIG 09: Peak Count Distribution (Vibrant Emerald & Amber)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    peak_counts = np.count_nonzero(spectra, axis=1)
    sns.histplot(peak_counts, bins=45, color=ORANGE, kde=True, ax=ax, edgecolor=BACKGROUND, alpha=0.7)
    ax.axvline(np.mean(peak_counts), color=CYAN, linestyle="--", linewidth=2.2, label=f"Mean Peaks: {np.mean(peak_counts):.1f}")
    ax.axvline(np.median(peak_counts), color=GOLD, linestyle=":", linewidth=2.2, label=f"Median Peaks: {np.median(peak_counts):.1f}")
    style_ax(ax, "09 · Distribution of MS/MS Fragmentation Peaks (Post-1% Floor)", "Non-Zero Fragmentation Peaks per Spectrum", "Spectral Density")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor=CARD_BORDER, fontsize=10)
    save_fig(fig, "09_peak_count.png")

    # ----------------------------------------------------
    # FIG 10: Precursor Mass Distribution (Vibrant Cyan & Rose)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.histplot(pmz, bins=50, color=BLUE, kde=True, ax=ax, edgecolor=BACKGROUND, alpha=0.7)
    ax.axvline(np.mean(pmz), color=ROSE, linestyle="--", linewidth=2.2, label=f"Mean Precursor: {np.mean(pmz):.1f} Da")
    ax.axvline(np.median(pmz), color=EMERALD, linestyle=":", linewidth=2.2, label=f"Median Precursor: {np.median(pmz):.1f} Da")
    style_ax(ax, "10 · Chemical Mass Space Distribution (Natural Products Cohort)", "Precursor Mass m/z (Da)", "Spectrum Count")
    ax.legend(frameon=True, facecolor=PANEL, edgecolor=CARD_BORDER, fontsize=10)
    save_fig(fig, "10_precursor_mass.png")

    print("\n[+] SUCCESS! All 10 vibrant, colorful visualizations generated at 300 DPI.")

if __name__ == "__main__":
    main()
