"""Generate live training plots and data diagnostics in graphs/."""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.config import CFG
from src.model import SpectrumFPNet
from src.utils import get_logger

log = get_logger("viz")

BACKGROUND = "#0b1020"
PANEL = "#111a2e"
TEXT = "#e5e7eb"
MUTED = "#9ca3af"
PALETTE = ["#22d3ee", "#a78bfa", "#fb7185", "#fbbf24", "#34d399"]
sns.set_theme(style="darkgrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": PANEL,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": "#334155",
    "grid.alpha": 0.45,
})


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, color=TEXT, fontweight="bold", pad=16)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")


def _save(fig, name):
    path = CFG.GRAPH_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(temp_path, format="png")
        temp_path.replace(path)
    finally:
        plt.close(fig)
    log.info(f"saved {path}")


def _history(history=None):
    if history is None:
        history_path = CFG.LOG_DIR / "history.json"
        log_path = CFG.LOG_DIR / "train.log"
        if log_path.is_file():
            with log_path.open(encoding="utf-8") as file:
                lines = file.readlines()
            session_starts = [
                index for index, line in enumerate(lines)
                if "INFO | Device:" in line
            ]
            session_start = session_starts[-1] if session_starts else 0
            pattern = re.compile(
                r"Epoch\s+(\d+)\s+\|\s+tr\s+([\d.eE+-]+)\s+\|\s+"
                r"va\s+([\d.eE+-]+)\s+\|\s+AP\s+([\d.eE+-]+)\s+\|\s+"
                r"AUC\s+([\d.eE+-]+)\s+\|\s+bitacc\s+([\d.eE+-]+)\s+\|\s+"
                r"lr\s+([\d.eE+-]+)"
            )
            parsed = {
                "epoch": [], "train_loss": [], "val_loss": [],
                "val_ap": [], "val_auc": [], "val_bit_acc": [], "lr": [],
            }
            for line in lines[session_start:]:
                match = pattern.search(line)
                if match:
                    epoch, *values = match.groups()
                    parsed["epoch"].append(int(epoch))
                    for key, value in zip(
                        ("train_loss", "val_loss", "val_ap", "val_auc",
                         "val_bit_acc", "lr"),
                        values,
                    ):
                        parsed[key].append(float(value))
            if parsed["epoch"]:
                return parsed
            if session_starts:
                raise RuntimeError(
                    "The current training run has not completed an epoch yet. "
                    "Wait for its first validation pass before generating "
                    "training charts."
                )

        if not history_path.is_file():
            raise FileNotFoundError(
                f"No completed training epochs found in {log_path} or "
                f"{history_path}. Wait for the current epoch to finish."
            )
        with history_path.open(encoding="utf-8") as file:
            history = json.load(file)
    if not history.get("epoch"):
        raise ValueError("Training history contains no completed epochs.")
    return history


def _validation_sample(path=None):
    sample_path = path or CFG.LOG_DIR / "validation_sample.npz"
    checkpoint_path = CFG.MODEL_DIR / "best_model.pt"
    if sample_path.is_file():
        with np.load(sample_path) as data:
            if "checkpoint_mtime" not in data:
                return data["probabilities"], data["targets"]
            saved_mtime = float(data["checkpoint_mtime"])
            if (
                not checkpoint_path.is_file()
                or checkpoint_path.stat().st_mtime <= saved_mtime
            ):
                return data["probabilities"], data["targets"]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Validation predictions and model checkpoint are unavailable. "
            f"Complete at least one epoch to create {checkpoint_path}."
        )

    import torch
    from src.dataset import get_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SpectrumFPNet(checkpoint["in_dim"], CFG.FP_BITS).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    _, validation_loader = get_dataloaders()
    probabilities, targets = [], []
    row_count = 0
    with torch.no_grad():
        for xb, yb in validation_loader:
            take = min(1024 - row_count, len(xb))
            probabilities.append(
                torch.sigmoid(model(xb[:take].to(device))).cpu().numpy()
            )
            targets.append(yb[:take].numpy())
            row_count += take
            if row_count >= 1024:
                break

    if not row_count:
        raise RuntimeError("Validation loader produced no rows for plotting.")
    probabilities = np.concatenate(probabilities).astype(np.float32)
    targets = np.concatenate(targets).astype(np.uint8)
    temp_path = sample_path.with_name(
        f".{sample_path.name}.{os.getpid()}.tmp"
    )
    with temp_path.open("wb") as file:
        np.savez_compressed(
            file,
            probabilities=probabilities,
            targets=targets,
            checkpoint_mtime=checkpoint_path.stat().st_mtime,
        )
    temp_path.replace(sample_path)
    log.info(f"Saved validation plot sample to {sample_path}")
    return probabilities, targets


def generate_training_graphs(history=None, validation_path=None):
    """Refresh the seven training charts after each completed epoch."""
    history = _history(history)
    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], color=PALETTE[0], marker="o",
            markersize=4, linewidth=2.5, label="Training")
    ax.plot(epochs, history["val_loss"], color=PALETTE[2], marker="o",
            markersize=4, linewidth=2.5, label="Validation")
    _style(ax, "01 · Training and validation loss", "Epoch", "BCE loss")
    ax.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=TEXT)
    _save(fig, "01_loss_curves.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["lr"], color=PALETTE[1], marker="o",
            markersize=4, linewidth=2.5)
    ax.set_yscale("log")
    _style(ax, "02 · Learning-rate schedule", "Epoch", "Learning rate")
    _save(fig, "02_lr_schedule.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["val_ap"], color=PALETTE[0], marker="o",
            markersize=4, linewidth=2.5, label="Average precision")
    ax.plot(epochs, history["val_auc"], color=PALETTE[3], marker="o",
            markersize=4, linewidth=2.5, label="ROC AUC")
    ax.set_ylim(0, 1)
    _style(ax, "03 · Sampled validation fingerprint metrics", "Epoch", "Score")
    ax.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=TEXT)
    _save(fig, "03_ap_auc.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["val_bit_acc"], color=PALETTE[4], marker="o",
            markersize=4, linewidth=2.5)
    ax.set_ylim(0, 1)
    _style(ax, "04 · Sampled validation bit accuracy", "Epoch", "Accuracy")
    _save(fig, "04_bit_accuracy.png")

    probabilities, targets = _validation_sample(validation_path)
    precision, recall, _ = precision_recall_curve(
        targets.ravel(), probabilities.ravel()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, color=PALETTE[1], linewidth=2.5)
    ax.fill_between(recall, precision, color=PALETTE[1], alpha=0.2)
    _style(ax, "05 · Sampled validation precision–recall", "Recall", "Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "05_precision_recall.png")

    predicted_bits = probabilities.ravel() >= 0.5
    true_bits = targets.ravel().astype(bool)
    matrix = confusion_matrix(true_bits, predicted_bits, labels=[False, True])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="mako",
        xticklabels=["Off", "On"], yticklabels=["Off", "On"],
        linewidths=1, linecolor=BACKGROUND, cbar=False, ax=ax,
        annot_kws={"color": "white", "fontweight": "bold"},
    )
    _style(ax, "06 · Sampled validation bit confusion matrix",
           "Predicted bit", "True bit")
    _save(fig, "06_confusion_matrix.png")

    true_activity = targets.sum(axis=1)
    predicted_activity = probabilities.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(true_activity, predicted_activity, color=PALETTE[0],
               alpha=0.55, s=24, edgecolors="none")
    limit = max(float(true_activity.max()), float(predicted_activity.max()), 1)
    ax.plot([0, limit], [0, limit], color=PALETTE[2], linestyle="--",
            linewidth=2, label="Ideal agreement")
    _style(ax, "07 · Sampled predicted vs. true active fingerprint bits",
           "True active bits", "Predicted active bits")
    ax.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=TEXT)
    _save(fig, "07_fingerprint_activity.png")


def _training_sample(max_samples=2000):
    chunk_dir = CFG.PROC_DIR / "train_chunks"
    files = sorted(chunk_dir.glob("train_*.npz"))
    if files:
        count = min(8, len(files))
        indices = np.linspace(0, len(files) - 1, count, dtype=int)
        per_file = max(1, max_samples // count)
        spectra_parts, mass_parts = [], []
        rng = np.random.default_rng(42)
        for index in indices:
            with np.load(files[index], allow_pickle=True) as data:
                row_count = len(data["precursor_mz"])
                chosen = rng.choice(
                    row_count, size=min(per_file, row_count), replace=False
                )
                spectra_parts.append(data["spectra"][chosen])
                mass_parts.append(data["precursor_mz"][chosen])
        return np.concatenate(spectra_parts), np.concatenate(mass_parts)

    train_file = CFG.PROC_DIR / "train.npz"
    if train_file.is_file():
        with np.load(train_file, allow_pickle=True) as data:
            row_count = len(data["precursor_mz"])
            chosen = np.random.default_rng(42).choice(
                row_count, size=min(max_samples, row_count), replace=False
            )
            return data["spectra"][chosen], data["precursor_mz"][chosen]

    raise FileNotFoundError(
        f"No processed training data found in {chunk_dir} or {train_file}. "
        "Run preprocessing first."
    )


def generate_data_diagnostics():
    """Generate the three data-distribution charts from a bounded chunk sample."""
    spectra, precursor_mz = _training_sample()
    rng = np.random.default_rng(42)
    if len(spectra) > 1000:
        chosen = rng.choice(len(spectra), 1000, replace=False)
        spectra = spectra[chosen]
        precursor_mz = precursor_mz[chosen]

    if len(spectra) >= 3:
        feature_count = min(500, spectra.shape[1])
        feature_idx = rng.choice(spectra.shape[1], feature_count, replace=False)
        perplexity = min(30, max(1, (len(spectra) - 1) // 3))
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        ).fit_transform(spectra[:, feature_idx])
        fig, ax = plt.subplots(figsize=(10, 8))
        points = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=precursor_mz,
            cmap="viridis", s=22, alpha=0.8, linewidths=0,
        )
        fig.colorbar(points, ax=ax, label="Precursor m/z")
        _style(ax, "08 · t-SNE of sampled training spectra")
        _save(fig, "08_tsne.png")
    else:
        log.warning("Skipping t-SNE: fewer than three training spectra sampled.")

    peak_counts = np.count_nonzero(spectra, axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(peak_counts, bins=50, color=PALETTE[2],
                 edgecolor=BACKGROUND, ax=ax)
    _style(ax, "09 · Binned peak-count distribution",
           "Non-zero spectrum bins", "Sampled spectra")
    _save(fig, "09_peak_count.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(precursor_mz, bins=50, color=PALETTE[0],
                 edgecolor=BACKGROUND, ax=ax)
    _style(ax, "10 · Precursor mass distribution",
           "Precursor m/z", "Sampled spectra")
    _save(fig, "10_precursor_mass.png")
    log.info("All 10 charts generated from training metrics and sampled data.")


def main():
    generate_training_graphs()
    generate_data_diagnostics()


if __name__ == "__main__":
    main()
