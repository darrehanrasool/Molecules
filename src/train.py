"""Train the spectrum-to-fingerprint model."""
import argparse
import os
import time, json
import numpy as np, torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.config import pick_device
from src.config import CFG
from src.dataset import get_dataloaders
from src.model import SpectrumFPNet
from src.utils import set_seed, get_logger

log = get_logger("train")
LOG_EVERY_BATCHES = 100
METRIC_SAMPLE_ROWS = 1024


def evaluate(model, loader, device, lossf):
    model.eval()
    total_loss = 0.0
    total_rows = 0
    total_batches = len(loader)
    sampled_probabilities = []
    sampled_targets = []
    sampled_rows = 0

    with torch.no_grad():
        for batch_index, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            total_loss += lossf(logits, yb).item() * xb.size(0)
            total_rows += xb.size(0)
            sample_count = min(METRIC_SAMPLE_ROWS - sampled_rows, xb.size(0))
            if sample_count:
                sampled_probabilities.append(
                    torch.sigmoid(logits[:sample_count]).cpu().numpy()
                )
                sampled_targets.append(yb[:sample_count].cpu().numpy())
                sampled_rows += sample_count

            if batch_index == 1 or batch_index % LOG_EVERY_BATCHES == 0:
                log.info(
                    f"Validation metrics: batch {batch_index}/{total_batches} "
                    f"({100 * batch_index / total_batches:.1f}%)"
                )

    if not total_rows or not sampled_rows:
        raise RuntimeError("Validation loader produced no examples.")

    from sklearn.metrics import average_precision_score, roc_auc_score

    P = np.concatenate(sampled_probabilities)
    Y = np.concatenate(sampled_targets)
    ap = average_precision_score(Y.ravel(), P.ravel())
    auc = roc_auc_score(Y.ravel(), P.ravel())
    acc = ((P > 0.5).astype(np.uint8) == Y).mean()
    return total_loss / total_rows, ap, auc, acc, P, Y


def _save_json(path, value):
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
    temp_path.replace(path)


def _save_validation_sample(probabilities, targets):
    path = CFG.LOG_DIR / "validation_sample.npz"
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp_path.open("wb") as file:
        np.savez_compressed(
            file,
            probabilities=probabilities.astype(np.float32),
            targets=targets.astype(np.uint8),
        )
    temp_path.replace(path)


def _validate_resume_history(history, completed_epoch):
    required_keys = (
        "epoch", "train_loss", "val_loss", "val_ap", "val_auc",
        "val_bit_acc", "lr",
    )
    if not isinstance(history, dict) or any(
        key not in history for key in required_keys
    ):
        raise ValueError(
            "Resume checkpoint contains incomplete training history."
        )

    epochs = history["epoch"]
    if epochs != list(range(1, completed_epoch + 1)):
        raise ValueError(
            "Resume checkpoint epoch does not match its completed history."
        )
    if any(len(history[key]) != completed_epoch for key in required_keys):
        raise ValueError(
            "Resume checkpoint history has inconsistent lengths."
        )


def _load_resume(model, opt, sched, device, from_best=False):
    checkpoint_path = CFG.MODEL_DIR / "resume_checkpoint.pt"
    if checkpoint_path.is_file() and not from_best:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        required = (
            "model", "optimizer", "scheduler", "epoch", "history", "best_ap",
        )
        if any(key not in checkpoint for key in required):
            raise ValueError(
                f"Resume checkpoint is incomplete: {checkpoint_path}"
            )

        completed_epoch = int(checkpoint["epoch"])
        _validate_resume_history(checkpoint["history"], completed_epoch)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        sched.load_state_dict(checkpoint["scheduler"])
        log.info(
            f"Resuming from full training checkpoint after epoch "
            f"{completed_epoch}; optimizer and scheduler state restored."
        )
        return (
            completed_epoch,
            checkpoint["history"],
            float(checkpoint["best_ap"]),
            sched,
        )

    best_path = CFG.MODEL_DIR / "best_model.pt"
    history_path = CFG.LOG_DIR / "history.json"
    if not best_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(
            "No resume checkpoint or compatible best_model.pt/history.json "
            "found. Train once without --resume first."
        )

    checkpoint = torch.load(best_path, map_location=device)
    with history_path.open(encoding="utf-8") as file:
        history = json.load(file)
    completed_epoch = int(history["epoch"][-1]) if history.get("epoch") else 0
    _validate_resume_history(history, completed_epoch)
    if completed_epoch == 0 or "ap" not in checkpoint:
        raise ValueError(
            "The existing best-model checkpoint cannot be matched to a "
            "completed epoch in history.json."
        )
    best_ap = float(checkpoint["ap"])
    if from_best:
        best_index = int(np.argmax(history["val_ap"]))
        if not np.isclose(
            best_ap, history["val_ap"][best_index], rtol=0.0, atol=1e-8
        ):
            raise ValueError(
                "best_model.pt does not match the best AP in history.json."
            )
        completed_epoch = int(history["epoch"][best_index])
        history = {key: values[:best_index + 1] for key, values in history.items()}
        model.load_state_dict(checkpoint["model"])
        for group in opt.param_groups:
            group["initial_lr"] = CFG.LR
        sched = CosineAnnealingLR(
            opt,
            T_max=CFG.EPOCHS,
            last_epoch=completed_epoch - 1,
        )
        log.warning(
            f"Starting from best weights at epoch {completed_epoch} "
            f"(AP={best_ap:.4f}); optimizer state is fresh."
        )
        return completed_epoch, history, best_ap, sched

    latest_ap = float(history["val_ap"][-1])
    if (
        not np.isclose(best_ap, latest_ap, rtol=0.0, atol=1e-8)
        or latest_ap < max(history["val_ap"])
    ):
        raise ValueError(
            "best_model.pt does not match the latest completed epoch in "
            "history.json. Refusing to resume from mismatched weights."
        )

    model.load_state_dict(checkpoint["model"])
    for group in opt.param_groups:
        group["initial_lr"] = CFG.LR
    sched = CosineAnnealingLR(
        opt,
        T_max=CFG.EPOCHS,
        last_epoch=completed_epoch - 1,
    )
    log.warning(
        f"Resuming weights from epoch {completed_epoch} (AP={best_ap:.4f}); "
        "the original optimizer state was not saved, so AdamW state is fresh."
    )
    return completed_epoch, history, best_ap, sched


def _save_resume_checkpoint(model, opt, sched, epoch, history, best_ap, in_dim):
    path = CFG.MODEL_DIR / "resume_checkpoint.pt"
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_ap": best_ap,
            "in_dim": in_dim,
        },
        temp_path,
    )
    temp_path.replace(path)


def main(resume=False, from_best=False):
    set_seed()
    device = pick_device()
    log.info(f"Device: {device}")
    log.info("Loading processed training chunks and preparing data loaders...")
    tr_loader, va_loader = get_dataloaders()
    log.info(
        f"Data ready: {len(tr_loader.dataset):,} training rows "
        f"({len(tr_loader):,} batches), "
        f"{len(va_loader.dataset):,} validation rows "
        f"({len(va_loader):,} batches)."
    )
    in_dim = CFG.N_BINS + 1
    model = SpectrumFPNet(in_dim, CFG.FP_BITS).to(device)
    log.info(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=CFG.EPOCHS)
    lossf = nn.BCEWithLogitsLoss()

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_ap": [], "val_auc": [], "val_bit_acc": [], "lr": []}
    best_ap = -1.0
    start_epoch = 0
    if resume:
        start_epoch, history, best_ap, sched = _load_resume(
            model, opt, sched, device, from_best=from_best
        )
        if start_epoch >= CFG.EPOCHS:
            raise ValueError(
                f"Training already reached epoch {start_epoch}/{CFG.EPOCHS}."
            )
        log.info(f"Next training epoch: {start_epoch + 1}/{CFG.EPOCHS}.")

    for epoch in range(start_epoch + 1, CFG.EPOCHS + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        total_batches = len(tr_loader)
        log.info(
            f"Epoch {epoch:02d}/{CFG.EPOCHS}: training started "
            f"({total_batches:,} batches)."
        )
        for batch_index, (xb, yb) in enumerate(tr_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = lossf(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item() * xb.size(0)
            if batch_index == 1 or batch_index % LOG_EVERY_BATCHES == 0:
                elapsed = time.time() - t0
                log.info(
                    f"Epoch {epoch:02d}/{CFG.EPOCHS} training: "
                    f"batch {batch_index:,}/{total_batches:,} "
                    f"({100 * batch_index / total_batches:.1f}%), "
                    f"loss={loss.item():.4f}, elapsed={elapsed / 60:.1f} min"
                )
        tr_loss = running / len(tr_loader.dataset)

        log.info(f"Epoch {epoch:02d}/{CFG.EPOCHS}: validation started.")
        v_loss, ap, auc, bacc, metric_probs, metric_targets = evaluate(
            model, va_loader, device, lossf
        )
        log.info(
            f"Validation loss uses all {len(va_loader.dataset):,} rows; "
            f"AP/AUC/bit accuracy use a {len(metric_targets):,}-row sample."
        )
        lr_now = opt.param_groups[0]["lr"]
        sched.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["val_ap"].append(ap)
        history["val_auc"].append(auc)
        history["val_bit_acc"].append(bacc)
        history["lr"].append(lr_now)

        log.info(f"Epoch {epoch:02d} | tr {tr_loss:.4f} | va {v_loss:.4f} "
                 f"| AP {ap:.4f} | AUC {auc:.4f} | bitacc {bacc:.4f} "
                 f"| lr {lr_now:.2e} | {time.time()-t0:.1f}s")

        _save_json(CFG.LOG_DIR / "history.json", history)
        _save_validation_sample(metric_probs, metric_targets)
        from src.visualize import generate_training_graphs
        generate_training_graphs(history)

        if ap > best_ap:
            best_ap = ap
            torch.save({"model": model.state_dict(), "in_dim": in_dim,
                        "ap": ap, "epoch": epoch},
                       CFG.MODEL_DIR / "best_model.pt")
            log.info(f"  ↳ saved best (AP={ap:.4f})")

        _save_resume_checkpoint(
            model, opt, sched, epoch, history, best_ap, in_dim
        )

    torch.save({"model": model.state_dict(), "in_dim": in_dim},
               CFG.MODEL_DIR / "final_model.pt")
    _save_json(CFG.LOG_DIR / "history.json", history)
    log.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    resume_options = parser.add_mutually_exclusive_group()
    resume_options.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest checkpoint or verified best-model history",
    )
    resume_options.add_argument(
        "--from-best",
        action="store_true",
        help="restart from best_model.pt and its matching completed epoch",
    )
    args = parser.parse_args()
    main(resume=args.resume or args.from_best, from_best=args.from_best)