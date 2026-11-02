"""Torch Dataset for binned spectra → Morgan fingerprint."""
import numpy as np, torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from src.config import CFG
from src.utils import set_seed

class ChunkedSpectrumDataset(IterableDataset):
    def __init__(self, files, split, val_frac, seed):
        self.files = tuple(files)
        self.split = split
        self.val_frac = val_frac
        self.seed = seed
        self.row_counts = []

        for path in self.files:
            with np.load(path, allow_pickle=True) as data:
                self.row_counts.append(len(data["precursor_mz"]))

        self.total_rows = sum(self.row_counts)
        self.split_rows = sum(
            int(count * val_frac) for count in self.row_counts
        )

    def __len__(self):
        if self.split == "validation":
            return self.split_rows
        return self.total_rows - self.split_rows

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1

        for file_index in range(worker_id, len(self.files), worker_count):
            with np.load(self.files[file_index], allow_pickle=True) as data:
                spectra = data["spectra"].astype(np.float32, copy=False)
                precursor_mz = data["precursor_mz"]
                fingerprints = data["fp"].astype(np.float32)

            precursor_feature = (
                precursor_mz / 1000.0
            ).reshape(-1, 1).astype(np.float32)
            features = np.concatenate(
                [spectra, precursor_feature], axis=1
            )

            rng = np.random.default_rng(self.seed + file_index)
            indices = rng.permutation(self.row_counts[file_index])
            val_count = int(self.row_counts[file_index] * self.val_frac)
            if self.split == "validation":
                indices = indices[:val_count]
            else:
                indices = indices[val_count:]
                if worker:
                    rng = np.random.default_rng(worker.seed + file_index)
                    rng.shuffle(indices)
                else:
                    rng.shuffle(indices)

            for index in indices:
                yield torch.from_numpy(features[index]), torch.from_numpy(
                    fingerprints[index]
                )

def get_dataloaders(val_frac=0.05, batch=CFG.BATCH, workers=CFG.NUM_WORKERS):
    set_seed()
    chunk_dir = CFG.PROC_DIR / "train_chunks"
    files = sorted(chunk_dir.glob("train_*.npz"))
    if not files:
        train_file = CFG.PROC_DIR / "train.npz"
        if train_file.is_file():
            files = [train_file]
        else:
            raise FileNotFoundError(
                f"No processed training data found in {chunk_dir}. "
                "Run preprocessing first: "
                "python -m src.preprocessing"
            )

    tr = ChunkedSpectrumDataset(files, "train", val_frac, CFG.SEED)
    va = ChunkedSpectrumDataset(files, "validation", val_frac, CFG.SEED)
    return (
        DataLoader(tr, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True),
        DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True),
    )