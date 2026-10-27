"""IO utilities for saving and loading data and predictions."""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Loads a tabular dataset from CSV or Parquet format with fallback."""
    p = Path(path)
    if not p.exists():
        # Check if fallback csv exists
        if p.suffix == ".parquet" and p.with_suffix(".csv").exists():
            return pd.read_csv(p.with_suffix(".csv"))
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if p.suffix == ".parquet":
        try:
            return pd.read_parquet(p)
        except ImportError:
            # Fallback to csv if parquet engine is missing
            csv_path = p.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    elif p.suffix == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Saves a dataframe to CSV or Parquet, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".parquet":
        try:
            df.to_parquet(p, index=False)
            return
        except ImportError:
            # If pyarrow/fastparquet is not installed, fallback to CSV
            csv_path = p.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return
    elif p.suffix == ".csv":
        df.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")


def load_json(path: str) -> Any:
    """Loads a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Saves data to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
