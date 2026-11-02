"""Parsing helpers for uploaded spectrum files."""
import io
import re

import numpy as np
import pandas as pd


def _normalize_column_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_spectrum_csv(contents):
    """Return m/z and intensity arrays from a headered CSV upload."""
    try:
        frame = pd.read_csv(io.BytesIO(contents))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read the uploaded CSV: {exc}") from exc

    columns = {
        _normalize_column_name(column): column
        for column in frame.columns
    }
    if {"moleculeid", "smiles"}.issubset(columns):
        raise ValueError(
            "This is a submission template (molecule_id, smiles), not a "
            "spectrum file. Upload measured peak data with m/z and intensity "
            "columns to run a prediction."
        )

    mz_aliases = (
        "mz", "mzda", "masscharge", "massovercharge", "masstocharge",
    )
    intensity_aliases = (
        "intensity", "relativeintensity", "normalizedintensity",
        "relintensity", "abundance",
    )
    mz_column = next(
        (columns[name] for name in mz_aliases if name in columns), None
    )
    intensity_column = next(
        (columns[name] for name in intensity_aliases if name in columns), None
    )

    if mz_column is None or intensity_column is None:
        found_columns = ", ".join(str(column) for column in frame.columns)
        raise ValueError(
            "This is not a spectrum CSV: expected named m/z and intensity "
            "columns (for example: m/z,intensity). "
            f"Found columns: {found_columns or '(none)'}."
        )

    mzs = pd.to_numeric(frame[mz_column], errors="coerce").to_numpy(
        dtype=np.float32
    )
    intensities = pd.to_numeric(
        frame[intensity_column], errors="coerce"
    ).to_numpy(dtype=np.float32)
    if not len(mzs) or not len(intensities):
        raise ValueError("The CSV must contain at least one spectrum peak.")
    if not np.isfinite(mzs).all() or not np.isfinite(intensities).all():
        raise ValueError(
            "The m/z and intensity columns must contain only finite numbers."
        )
    return mzs, intensities
