"""Pytest fixtures and test configuration."""

import pytest
import numpy as np
from molecules.data.loader import MassSpectrumRecord


@pytest.fixture
def sample_spectrum() -> MassSpectrumRecord:
    """Fixture providing a mock LC-MS/MS spectrum record."""
    peaks = np.array([
        [105.033, 0.42],
        [153.018, 0.85],
        [209.045, 0.23],
        [254.057, 1.00], # Base peak
        [255.065, 0.15],
    ], dtype=np.float32)

    return MassSpectrumRecord(
        molecule_id="m_0014ef",
        precursor_mz=255.0652,
        adduct="[M-H]-",
        peaks=peaks,
        collision_energy=50.0,
        smiles="O=C1C=C(c2ccccc2)Oc3cc(O)cc(O)c13", # Chrysin
    )


@pytest.fixture
def glucose_isomers():
    """Fixture providing glucose stereoisomers from competition overview."""
    return {
        "chiral": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "achiral": "OCC1OC(O)C(O)C(O)C1O",
        "expected_inchikey14": "WQZGKKKJIJFFOK",
    }
