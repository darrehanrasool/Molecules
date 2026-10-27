"""Unit tests for RDKit tautomer canonicalization and InChIKey14 matching."""

from molecules.evaluation.canonicalization import (
    smiles_to_inchikey14,
    are_structures_equivalent,
)


def test_glucose_tautomer_canonicalization(glucose_isomers):
    """Verifies that glucose stereoisomers map to the canonical competition InChIKey14."""
    key_chiral = smiles_to_inchikey14(glucose_isomers["chiral"])
    key_achiral = smiles_to_inchikey14(glucose_isomers["achiral"])

    assert key_chiral == glucose_isomers["expected_inchikey14"]
    assert key_achiral == glucose_isomers["expected_inchikey14"]
    assert key_chiral == key_achiral
    assert are_structures_equivalent(glucose_isomers["chiral"], glucose_isomers["achiral"])


def test_empty_or_invalid_smiles():
    assert smiles_to_inchikey14("") is None
    assert smiles_to_inchikey14("   ") is None
    assert smiles_to_inchikey14(None) is None


def test_different_molecules_not_equivalent(glucose_isomers):
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    assert not are_structures_equivalent(glucose_isomers["chiral"], aspirin)
