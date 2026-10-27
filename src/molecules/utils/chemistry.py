"""Chemistry utilities for monoisotopic mass and adduct calculations."""

from typing import Dict

# Exact monoisotopic masses of common chemical elements & particles
ELEMENT_MASSES: Dict[str, float] = {
    "H": 1.00782503223,
    "C": 12.00000000000,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Na": 22.9897692820,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "K": 38.9637064864,
    "Br": 78.9183376,
    "I": 126.9044719,
    "electron": 0.000548579909,
    "proton": 1.007276466621,
}

# Mass shifts for canonical electrospray ionization (ESI) adducts
ADDUCT_SHIFTS: Dict[str, float] = {
    "[M+H]+": 1.007276,
    "[M+NH4]+": 18.033823,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M]+": -0.0005486,
    "[M-H]-": -1.007276,
    "[M+Cl]-": 34.969402,
    "[M+FA-H]-": 44.998201, # Formate adduct
    "[M+HCOO]-": 44.998201,
    "[M+CH3COO]-": 59.013851, # Acetate adduct
    "[M]-": 0.0005486,
}


def precursor_mz_to_neutral_mass(precursor_mz: float, adduct: str = "[M+H]+") -> float:
    """Calculates neutral monoisotopic mass from precursor m/z and adduct type."""
    shift = ADDUCT_SHIFTS.get(adduct.strip())
    if shift is None:
        # Fallback heuristic: positive defaults to [M+H]+, negative to [M-H]-
        if "+" in adduct:
            shift = 1.007276
        elif "-" in adduct:
            shift = -1.007276
        else:
            shift = 1.007276
    return precursor_mz - shift


def ppm_error(experimental_mass: float, theoretical_mass: float) -> float:
    """Calculates absolute mass accuracy error in parts per million (ppm)."""
    if theoretical_mass == 0.0:
        return 0.0
    return abs(experimental_mass - theoretical_mass) / theoretical_mass * 1e6


def within_ppm_tolerance(m1: float, m2: float, tolerance_ppm: float = 10.0) -> bool:
    """Checks whether two masses fall within a parts-per-million tolerance window."""
    return ppm_error(m1, m2) <= tolerance_ppm


def calculate_monoisotopic_mass(formula: str) -> float:
    """Calculates exact monoisotopic mass from a simple Hill system formula string."""
    import re
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    total_mass = 0.0
    for element, count_str in tokens:
        count = int(count_str) if count_str else 1
        if element in ELEMENT_MASSES:
            total_mass += ELEMENT_MASSES[element] * count
    return total_mass
