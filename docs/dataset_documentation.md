# Dataset Documentation & Schema Specifications

## 1. Primary Competition Dataset

The competition provides MS/MS spectral data for small molecules detected in complex biological extracts.

### Core Fields
- `molecule_id`: Unique identifier for each query mass spectrum (e.g., `m_0014ef`).
- `precursor_mz`: High-resolution precursor ion mass-to-charge ratio ($m/z$).
- `adduct`: Ionization adduct type (e.g., `[M+H]+`, `[M+NH4]+`, `[M-H]-`).
- `collision_energy`: Collision energy setting (eV) utilized in CID fragmentation.
- `peaks`: Array of $(m/z, \text{intensity})$ pairs representing fragment ions.
- `smiles`: (Training set only) Canonical ground truth 2D chemical structure in SMILES format.

---

## 2. External Reference Datasets Allowed

The competition rules allow publicly available external data and pre-trained models. Key reference libraries include:
- **MassBank of North America (MoNA)**: Millions of curated experimental and in-silico mass spectra.
- **GNPS (Global Natural Products Social Molecular Networking)**: Community-curated natural product spectra.
- **HMDB (Human Metabolome Database)**: Comprehensive endogenous metabolite library with experimental and predicted MS/MS spectra.
- **PubChem Compound**: Canonical 2D structure catalog (>110M compounds) filtered by natural-product-like molecular formulas.
