# Systematic Review: Small Molecule Structure Elucidation from Tandem Mass Spectra

## Abstract
Small molecule identification from liquid chromatography tandem mass spectrometry (LC-MS/MS) is the foundational bottleneck in natural products discovery and untargeted metabolomics. This survey reviews the evolution of computational methods from early rule-based fragment interpretation to modern deep graph transformers and generative foundation models.

---

## 1. Paradigm Taxonomy

Current computational strategies fall into three primary categories:

```
                      Computational MS/MS Identification
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
1. Spectral Library Matching   2. In-Silico Structure Retrieval   3. De Novo Molecular Generation
  • Direct Peak Matching         • Fragmentation Trees (SIRIUS)     • Autoregressive SMILES (Spec2Mol)
  • Modified Cosine (GNPS)       • CSI:FingerID (SVM)               • Diffusion Models on Graphs
  • Deep Siamese (MS2DeepScore)  • Neural Fingerprint (MassFormer)  • Chemical Language Models
```

### Paradigm 1: Spectral Library Matching
- **Strengths**: High precision for known molecules present in public repositories (GNPS, MassBank, MoNA).
- **Failure Modes**: Utterly fails on dark chemical matter (molecules whose experimental spectra were never previously recorded). In biological extracts, >80% of detected peaks lack library matches.

### Paradigm 2: In-Silico Structure Candidate Retrieval
- **Approach**: For an unknown precursor mass, retrieve all candidate structures from massive chemical catalogs (PubChem, ChemSpider) matching the monoisotopic mass. Predict simulated spectra or predicted molecular fingerprints and rank candidates.
- **State-of-the-Art**: SIRIUS / CSI:FingerID, CFM-ID 4.0, MassFormer.
- **Relevance to CASMI 2026**: High. The competition allows external candidate databases and scores top 25 candidates using MRR@25.

### Paradigm 3: De Novo Chemical Graph Generation
- **Approach**: Generate novel SMILES or 2D molecular graphs directly from the spectrum conditioning without needing pre-existing candidate databases.
- **Challenges**: High frequency of chemically invalid structures, difficulty respecting precursor mass constraints without constrained decoding.
