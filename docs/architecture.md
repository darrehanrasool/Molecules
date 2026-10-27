# Platform Architecture & Pipeline Design

## End-to-End Inference Pipeline

The solution pipeline is structured into five loosely coupled, modular stages designed to run offline within Kaggle's 9-hour runtime constraint:

```
[ Raw LC-MS/MS Spectrum ]
        │
        ▼
[ Stage 1: Spectral Preprocessing & Peak Filtering ]
        │ - Noise removal (<0.5% base peak)
        │ - Precursor m/z extraction & adduct identification
        │ - Dense 0.1 Da binning & neutral loss calculation
        ▼
[ Stage 2: Candidate Generation & Mass Filtering ]
        │ - Exact precursor monoisotopic mass calculation
        │ - Precursor mass tolerance window (±10 ppm)
        │ - Candidate retrieval from external compound databases (PubChem / HMDB / GNPS)
        ▼
[ Stage 3: Deep Neural Scoring & Spectral Embedding ]
        │ - MassFormer / Spec2Vec neural spectrum encoder
        │ - Predicted molecular fingerprint (Morgan Chiral 2048)
        │ - Cosine & entropy spectral similarity against candidate in-silico spectra
        ▼
[ Stage 4: Deduplication & Candidate Reranking ]
        │ - RDKit Tautomer Canonicalization (pinned to 2026.03.3)
        │ - Collapse identical InChIKey14 candidates (retain top score)
        │ - Hybrid consensus score fusion
        ▼
[ Stage 5: Top-25 Semicolon-Separated Formatter ]
        │ - Select top <= 25 ranked candidate SMILES
        │ - Generate compliant submission.csv
```

## Stage Breakdown

### 1. Spectral Preprocessing (`molecules.data.preprocessor`)
Mass spectra can contain thousands of low-intensity noise spikes and baseline artifacts.
- Filter peaks below $0.5\%$ relative intensity.
- Normalize peak intensities against the base peak (abundance = 1.0) or Total Ion Current (TIC).
- Calculate neutral loss spectra: $\Delta m/z = \text{precursor } m/z - \text{fragment } m/z$.

### 2. Candidate Filtering (`molecules.models.retrieval`)
For an observed precursor mass $m/z_{\text{obs}}$ and adduct $A$:
$$\text{Neutral Mass} = m/z_{\text{obs}} \cdot |z| - \text{mass}(A)$$
We filter external compound catalogs within a narrow $\pm 10\,\text{ppm}$ tolerance window:
$$\left| \frac{\text{Candidate Mass} - \text{Neutral Mass}}{\text{Neutral Mass}} \right| \times 10^6 \le 10$$

### 3. Neural Scoring (`molecules.models.neural_encoder`)
A 1D-CNN and MLP encoder projects the binned spectrum into a continuous 256-dimensional embedding space, trained to predict the molecular fingerprint of the true structure.

### 4. Reranking and Deduplication (`molecules.postprocessing.ranker`)
Since the metric evaluates InChIKey14, submitting multiple stereoisomers or tautomers that share the exact same InChIKey14 wastes candidate slots. Our ranker maps each candidate to its InChIKey14, keeps only the highest-scoring candidate for each distinct connectivity skeleton, and selects up to 25 unique candidates.
