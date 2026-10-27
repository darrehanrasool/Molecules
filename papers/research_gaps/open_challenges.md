# Open Research Challenges in Natural Product Identification

## 1. Dark Chemical Matter & Novel Scaffolds
Over 70% of microbial and plant secondary metabolites possess structural motifs absent from synthetic catalogs. Models reliant strictly on nearest-neighbor catalog lookups fail when the correct chemical skeleton is uncataloged.

## 2. In-Source Fragmentation & Adduct Ambiguity
Molecules often undergo non-canonical in-source fragmentations or form unexpected adduct species (e.g. $[M-H_2O+H]^+$, $[2M+H]^+$ dimers). Mislabeled precursor masses invalidate candidate retrieval windows.

## 3. Instrument Variance & Collision Energy Discrepancies
Fragmentation profiles vary widely across instrument geometries (Orbitrap vs timsTOF vs Q-TOF) and collision energy regimes (10 eV vs 50 eV). Robust models must learn collision-energy-conditioned representations.

## 4. Tautomer and Stereocenter Invariance
The CASMI 2026 challenge explicitly evaluates InChIKey14 connectivity. Many existing generative chemistry models waste probability mass distinguishing stereocenters rather than optimizing connectivity coverage.
