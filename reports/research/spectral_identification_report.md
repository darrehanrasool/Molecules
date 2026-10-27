# Research Report: Spectral Representation & InChIKey14 Optimization in CASMI 2026

**Authors**: Dar Rehan Rasool, Dr. Asif Assad, Muneeb Syed Shah, Shafiya Bashir Pampori  
**Affiliation**: Centre of Artificial Intelligence, Islamic University of Science and Technology (IUST), Kashmir  

## Executive Summary
The Enveda CASMI 2026 challenge evaluates models on Mean Reciprocal Rank @ 25 (MRR@25) utilizing RDKit 2026.03.3 tautomer canonicalization and InChIKey14 matching. Because the evaluation is invariant to stereocenters and tautomers, a standard ranked list containing multiple stereoisomers (e.g. D-glucose, L-glucose, alpha-D-glucose) unnecessarily consumes scarce ranking budget.

We designed a dual-stage architecture:
1. Candidate retrieval filtered by high-resolution precursor monoisotopic mass ($\pm 10\,\text{ppm}$).
2. Score assignment using a deep neural fingerprint predictor trained on tandem MS fragmentation spectra.
3. Post-processing InChIKey14 deduplication that collapses structural variants to their maximal confidence score.

Empirical results demonstrate an improvement in MRR@25 from 0.2592 (precursor cosine baseline) to 0.5380 (hybrid consensus model).
