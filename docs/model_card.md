# Model Card: CASMI 2026 Small Molecule Identification System

## Model Details
- **Architecture**: Hybrid Precursor-Constrained Retrieval + Neural Spectral Embedding Reranker.
- **Model Version**: 1.0.0
- **Organization**: Centre of Artificial Intelligence, Islamic University of Science and Technology (IUST), Kashmir.
- **Supervisors & Authors**: Dr. Asif Assad, Dar Rehan Rasool, Muneeb Syed Shah, Shafiya Bashir Pampori.

## Intended Use
- **Primary Use Case**: Predicting 2D chemical structure candidates (SMILES) of small molecules and natural products from tandem mass spectra (LC-MS/MS).
- **Evaluation Metric**: Mean Reciprocal Rank @ 25 (MRR@25) matching InChIKey14 under RDKit 2026.03.3 tautomer canonicalization.

## Performance Boundaries & Limitations
- **Resolution Limit**: Performs best on high-resolution MS/MS spectra ($<15\,\text{ppm}$). Performance on low-resolution unit-mass instruments (ion traps) requires wider precursor search tolerances.
- **Novel Scaffold Generalization**: For structures with entirely novel scaffolds not present in candidate databases, de-novo graph generation is required.
- **Stereochemistry & Tautomers**: The model optimizes connectivity matching (InChIKey14) and does not distinguish stereoisomers or specific tautomeric states.
