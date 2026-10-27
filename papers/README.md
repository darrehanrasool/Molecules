# CASMI & Mass Spectrometry Research Literature Repository

This directory contains the state-of-the-art research literature, academic references, extracted methodologies, mathematical formulations, baseline benchmarks, and open research gap analyses for small molecule identification from mass spectra.

## Directory Layout

- `pdf/`: Full-text research papers, challenge descriptions, and clinical metabolomics surveys:
  - CASMI Competition Overview & Benchmark Evaluations
  - Dührkop et al. (2019), *Nature Biotechnology*, "SIRIUS 4: a rapid tool for turning tandem mass spectra into metabolite formulas and structures"
  - Dührkop et al. (2015), *PNAS*, "Searching molecular structure databases with tandem mass spectra using CSI:FingerID"
  - Huber et al. (2021), *PLOS Computational Biology*, "MS2DeepScore: a novel deep learning similarity measure to compare tandem mass spectra"
  - Rombouts et al. (2020), *Bioinformatics*, "Spec2Vec: Improved mass spectral similarity scoring through learning of structural relationships"
  - Bushuiev et al. (2023), *Bioinformatics*, "MassFormer: tandem mass spectrum representation learning with graph transformers"
- `metadata/`: Structured JSON catalog (`papers_metadata.json`) containing citation metadata, DOIs, architecture specifications, and reproducibility statuses.
- `literature_reviews/`: Systematic surveys comparing spectral library retrieval, neural fingerprint predictors, and de novo molecular graph generators (`msms_small_molecule_identification_survey.md`).
- `methodology_extraction/`: Mathematical equations, modified cosine formulas, spectral entropy, and MRR@25 ranking derivations (`mathematical_formulations.md`).
- `baseline_comparisons/`: Side-by-side benchmark comparison matrix evaluating reported metrics against baseline models (`benchmark_comparison_matrix.md`).
- `research_gaps/`: Critical examination of dark chemical matter, adduct ambiguity, and scaffold out-of-distribution generalization (`open_challenges.md`).
