# CASMI 2026 From-Zero Experiment Report

> This report must be completed from `casmi_run/manifests/final_run_manifest.json`
> and the generated metric files. Do not copy historical values from the old
> reports.

## 1. Research question

Can tandem mass spectra and precursor-ion information be used to rank the
correct 2D molecular connectivity within the top 25 candidates under the
competition's InChIKey14 MRR@25 metric?

## 2. Dataset and audit

| Quantity | Measured value |
|---|---:|
| Raw training rows | `...` |
| Raw test rows | `...` |
| Training columns | `...` |
| Valid SMILES fraction | `...` |
| Unique InChIKey14 groups | `...` |
| Adduct categories | `...` |
| Missing-value findings | `...` |

## 3. Leakage-safe experimental design

All spectra belonging to one chemical connectivity group were assigned to the
same partition using a deterministic seed. The train/validation overlap count
was `...` and must be zero. This is essential because one molecule may have
multiple collision-energy spectra.

## 4. Preprocessing

1. Remove non-finite and non-positive peaks.
2. Remove peaks above precursor `m/z + 2 Da`.
3. Remove peaks below 1% of the spectrum base peak.
4. Keep at most 128 most intense peaks.
5. Apply square-root intensity transform.
6. Bin at 0.1 Da from 0 to 1200 Da.
7. Normalize each spectrum by its maximum bin intensity.
8. Append normalized precursor `m/z`.

## 5. Mathematical formulation

The network predicts a 2048-dimensional Morgan fingerprint using
BCE-with-logits. Candidate similarity is continuous Tanimoto, followed by
neutral-mass filtering and InChIKey14 deduplication. The primary metric is
MRR@25:

\[
\mathrm{MRR@25}=\frac1N\sum_{i=1}^{N}
\begin{cases}
1/r_i,&r_i\le25\\
0,&\text{otherwise}.
\end{cases}
\]

## 6. Baselines and model

| System | Top-1 | Top-5 | Top-25 | MRR@25 |
|---|---:|---:|---:|---:|
| Precursor mass-only baseline | `...` | `...` | `...` | `...` |
| Neural fingerprint retrieval | `...` | `...` | `...` | `...` |

Model architecture: `...`  
Loss: `...`  
Optimizer and schedule: `...`  
Checkpoint selection criterion: validation MRR@25.

## 7. Candidate coverage

The correct target survived the 15 ppm mass filter for `...%` of validation
queries and the 30 ppm fallback for `...%`. Queries where the target is absent
from the candidate library are retrieval-impossible and must be reported
separately from ranking errors.

## 8. Results and limitations

The measured validation MRR@25 was `...`. The official hidden-test score was
`...` after Kaggle submission. Test accuracy was not claimed before the
leaderboard evaluation because test labels are withheld.

Known limitations include isobaric structures, missing candidate-library
coverage, adduct ambiguity, instrument/domain shift, and the information loss
from representing a full molecular graph with a fixed fingerprint.

## 9. Reproducibility

Attach:

- `final_run_manifest.json`;
- `data_contract.json`;
- `split.json`;
- `training_history.csv`;
- `validation_mass_coverage.csv`;
- selected checkpoint hash;
- final `submission.csv`;
- Kaggle notebook output and official score screenshot/export.
