# Metric Evaluation: Mean Reciprocal Rank @ 25 (MRR@25) & InChIKey14 Matching

## Overview

In the **Enveda CASMI 2026** competition, models are evaluated using **Mean Reciprocal Rank @ 25 (MRR@25)**. This metric measures the rank position of the first correct 2D chemical structure candidate within a list of up to 25 ranked guesses.

---

## 1. Mathematical Formulation

Given a test dataset with $N$ molecules, for each molecule $i \in \{1, \dots, N\}$, the model outputs a ranked list of up to 25 candidate SMILES strings:

$$\hat{\mathcal{S}}_i = \left( s_{i,1}, s_{i,2}, \dots, s_{i,K_i} \right), \quad 1 \le K_i \le 25$$

Let $\text{rank}_i$ be the 1-based index of the first candidate $s_{i,k}$ whose canonical 2D connectivity matches the ground truth molecule $y_i$:

$$\text{rank}_i = \min \left\{ k \in \{1, \dots, K_i\} \mid \text{Match}(s_{i,k}, y_i) = \text{True} \right\}$$

If no candidate matches the ground truth (or if $K_i < 25$ and none matches), we define $\text{rank}_i = \infty$, giving a score of $0$.

The Mean Reciprocal Rank @ 25 is computed as:

$$\text{MRR@25} = \frac{1}{N} \sum_{i=1}^{N} \text{RR}_i$$

Where the reciprocal rank $\text{RR}_i$ is defined as:

$$\text{RR}_i = \begin{cases} 
\frac{1}{\text{rank}_i} & \text{if } 1 \le \text{rank}_i \le 25 \\
0.0 & \text{if no correct guess in top 25}
\end{cases}$$

### Scoring Schedule Table

| Rank of 1st Correct Guess | Score ($\text{RR}_i$) | Rank of 1st Correct Guess | Score ($\text{RR}_i$) |
| :---: | :---: | :---: | :---: |
| **1** | **1.0000** | **14** | 0.0714 |
| **2** | **0.5000** | **15** | 0.0667 |
| **3** | **0.3333** | **16** | 0.0625 |
| **4** | **0.2500** | **17** | 0.0588 |
| **5** | **0.2000** | **18** | 0.0556 |
| **6** | **0.1667** | **19** | 0.0526 |
| **7** | **0.1429** | **20** | 0.0500 |
| **8** | **0.1250** | **21** | 0.0476 |
| **9** | **0.1111** | **22** | 0.0455 |
| **10** | **0.1000** | **23** | 0.0435 |
| **11** | 0.0909 | **24** | 0.0417 |
| **12** | 0.0833 | **25** | **0.0400** |
| **13** | 0.0769 | **> 25 / None** | **0.0000** |

---

## 2. Chemical Structure Matching Rule: InChIKey14

The competition evaluates **2D atom connectivity**. Stereochemistry (cis/trans, R/S stereocenters) and tautomeric resonance forms are not penalized.

### Verification Protocol

1. **RDKit Tautomer Canonicalization (pinned to 2026.03.3)**:
   ```python
   from rdkit import Chem
   from rdkit.Chem.MolStandardize import rdMolStandardize

   def canonicalize_tautomer(smiles: str) -> str:
       mol = Chem.MolFromSmiles(smiles)
       if mol is None:
           return ""
       te = rdMolStandardize.TautomerEnumerator()
       canon_mol = te.Canonicalize(mol)
       return Chem.MolToInchiKey(canon_mol)
   ```

2. **InChIKey Truncation (InChIKey14)**:
   An InChIKey consists of 27 uppercase characters arranged in three blocks separated by hyphens:
   - **Block 1 (14 characters)**: Encodes molecular skeleton and atom connectivity.
   - **Block 2 (10 characters)**: Encodes stereochemistry and isotopic substitution.
   - **Block 3 (1 character)**: Encodes protonation state / net charge.

   $$\text{InChIKey14} = \text{InChIKey}[:14]$$

   A prediction is scored as **Correct** if and only if:
   $$\text{InChIKey14}(s_{i,k}) == \text{InChIKey14}(y_i)$$

### Canonical Example: Glucose

Both of the following representations evaluate to identical InChIKey14 (`WQZGKKKJIJFFOK`):
- `OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O` (Chiral explicit D-glucose)
- `OCC1OC(O)C(O)C(O)C1O` (Achiral flat representation)

Both achieve rank 1 and score 1.0 against the reference answer.

---

## 3. Submission Format Validation Rules

1. **File Name**: Strictly `submission.csv`.
2. **Required Columns**: `molecule_id` and `smiles`.
3. **Delimiter**: Predictions are joined by semicolons (`;`).
4. **Candidate Bound**: Maximum 25 candidate SMILES strings per molecule.
5. **Coverage**: Every `molecule_id` from the test set must appear exactly once.
6. **No Nulls**: Neither column may contain null or empty values.
