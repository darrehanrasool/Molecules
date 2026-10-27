# Methodology: Algorithmic Formulations & Modeling Strategies

## 1. Mathematical Similarity Functions

Comparing an experimental MS/MS spectrum $S_A$ against a reference spectrum $S_B$:

### A. Modified Cosine Similarity
$$\text{Cos}(S_A, S_B) = \frac{\sum_{i,j} w_{A,i} w_{B,j}}{\sqrt{\sum_i w_{A,i}^2} \sqrt{\sum_j w_{B,j}^2}}$$
Where peak weights combine intensity and $m/z$:
$$w = I^m \cdot (m/z)^n \quad (m \approx 0.5, \, n \approx 2.0)$$
In modified cosine, peaks are matched if:
1. Direct fragment match: $|m/z_{A,i} - m/z_{B,j}| \le \text{tolerance}$
2. Mass-shifted match (accounting for precursor mass difference $\Delta M$):
$$|(m/z_{A,i} - m/z_{B,j}) - \Delta M| \le \text{tolerance}$$

### B. Spectral Entropy Similarity
Spectral entropy measures the information content of a mass spectrum:
$$S = -\sum_{i=1}^k p_i \ln p_i \quad \text{where } p_i = \frac{I_i}{\sum_j I_j}$$
The entropy similarity score between spectra $A$ and $B$:
$$\text{Sim}_{\text{entropy}}(A, B) = 1 - \frac{2 S_{AB} - S_A - S_B}{\ln 4}$$
Where $S_{AB}$ is the entropy of the merged spectrum.

---

## 2. Neural Molecular Fingerprint Prediction

Instead of direct library lookup, we train a neural network $f_\theta$ to map the binned spectrum $x \in \mathbb{R}^D$ to the probability vector of molecular substructure bits $\hat{y} \in [0, 1]^M$ (e.g., Morgan / ECFP6 fingerprint, 2048 bits):

$$\hat{y} = \sigma(f_\theta(x))$$

### Loss Function: Focal Binary Cross-Entropy
$$\mathcal{L}(y, \hat{y}) = -\sum_{j=1}^{M} \left[ \alpha (1 - \hat{y}_j)^\gamma y_j \ln \hat{y}_j + (1 - \alpha) \hat{y}_j^\gamma (1 - y_j) \ln(1 - \hat{y}_j) \right]$$

---

## 3. Top-25 Candidate Selection & InChIKey14 Optimization

Because evaluation uses InChIKey14:
1. For every candidate SMILES $c_j$, compute $K_j = \text{InChIKey14}(c_j)$.
2. Group candidates by $K_j$, keeping the maximum confidence score within each group:
$$S(K_j) = \max_{c \in K_j} \text{Confidence}(c)$$
3. Rank unique $K_j$ descending by $S(K_j)$ and select up to 25 unique representations:
$$\hat{\mathcal{C}} = [c_{(1)}, c_{(2)}, \dots, c_{(25)}]$$
This guarantees no candidate slot is wasted on redundant stereocenters or tautomers.
