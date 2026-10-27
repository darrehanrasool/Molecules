# Extracted Mathematical Formulations in Computational MS/MS

## 1. Mass Accuracy and Tolerance Calculations

Given theoretical mass $m_{\text{th}}$ and experimental mass $m_{\text{exp}}$:

$$\Delta m_{\text{ppm}} = \frac{|m_{\text{exp}} - m_{\text{th}}|}{m_{\text{th}}} \times 10^6$$

The search window in Daltons for a specified PPM tolerance $\tau$:

$$\Delta m_{\text{Da}} = m_{\text{th}} \times \frac{\tau}{10^6}$$

For example, at $m/z = 500.0$ and $\tau = 10\,\text{ppm}$, $\Delta m_{\text{Da}} = 500 \times 10^{-5} = 0.005\,\text{Da}$.

---

## 2. Mean Reciprocal Rank (MRR@K)

For $N$ evaluation queries, with $\text{rank}_i$ indicating the position of the first ground-truth matching candidate:

$$\text{MRR@K} = \frac{1}{N} \sum_{i=1}^{N} \psi(\text{rank}_i)$$

Where:
$$\psi(r) = \begin{cases} \frac{1}{r} & \text{if } 1 \le r \le K \\ 0 & \text{if } r > K \text{ or unranked} \end{cases}$$

Here, $K = 25$, providing bounded scoring with hyper-sensitivity to top-3 placement.

---

## 3. Tanimoto Structural Similarity

For two binary fingerprint vectors $u, v \in \{0, 1\}^D$:

$$T(u, v) = \frac{|u \cap v|}{|u \cup v|} = \frac{\sum_{j=1}^D u_j v_j}{\sum_{j=1}^D u_j + \sum_{j=1}^D v_j - \sum_{j=1}^D u_j v_j}$$
