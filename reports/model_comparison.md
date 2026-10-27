# Model Benchmark & Performance Evaluation Report

This report summarizes comparative performance metrics across baseline and neural approaches evaluated on our validation partitions.

## Evaluation Metric Summary (MRR@25)

| Model Name | Precursor Filtering | Feature Representation | MRR@25 | Top-1 Acc | Top-5 Acc | Top-25 Acc | Mean Latency (ms) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Library Match** | None | Raw Peaks Cosine | 0.2074 | 0.1840 | 0.2410 | 0.2850 | 8.5 |
| **Precursor Filtered Cosine** | $\pm 15$ ppm | Binned Dot Product | 0.2592 | 0.2210 | 0.3120 | 0.4150 | 28.4 |
| **Spectral Entropy Matcher** | $\pm 10$ ppm | Normalized Entropy | 0.3145 | 0.2640 | 0.3850 | 0.4920 | 36.1 |
| **MS2DeepScore Siamese** | $\pm 15$ ppm | Deep Siamese Embedding | 0.3802 | 0.3100 | 0.4900 | 0.6200 | 95.0 |
| **Neural FP + Tanimoto Retrieval** | $\pm 10$ ppm | Predicted Morgan 2048 | 0.4680 | 0.3820 | 0.5890 | 0.7250 | 54.2 |
| **Hybrid Consensus Reranker** | $\pm 10$ ppm | Multi-modal Score Fusion | **0.5380** | **0.4450** | **0.6720** | **0.8050** | **85.0** |

---

## Key Observations
1. **Precursor Filtering Impact**: Restricting candidate search spaces using instrument ppm tolerance eliminates >99.9% of spurious molecular formulas.
2. **InChIKey14 Deduplication**: Collapsing tautomers and stereoisomers into distinct InChIKey14 skeletons before truncating to top 25 yields a $+0.045$ boost in MRR@25.
3. **Consensus Superiority**: Ensembling spectral similarity with neural fingerprint tanimoto scores achieves the highest recall in top 25 candidates (80.5%).
