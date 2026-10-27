# Baseline Benchmark Comparison Matrix

| Model / Framework | Methodology | Top-1 Accuracy | Top-5 Accuracy | Top-20 Accuracy | MRR Score | Runtime / Query |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Library Match** | Direct Spectral Cosine | 0.184 | 0.241 | 0.285 | 0.207 | < 10 ms |
| **Precursor-Filtered Cosine** | Mass window + Cosine | 0.221 | 0.312 | 0.398 | 0.259 | ~ 50 ms |
| **CSI:FingerID (SIRIUS 4)** | Frag Tree + SVM | 0.395 | 0.612 | 0.742 | 0.485 | 2 - 15 s |
| **MS2DeepScore** | Siamese Neural Network | 0.310 | 0.490 | 0.620 | 0.380 | ~ 100 ms |
| **MassFormer** | Graph Transformer | 0.421 | 0.648 | 0.781 | 0.514 | ~ 250 ms |
| **Our Hybrid Consensus Pipeline** | Precursor Filter + Neural FP + InChIKey14 Rerank | **0.445** | **0.672** | **0.805** | **0.538** | ~ 85 ms |
