# Preliminary Validation Metrics

- **Validation Fold**: 5-Fold Scaffold Partition (Stratified by InChIKey skeleton)
- **Total Evaluated Spectra**: 500 query spectra
- **Top-1 Hits**: 223 / 500 (44.6%)
- **Top-5 Hits**: 336 / 500 (67.2%)
- **Top-25 Hits**: 402 / 500 (80.4%)
- **Unranked / Misses (>25)**: 98 / 500 (19.6%)
- **Aggregate MRR@25**: **0.5380**
- **Average Inference Latency**: 85.0 ms / spectrum
- **Kaggle 9h Budget Capacity**: ~380,000 spectra (well within budget)
