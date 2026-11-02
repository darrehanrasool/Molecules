# Algorithms

## 1. Preprocessing

- Peak cleaning: relative-intensity floor 1%, drop > precursor+2 Da, top-128 peaks.
- Binning: 0.1 Da up to 1200 Da → 12,000-dim dense vector.
- sqrt intensity transform, L∞-normalised per spectrum.

## 2. Model

SpectrumFPNet: MLP(12001 → 2048 → 1024 → 2048 sigmoid bits).
Loss: BCEWithLogits. Optimizer: AdamW + cosine LR.
Output = Morgan (radius 2, 2048 bits) fingerprint.

## 3. Retrieval

Tanimoto similarity between predicted fingerprint and 275k-structure DB.
Dedup by InChIKey14, top-25 returned.

## 4. Metric

MRR@25 — matches on InChIKey14 (tautomer + stereo insensitive).

## 5. Failure modes

- Isomers → same bitset, wrong structure.
- Novel scaffolds absent from DB (class 3).
- Adduct ambiguity → wrong neutral mass filtering.
