# Recognition Systems Reimplementation Guide

This folder documents each recognition system in this project at implementation level, focused on how to reproduce the experiments with **from-scratch logic** (no ready-made ANN/HMM/DTW/indexing libraries).

## Documents

1. [DTW Baseline](./dtw_baseline.md)
2. [HMM Online (Train-at-Compare)](./hmm_online.md)
3. [Proximity Index Recognition](./proximity_indexes.md)
4. [ANN Recognition (From Scratch)](./ann_recognition.md)
5. [HMM Persisted (Pre-trained Bundles)](./hmm_persisted.md)

## Shared Experimental Protocol

1. Fix sampling and frame setup (default in project):
   - `fs=8000 Hz`, `frame=30 ms`, `hop=10 ms`, `window=Hamming`, `pre-emphasis=0.97`.
2. Use a single feature type per experiment (`mfcc` recommended first).
3. Ensure labels are balanced enough across classes.
4. Split data (train/test) with fixed random seed for reproducibility.
5. Track metrics:
   - Top-1 accuracy
   - Per-class accuracy (or confusion matrix)
   - Inference time per utterance
6. Keep experiment metadata:
   - feature type, segmentation parameters, model hyperparameters, dictionary size.

## Reproducibility Checklist

- Use the same dictionary snapshot for all methods being compared.
- Keep feature extraction parameters identical between train and test.
- Keep random seed fixed when training ANN or splitting data.
- Report both performance and compute cost (train/query time).
