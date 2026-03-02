# HMM Persisted Recognition (Pre-trained Bundles)

## Objective

Train HMM models offline once, persist them, and later run inference without re-training at compare time.

Implementation reference: `speech_hmm_persist.py`.

## Why This Mode

Compared with online HMM mode, this separates:
- training phase
- inference phase

This is better for repeatability and deployment-like workflows.

## Training Phase

Inputs:
- Dictionary entries
- feature type (`mfcc`/`bark`/`lpc`/`wavelet`)
- model type (`discrete` or `continuous`)
- `n_states`, `n_iters`, and `codebook_size` for discrete mode

Process:
1. Group sequences by label.
2. If discrete:
   - train shared codebook with from-scratch k-means
   - quantize sequences
   - train one discrete HMM per label
3. If continuous:
   - train one Gaussian HMM per label
4. Pack bundle metadata + models.

Outputs:
- Bundle dict containing model parameters per label.

## Persistence

- Save/load compressed bundle (`.hmm.gz`) with:
  - model type
  - feature type
  - optional codebook
  - per-label HMM parameters.

## Inference Phase

1. Extract query features using bundle feature type.
2. Score query against each label model:
   - discrete: quantize then forward log-likelihood
   - continuous: Gaussian forward log-likelihood
3. Rank by score descending.
4. Predict top label.

## Reimplementation Experiment Steps

1. Freeze dictionary snapshot.
2. Train bundle and save to disk.
3. Evaluate on held-out utterances:
   - load bundle
   - extract features
   - score all labels
   - compute Top-1 accuracy.
4. Report:
   - training time (offline)
   - model file size
   - per-query inference time.

## Comparison Recommendation

Compare persisted HMM against:
- DTW baseline
- online HMM
- ANN
- proximity index + DTW

using same feature type and same test split.
