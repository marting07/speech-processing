# HMM Online Recognition (Train-at-Compare)

## Objective

Recognize isolated words with HMMs where models are trained on-the-fly from current dictionary entries each time recognition is executed.

Implementation references:
- `speech_hmm.py`
- call path in `speech_recognition_app.py::compare_audio`

## Model Family

Two variants:
1. `hmm-discrete`
2. `hmm-continuous` (diagonal Gaussian emissions)

Both use left-right topology:
- state `i` can stay at `i` or move to `i+1`.

## Discrete HMM Pipeline

1. Gather all training frames from dictionary sequences.
2. Learn codebook with from-scratch k-means (`_kmeans_codebook`).
3. Quantize each feature frame to symbol index (`_quantize_sequence`).
4. Train one HMM per label via Baum-Welch (`_train_discrete_hmm`):
   - forward/backward with scaling
   - gamma/xi accumulation
   - update `pi`, `A`, `B`
5. Quantize query sequence.
6. Score each label model with forward log-likelihood.
7. Predict label with highest log-likelihood.

## Continuous HMM Pipeline

1. For each label, collect feature sequences.
2. Initialize per-state Gaussian params by temporal partitioning.
3. Train via Baum-Welch-style updates (`_train_continuous_hmm`):
   - update `pi`, `A`, means `mu`, variances `var`
4. Score query by forward log-likelihood.
5. Predict label with highest score.

## Reimplementation Steps

1. Fix feature type and extraction parameters.
2. Group dictionary sequences by label.
3. Train models at inference time (current design).
4. Evaluate on held-out utterances.
5. Report:
   - accuracy
   - average compare latency (includes re-training cost).

## Important Note

This mode is intentionally expensive because training is done during compare; use the persisted-HMM mode when you need offline-trained reusable models.
