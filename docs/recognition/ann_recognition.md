# ANN Recognition (From Scratch)

## Objective

Classify isolated words with a from-scratch feed-forward neural network trained on dictionary-derived features.

Implementation reference: `speech_ann.py` and ANN mode in `speech_recognition_app.py`.

## Input Representation

Variable-length feature sequence `X in R^(T x D)` is converted to fixed vector:
- `x = [mean(X), std(X)] in R^(2D)`

No deep-learning library is used for model logic.

## Network Architecture

Single-hidden-layer MLP:
- Input: `2D`
- Hidden: `H` units, ReLU
- Output: `C` classes, softmax

Parameters:
- `W1 in R^(2D x H)`, `b1 in R^H`
- `W2 in R^(H x C)`, `b2 in R^C`

## Training (SGD + Backprop)

Per mini-batch:
1. Forward:
   - `z1 = xW1 + b1`
   - `a1 = ReLU(z1)`
   - `z2 = a1W2 + b2`
   - `p = softmax(z2)`
2. Loss:
   - cross-entropy with one-hot labels.
3. Backward:
   - gradients for `W2,b2,W1,b1`.
4. Update:
   - `theta <- theta - lr * grad`.

Normalization:
- Train-set mean/std saved in model and reused at inference.

## Inference

1. Extract selected feature sequence from segment.
2. Convert to pooled vector.
3. Normalize with model stats.
4. Forward pass and argmax probability.

## Persistence

Model save/load via compressed pickle (`.ann.gz`) includes:
- weights/biases
- normalization stats
- feature type
- label map.

## Reimplementation Experiment Steps

1. Build dictionary and pick feature type.
2. Convert entries to `(X,y)` vectors.
3. Split train/test with fixed seed.
4. Train MLP with selected hyperparameters.
5. Evaluate accuracy/loss.
6. Save model and test on new recorded segmented words.

## Practical Notes

- If many classes exist, increase hidden units and epochs.
- Class imbalance strongly affects ANN accuracy; rebalance labels when possible.
