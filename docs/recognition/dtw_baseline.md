# DTW Baseline Recognition

## Objective

Recognize isolated words by directly comparing a query feature sequence against dictionary feature sequences using Dynamic Time Warping (DTW).

## From-Scratch Computation

Implementation reference: `speech_dsp.py::dtw_distance`.

1. Input sequences:
   - Query: `Q in R^(Nq x D)`
   - Reference: `R in R^(Nr x D)`
2. Local frame distance:
   - `d(i,j) = ||Q_i - R_j||_2`
3. Dynamic programming matrix `C`:
   - size `(Nq+1) x (Nr+1)`
   - initialize `C(0,0)=0`, rest `inf`
4. Recurrence (weighted diagonal):
   - `C(i,j)=min(C(i-1,j-1)+2d(i,j), C(i-1,j)+d(i,j), C(i,j-1)+d(i,j))`
5. Normalized score:
   - `DTW = C(Nq,Nr)/(Nq+Nr)`
6. Decision:
   - predict label of reference with minimum DTW score.

## Data Flow in App

1. Segment utterance.
2. Extract selected feature type (`mfcc`, `bark`, `lpc`, `wavelet`).
3. Filter dictionary to same feature type.
4. Run DTW query-vs-entry and choose minimum distance.

## Experiment Reimplementation Steps

1. Build a dictionary with labeled isolated words.
2. Hold out a test set not used to build templates.
3. For each test utterance:
   - compute feature matrix
   - compare against all dictionary templates with DTW
   - record predicted label
4. Compute overall and per-class accuracy.

## Recommended First Experiment

- Feature: `mfcc`
- One template per class (then extend to multi-template)
- Compare effect of segmentation quality on DTW accuracy.
