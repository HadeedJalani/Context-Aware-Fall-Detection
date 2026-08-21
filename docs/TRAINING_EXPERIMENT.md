# Training Experiment — UP-Fall Development Run

## Objective

Train a temporal classifier using the project's nine engineered pose features and optimize for **Macro F1 and FALLING recall**, not raw accuracy alone.

## Configuration

- Input: 30-frame temporal windows
- Features: 9
- Classes: `NORMAL`, `FALLING`, `FALLEN`
- Model: 2-layer GRU, 64 hidden units
- Regularization: dropout + AdamW weight decay
- Loss: moderately class-weighted cross entropy
- Normalization: training-split mean/std
- Selection: validation Macro F1 with early stopping

## Development result

| Metric | Value |
|---|---:|
| Accuracy | **97.81%** |
| Macro F1 | **0.877** |
| NORMAL recall | 1.000 |
| FALLING recall | **0.639** |
| FALLEN recall | 0.994 |

Confusion matrix (true × predicted):

```text
[[  6,   0,   0],
 [  1,  23,  12],
 [  0,   5, 774]]
```

The earlier development run reported ~89.6% accuracy, Macro F1 ~0.36 and FALLING recall 0.0. The revised training recipe therefore represents a substantial improvement in class-balanced performance.

## Dataset caveat

This is **not yet the final benchmark**. The supplied UP-Fall-derived sequence artifact is strongly imbalanced and the test set contains only 6 NORMAL, 36 FALLING and 779 FALLEN sequences. The current experiment is not subject-disjoint external validation and the supplied artifact does not contain the full ADL-rich portion needed for a credible false-positive/generalization claim.

## Next validation gate

Before publishing final performance numbers:

1. Add the ADL portion of UP-Fall.
2. Split by subject/video, not random temporal windows.
3. Re-train the 5-feature baseline and 9-feature model on the same split.
4. Report Macro F1, per-class recall, FALLING recall and false-positive rate.
5. Test on a completely unseen subject set.
6. Run a cross-dataset test using UR Fall or another independently collected dataset.

This report intentionally distinguishes **development metrics** from final generalization claims.
