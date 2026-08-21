# Phase B.2 — FUKinect ADL Training Status

## Dataset selected

The project now uses **FUKinect-Fall** as the compact ADL/fall skeleton dataset for Phase B.2. The supplied archive was extracted and verified locally.

It contains 1,006 usable `.mat` skeleton recordings across six activities:

- walking
- bending
- sitting
- squatting
- lying
- falling

The published dataset description specifies 21 subjects, six actions, eight repetitions, 20 Kinect joints, and approximately 4–5 seconds per recording at 30 FPS.

## Why this dataset

The original UP-Fall vision dataset is far too large for the current development environment. FUKinect gives the project real ADL negatives and fall examples in a compact 3D skeleton representation without changing the project's core temporal-learning idea.

## Feature adaptation

The existing LSTM family is retained. Kinect's 20-joint 3D representation is converted into a context-aware temporal feature vector.

### Current 9-feature representation

1. torso orientation
2. normalized vertical body position
3. body shape ratio
4. normalized vertical velocity
5. normalized vertical acceleration
6. mean joint angle
7. joint-angle variability
8. normalized body motion
9. normalized depth velocity

The final three-state model remains:

```text
NORMAL → FALLING → FALLEN
```

FUKinect does not provide native `FALLING` / `FALLEN` frame annotations. Those two states are therefore derived around the strongest torso-orientation transition in each fall recording. This is explicitly treated as a heuristic temporal label.

## Evaluation protocol

For the main development benchmark, videos are split before temporal windows are created so no windows from the same recording appear in multiple splits.

A separate subject-disjoint stress test is also run. This is the stricter generalization experiment and is reported separately.

## Current results

| Features | Accuracy | Macro F1 | NORMAL Recall | FALLING Recall | FALLEN Recall |
|---|---:|---:|---:|---:|---:|
| 5 | 79.34% | 74.23% | 79.52% | 87.80% | 76.03% |
| 7 | 83.24% | 77.64% | 87.07% | 79.67% | 75.21% |
| **9** | **84.78%** | **79.84%** | **88.67%** | **90.24%** | **73.55%** |

The 9-feature representation is the current development winner because it provides the strongest overall accuracy and Macro F1 while also giving the strongest FALLING recall.

### Subject-disjoint stress test

The current 9-feature subject-disjoint run achieved approximately **47.7% accuracy / 31.3% Macro F1** on held-out subjects. This is intentionally reported rather than hidden: it shows that cross-subject generalization remains the main research weakness.

## Reproducibility

```bash
python tools/prepare_fukinect.py \
  --input data/raw/fukinect/skeleton \
  --output data/processed/fukinect_sequences.npz

python tools/train_fukinect_ablation.py \
  --data data/processed/fukinect_sequences.npz \
  --output outputs/fukinect_ablation.json
```

The dataset, generated sequences, and model weights remain local artifacts and are not committed to GitHub.

## Next step

Before Phase C, improve the temporal supervision around impact and run subject-wise cross-validation. The post-fall behavior engine should then combine the LSTM event signal with movement/immobility evidence rather than depending on a single classifier decision.
