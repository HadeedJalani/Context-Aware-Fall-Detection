# FUKinect-Fall ADL Integration Plan

## Why this dataset

The project will use the skeleton portion of the FUKinect-Fall dataset as the compact ADL source. The public dataset description reports 21 subjects, 6 actions, 8 repetitions per action, and 20-joint 3D skeleton coordinates. The six actions are walking, bending, sitting, squatting, lying and falling.

This is deliberately being used as a **compact ADL/generalization dataset**, not as a replacement for the project's primary YOLO11-Pose video pipeline.

## Mapping into this project

| FUKinect action | Project state |
|---|---|
| Walking | NORMAL |
| Bending | NORMAL |
| Sitting | NORMAL |
| Squatting | NORMAL |
| Lying | NORMAL |
| Falling | FALLING → FALLEN temporal samples |

The fall class will not be treated as a single static label. Temporal windows around the fall event will be used to construct the project's three-state sequence task.

## Evaluation protocol

- Split by subject, never by individual frames.
- Keep every repetition from a subject in the same split.
- Generate temporal windows only after the subject split.
- Fit normalization statistics on the training subjects only.
- Report accuracy, Macro F1, per-class precision/recall/F1, FALLING recall, and false-positive rate on NORMAL ADLs.
- Preserve the existing 9-feature representation so the dataset changes the evidence available to the model, not the project's core feature definition.

## Data handling

The repository does **not** commit the dataset archive. The public GitHub mirror contains a 22.4 MB `skeleton.rar` archive. After obtaining the archive, extract it locally under:

```text
data/raw/fukinect/
```

The next preprocessing step will inspect the extracted skeleton format and map its 20 Kinect joints into the subset required by `src/features.py`. No hard-coded joint ordering is being guessed before the actual archive is inspected.

## Scientific positioning

FUKinect-Fall is particularly useful here because its normal activities include exactly the kinds of movements that can generate false positives in a fall detector: bending, squatting, sitting and lying. The dataset was created with Kinect V1 and contains 3D skeleton coordinates, making it substantially lighter to integrate than a multi-hundred-gigabyte RGB/depth corpus.

The dataset remains an **external skeleton-domain evaluation/training source**. Final claims about camera-based real-world performance will continue to require evaluation on unseen RGB/video recordings.
