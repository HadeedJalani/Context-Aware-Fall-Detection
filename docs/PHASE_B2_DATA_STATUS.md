# Phase B.2 — Dataset and Evaluation Status

## Why this phase exists

The first development run reached strong aggregate accuracy, but its test split was heavily dominated by `FALLEN` sequences. The next experiment therefore prioritizes class balance, realistic ADL negatives, and subject-disjoint evaluation.

## UP-Fall activity mapping

The improved 3D skeleton release describes 11 activities: 5 fall scenarios and 6 ADLs. Activities 1–5 are mapped to fall transitions; activities 6–11 are mapped to `NORMAL`.

For fall sequences, the first impact marker is used only as an anchor. The default temporal labeling is:

- before the falling window: `NORMAL`
- immediately preceding impact: `FALLING`
- impact and after: `FALLEN`

This is explicitly a derived temporal label, not a claim that the source dataset provides a native three-state annotation.

## Current supplied artifact

The local UP-Fall skeleton archive available during development contains the fall-side skeleton files for five subjects/camera views. It does **not** contain the complete six-ADL portion required for a final false-positive/generalization experiment.

Therefore the current development checkpoint must not be described as a final ADL-validated model.

## Phase B.2 pipeline

```text
UP-Fall skeleton CSVs
        ↓
MediaPipe-33 → COCO-17 mapping
        ↓
9 engineered temporal features
        ↓
30-frame windows
        ↓
subject-disjoint train / validation / test
        ↓
5-feature / 7-feature / 9-feature ablation
        ↓
Macro-F1 + class recall + FPR
        ↓
unseen-subject evaluation
```

## Reproducibility

```bash
python tools/prepare_upfall_full.py \
  --input data/raw/upfall \
  --output data/processed/upfall_frames.jsonl

python tools/build_subject_disjoint_sequences.py \
  --frames data/processed/upfall_frames.jsonl \
  --output data/processed/subject_disjoint

python tools/train_ablation.py \
  --data data/processed/subject_disjoint/sequences.npz \
  --output outputs/ablation_results.json
```

The repository deliberately does not commit the dataset or trained weights.
