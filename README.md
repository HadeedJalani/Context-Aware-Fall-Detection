# Context-Aware Fall Detection

<div align="center">

### From fall detection to post-fall understanding.

A real-time computer-vision research prototype that detects falls, interprets what happens **after** a fall, prioritizes risk across multiple people, and automatically generates structured incident reports.

**YOLO11-Pose · BoT-SORT · Temporal LSTM · Behavioral State Machine · Risk Engine · Incident Reporting · ONNX**

</div>

---

## Overview

Traditional fall detection often answers a single question:

> **Did this person fall?**

This project is designed to answer the more useful questions that follow:

- Did the person remain on the ground?
- Are they moving after the fall?
- Are they attempting to recover?
- Did they successfully recover?
- Which person requires the most urgent attention when multiple people are present?
- What exactly happened during the incident?

The system combines **pose estimation, multi-object tracking, temporal sequence modeling, behavioral reasoning, risk prioritization, and automated reporting** into one pipeline.

> **Research / educational prototype — not a certified medical or emergency-response device.**

---

## Why this project is different

The reference approach ends primarily at **FALLEN + persistent alerting**. This implementation adds an additional intelligence layer around the event.

| Capability | Conventional fall classifier | This project |
|---|:---:|:---:|
| Pose-based person detection | ✓ | ✓ |
| Multi-person tracking | ✓ | ✓ |
| Temporal LSTM classification | ✓ | ✓ |
| Expanded temporal features | — | ✓ |
| Post-fall movement analysis | — | ✓ |
| Recovery-attempt detection | — | ✓ |
| Recovery completion state | — | ✓ |
| Per-person risk score | — | ✓ |
| Multi-person risk prioritization | — | ✓ |
| Automatic incident record | — | ✓ |
| JSON / TXT / PDF reporting | — | ✓ |
| ONNX classifier export path | — | ✓ |

The result is intended to behave more like a **context-aware monitoring system** than a single-frame fall classifier.

---

## System architecture

```text
                         CAMERA / VIDEO
                               │
                               ▼
                        ┌──────────────┐
                        │ YOLO11-Pose  │
                        │ Person +     │
                        │ Keypoints    │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  BoT-SORT    │
                        │ Multi-person │
                        │   tracking   │
                        └──────┬───────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │ Context Feature Engine │
                   │        9 features      │
                   └───────────┬────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │ 30-frame Track     │
                    │ Temporal Buffer    │
                    └─────────┬──────────┘
                              │
                              ▼
                       ┌────────────┐
                       │    LSTM    │
                       └─────┬──────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
            NORMAL        FALLING        FALLEN
                                           │
                                           ▼
                              ┌────────────────────┐
                              │ Post-Fall Behavior  │
                              │     Analysis        │
                              └─────────┬──────────┘
                                        │
                              ┌─────────┼─────────┐
                              ▼         ▼         ▼
                           FALLEN  RECOVERING  RECOVERED
                              │
                              ▼
                       ┌──────────────┐
                       │  Risk Engine │
                       └──────┬───────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Multi-person     │
                    │ Risk Prioritizer │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Incident Manager │
                    └────────┬─────────┘
                             │
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
               JSON         TXT         PDF
```

---

## Context-aware feature representation

Instead of relying only on a small set of static pose measurements, the temporal model receives an expanded feature vector for each tracked person.

| # | Feature | Purpose |
|---:|---|---|
| 1 | Body angle | Captures orientation relative to the vertical axis |
| 2 | Body height / position | Tracks changes in the person's vertical location |
| 3 | Body shape | Helps distinguish upright and horizontally extended configurations |
| 4 | Vertical velocity | Captures the direction and speed of vertical movement |
| 5 | Vertical acceleration | Captures rapid changes in the movement trajectory |
| 6 | Mean joint angle | Describes overall body articulation |
| 7 | Joint-angle variability | Captures changes in pose configuration |
| 8 | Immobility | Measures sustained lack of movement |
| 9 | Pose confidence | Provides a confidence signal for downstream reasoning |

The important distinction is that these features are interpreted **over time**, not independently frame-by-frame.

---

## Post-fall behavior analysis

The LSTM provides the three core temporal classes:

```text
NORMAL → FALLING → FALLEN
```

The behavioral layer then adds higher-level post-fall states:

```text
NORMAL
   │
   ▼
FALLING
   │
   ▼
FALLEN
   │
   ├──────── movement detected ────────► RECOVERING
   │                                      │
   │                                      ▼
   │                                  RECOVERED
   │
   └──────── sustained immobility ─────► CRITICAL / high-risk handling
```

This allows the system to distinguish between a person who has fallen and is **attempting to get back up** and a person who remains on the ground with little or no movement.

An incident can therefore capture information such as:

```text
Fall detected:       14:32:17
Ground duration:     38.4 sec
Recovery attempts:   2
Recovery status:     Not recovered
Risk score:          91 / 100
Priority:            CRITICAL
```

---

## Multi-person risk prioritization

BoT-SORT provides persistent identities for multiple people. Rather than treating every fall equally, the risk layer evaluates each tracked person independently.

```text
Person #1   NORMAL       08 / 100   LOW
Person #2   FALLEN       91 / 100   CRITICAL   ← highest priority
Person #3   FALLING      76 / 100   HIGH
Person #4   RECOVERED    14 / 100   LOW
```

This creates a clear monitoring priority when several people are visible simultaneously.

The risk score is a **system-level prioritization signal**, not a medical diagnosis.

---

## Automatic incident reporting

When an incident reaches a terminal state, the reporting layer can produce structured records containing:

- Track / person ID
- Event type
- Detection timestamp
- Fall duration
- Ground duration
- Peak movement velocity
- Detection confidence
- Risk score
- Risk level
- Recovery attempts
- Recovery status
- Recommended action

Supported report formats:

```text
outputs/incidents/
├── incident_XXXX.json
├── incident_XXXX.txt
└── incident_XXXX.pdf
```

This turns an ephemeral video event into an auditable incident record suitable for later analysis and evaluation.

---

## Machine-learning pipeline

```text
Pose keypoints
      │
      ▼
Feature extraction
      │
      ▼
Per-track sequences
      │
      ▼
30-frame windows
      │
      ├── Train
      ├── Validation
      └── Test
             │
             ▼
        LSTM training
             │
             ▼
       Best checkpoint
             │
             ▼
       Held-out metrics
```

The training pipeline supports:

- Training / validation / test sequence generation
- Feature normalization using training statistics
- Class-weighted cross entropy
- AdamW optimization
- Gradient clipping
- Early stopping
- Held-out test evaluation
- Accuracy
- Macro F1
- Per-class recall
- Confusion matrix

### Expected dataset schema

Feature extraction produces a CSV with columns matching:

```text
video_id
track_id
frame
label
body_angle
body_height
body_shape
vertical_velocity
vertical_acceleration
joint_angle_mean
joint_angle_variability
immobility
pose_confidence
```

The sequence generator converts these rows into fixed-length temporal windows for LSTM training.

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/HadeedJalani/Context-Aware-Fall-Detection.git
cd Context-Aware-Fall-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run video inference

```bash
python -m src.demo \
  --input path/to/video.mp4 \
  --output outputs/demo.mp4
```

### 4. Prepare temporal sequences

```bash
python tools/make_sequences.py \
  --csv data/processed/features.csv \
  --output data/processed/sequences.npz \
  --length 30
```

### 5. Train the LSTM

```bash
python -m src.train \
  --data data/processed/sequences.npz \
  --output models/fall_lstm.pt \
  --report outputs/test_metrics.json
```

### 6. Run inference with a trained classifier

```bash
python -m src.demo \
  --input path/to/video.mp4 \
  --output outputs/demo.mp4 \
  --classifier models/fall_lstm.pt
```

### 7. Export the classifier to ONNX

```bash
python tools/export_onnx.py \
  --checkpoint models/fall_lstm.pt \
  --output models/fall_lstm.onnx
```

> Model weights, datasets, generated videos, and other large artifacts are intentionally kept out of version control.

---

## Repository structure

```text
Context-Aware-Fall-Detection/
│
├── src/
│   ├── behavior.py            # Post-fall behavioral state machine
│   ├── features.py            # Context-aware feature extraction
│   ├── model.py               # LSTM architecture
│   ├── sequence.py            # Per-track temporal buffers
│   ├── dataset.py             # Dataset / normalization utilities
│   ├── train.py               # Training + evaluation
│   ├── risk.py                # Risk scoring and prioritization
│   ├── incident_manager.py    # Incident lifecycle management
│   ├── reporting/              # JSON / TXT / PDF generation
│   ├── demo.py                # Video inference entry point
│   └── webcam.py              # Live-camera entry point
│
├── tools/
│   ├── make_sequences.py      # Feature CSV → temporal windows
│   └── export_onnx.py         # LSTM → ONNX
│
├── data/
│   └── processed/             # Dataset artifacts / schemas
│
├── models/                    # Local model checkpoints (not committed)
├── outputs/                   # Evaluation and incident outputs
├── tests/                     # Automated tests
├── requirements.txt
└── README.md
```

---

## Evaluation and scientific reporting

Performance claims should only be made using **held-out videos that were not used during training**.

Recommended evaluation protocol:

1. Split by **video**, not individual frames.
2. Keep camera angles / scenes separated where possible.
3. Evaluate on completely unseen recordings.
4. Report class-wise precision, recall and F1.
5. Report false-positive rate for non-fall activity.
6. Measure detection latency.
7. Measure recovery-state accuracy.
8. Test multiple people in the same scene.
9. Test challenging activities such as sitting, lying down, exercising and rapid crouching.

This prevents frame-level leakage from producing misleadingly high numbers.

> **No fabricated benchmark is included in this repository.** Actual performance numbers should be generated from the project's real held-out evaluation set.

---

## Design philosophy

### Temporal context over single-frame decisions

A single frame can make sitting, crouching, exercising, lying down and falling look deceptively similar. Temporal information provides the additional context required to separate these behaviors.

### Detection → interpretation → prioritization

The project deliberately separates three layers:

```text
Detection       What is happening?
Interpretation  What happened after it?
Prioritization  Who needs attention first?
```

### Research first, deployment second

The architecture is designed so that the research pipeline can be evaluated before deployment-specific concerns are introduced. ONNX export provides a path toward PyTorch-free inference and future edge-device benchmarking.

---

## Roadmap

### Phase B — Current

- [x] YOLO11-Pose integration
- [x] BoT-SORT multi-person tracking
- [x] Expanded temporal feature representation
- [x] LSTM temporal classifier
- [x] Per-track sequence buffering
- [x] Post-fall behavior states
- [x] Risk prioritization
- [x] Incident management
- [x] JSON / TXT / PDF reporting
- [x] Training and evaluation pipeline
- [x] ONNX classifier export path
- [x] Video inference entry point
- [x] Webcam entry point

### Phase C — Planned

- [ ] Real-time monitoring dashboard
- [ ] Live risk-ranking visualization
- [ ] Configurable alert escalation
- [ ] External notification integrations
- [ ] End-to-end ONNX inference
- [ ] Edge-device benchmarking
- [ ] Latency / FPS profiling
- [ ] Robustness evaluation across camera viewpoints
- [ ] Larger unseen-video benchmark

---

## Research positioning

This project is intended as a **research and engineering prototype** demonstrating how pose estimation and temporal modeling can be combined with higher-level behavioral reasoning.

Its main contribution is the transition from:

> **fall classification**

into:

> **context-aware post-fall monitoring and incident understanding.**

The system should therefore be evaluated not only on whether it detects a fall, but also on whether it correctly interprets the subsequent behavior and prioritizes incidents appropriately.

---

## Safety & limitations

This repository is provided for **educational and research purposes**.

It is not a certified medical device, emergency-response system, or substitute for human supervision. Computer-vision predictions can fail because of occlusion, poor lighting, unusual poses, camera placement, crowded scenes, tracking errors, or activities that resemble falls.

Real-world deployment would require extensive validation, fail-safe alerting, privacy controls, and domain-specific safety certification.

---

## License

See the repository license for the applicable terms.

---

<div align="center">

**Context-Aware Fall Detection**

*Detect the event. Understand what follows. Prioritize the risk.*

</div>
