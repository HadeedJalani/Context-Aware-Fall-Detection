# Phase B.2 — FUKinect ADL Training Results

## Dataset

The uploaded `skeleton.rar` was extracted and inspected locally. It contains 1,006 `.mat` skeleton recordings across six activities:

- walking
- bending
- sitting
- squatting
- lying
- falling

The source dataset documentation describes FUKinect-Fall as 21 subjects, six actions, eight repetitions, 20 Kinect joints, and approximately 4–5 seconds per recording at 30 FPS. The archive supplied for this experiment contained 1,006 usable recordings rather than the nominal 1,008.

## Processing

The 20-joint Kinect skeleton was mapped to the project's COCO-style body representation. The existing temporal LSTM architecture was retained, while the feature construction was adapted for 3D Kinect coordinates:

1. torso orientation
2. normalized vertical body position
3. body shape ratio
4. normalized vertical velocity
5. normalized vertical acceleration
6. mean joint angle
7. joint-angle variability
8. normalized body motion
9. normalized depth velocity

The final three-state target remains:

`NORMAL → FALLING → FALLEN`

Because FUKinect provides action labels rather than native three-state annotations, `FALLING` and `FALLEN` are derived temporal labels anchored around the strongest torso-orientation transition in each fall recording. These labels are therefore heuristic and are documented as such.

## Ablation

The same temporal LSTM family was evaluated with 5, 7 and 9 input features using a video-disjoint split. No individual video contributes windows to more than one split.

| Representation | Accuracy | Macro F1 | NORMAL Recall | FALLING Recall | FALLEN Recall |
|---|---:|---:|---:|---:|---:|
| 5 features | 79.34% | 74.23% | 79.52% | 87.80% | 76.03% |
| 7 features | 83.24% | 77.64% | 87.07% | 79.67% | 75.21% |
| **9 features** | **84.78%** | **79.84%** | **88.67%** | **90.24%** | **73.55%** |

The 9-feature representation was selected because it achieved the strongest overall accuracy and Macro F1 and, importantly, the strongest `FALLING` recall in this experiment.

## Unseen-subject stress test

A separate subject-disjoint experiment was also executed. Subjects were separated before temporal windows were generated. The result was substantially weaker than the video-disjoint result, which is expected to be treated as a genuine generalization warning rather than hidden.

The current subject-disjoint 9-feature run achieved approximately **47.7% accuracy and 31.3% Macro F1** on the held-out subjects.

This indicates that the compact dataset has meaningful subject/domain variation and that the current derived FALLING/FALLEN labels are not yet strong enough to support a claim of robust cross-subject generalization.

## Interpretation

The result is still useful for the project because it demonstrates three things:

- ADL data can be incorporated without using the 812 GB UP-Fall vision archive.
- The expanded temporal representation improves the video-disjoint result over the smaller feature sets.
- Subject-disjoint testing exposes a real robustness gap that should be addressed before claiming deployment-grade performance.

The 84.78% / 79.84% result is therefore a **development benchmark**, not a production or medical-performance claim.

## Next experiment

Before Phase C, the recommended next improvement is to strengthen temporal supervision around the fall transition and evaluate with subject-wise cross-validation. The behavioral state machine can then use the LSTM event signal together with post-fall motion/immobility evidence rather than relying on the classifier alone.
