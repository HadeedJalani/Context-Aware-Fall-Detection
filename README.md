# Context-Aware Fall Detection

A research-oriented real-time fall detection prototype using **YOLO11-Pose + BoT-SORT + an LSTM temporal model**, extended with context-aware post-fall analysis.

## Differentiating features

- **Post-fall behavior state machine:** `NORMAL -> FALLING -> FALLEN -> RECOVERING -> RECOVERED`
- **Multi-person risk prioritization:** every tracked person receives an independent 0–100 risk score and severity level.
- **Expanded features:** body angle, body position, body shape, vertical velocity, acceleration placeholder, joint-angle statistics, immobility and pose confidence.
- **Incident reporting:** structured JSON/TXT and PDF reports containing fall duration, ground duration, peak velocity, confidence, recovery attempts and recommended action.

## Architecture

```text
Camera / Video
      |
      v
YOLO11-Pose
      |
      v
BoT-SORT multi-person tracking
      |
      v
Context-aware feature extraction
      |
      v
LSTM temporal classification
      |
      v
Post-fall behavior state machine
      |
      v
Per-person risk engine + prioritization
      |
      v
Incident JSON / TXT / PDF reports
```

## Project status

This is **Phase B**: the strong academic/portfolio version. Model weights and datasets are intentionally not bundled. The repository contains the architecture and core modules; the trained temporal checkpoint will be added after the dataset/training pipeline is finalized.

Phase C (dashboard, external alerts, edge benchmarking and further deployment work) will be added later.

## Run

```bash
pip install -r requirements.txt
python -m src.demo --input path/to/video.mp4 --output output/result.mp4
```

The demo currently uses the YOLO pose/tracking path and the new context modules. The LSTM checkpoint is intentionally a separate artifact so training/evaluation remains reproducible.

## Safety

This is a research/educational prototype, not a certified medical or emergency-response device.
