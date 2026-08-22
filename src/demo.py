from __future__ import annotations

import argparse
import cv2
from ultralytics import YOLO

from .features import extract_features
from .behavior import PersonIncident
from .model import FallLSTM
from .sequence import TemporalBuffer, predict
from .onnx_sequence import ONNXTemporalClassifier
from .incident_manager import IncidentManager


def run(
    input_path,
    output_path=None,
    pose_model="yolo11n-pose.pt",
    classifier_path=None,
    onnx_model=None,
    metadata_path=None,
    report_dir="outputs/incidents",
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    pose = YOLO(pose_model)
    incidents = {}
    previous = {}
    previous_previous = {}
    manager = IncidentManager(report_dir)

    torch_classifier = None
    torch_temporal = None
    mean = std = None
    onnx_classifier = None

    if onnx_model:
        onnx_classifier = ONNXTemporalClassifier(
            onnx_model,
            metadata_path=metadata_path,
            window=30,
        )
    elif classifier_path:
        torch_classifier, ckpt = FallLSTM.from_checkpoint(classifier_path)
        mean = ckpt.get("feature_mean")
        std = ckpt.get("feature_std")
        torch_temporal = TemporalBuffer(30, 9)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = pose.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
        )[0]

        active = []

        if result.keypoints is not None and result.boxes is not None:
            ids = result.boxes.id
            ids = (
                ids.int().cpu().tolist()
                if ids is not None
                else list(range(len(result.keypoints.data)))
            )
            active = ids

            for i, tid in enumerate(ids):
                kp = result.keypoints.data[i].cpu().numpy()
                feat = extract_features(
                    kp,
                    previous.get(tid),
                    previous_previous.get(tid),
                    1 / max(fps, 1),
                )
                previous_previous[tid] = previous.get(tid, kp)
                previous[tid] = kp

                if onnx_classifier:
                    ready = onnx_classifier.append(tid, feat)
                    if ready:
                        label, confidence, probs = onnx_classifier.predict(tid)
                    else:
                        label, confidence = "NORMAL", 0.0
                elif torch_classifier:
                    ready = torch_temporal.append(tid, feat)
                    if ready:
                        label, confidence, _ = predict(
                            torch_classifier,
                            torch_temporal,
                            tid,
                            mean,
                            std,
                        )
                    else:
                        label, confidence = "NORMAL", 0.0
                else:
                    label, confidence = "NORMAL", float(feat[-1])

                person = incidents.setdefault(
                    tid,
                    PersonIncident(tid),
                )
                person.update(
                    label,
                    confidence,
                    feat[3],
                    abs(float(feat[3])),
                )
                risk = manager.update(
                    person,
                    confidence,
                    abs(float(feat[3])),
                )

                x1, y1, x2, y2 = (
                    result.boxes.xyxy[i]
                    .cpu()
                    .numpy()
                    .astype(int)
                )
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"ID {tid} {person.state.value} "
                    f"{confidence:.0%} RISK {risk.score}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

        if onnx_classifier:
            onnx_classifier.clear_missing(active)
        elif torch_temporal:
            torch_temporal.clear_missing(active)

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.add_argument("--pose-model", default="yolo11n-pose.pt")
    p.add_argument("--classifier")
    p.add_argument("--onnx-model")
    p.add_argument("--metadata")
    p.add_argument("--report-dir", default="outputs/incidents")
    a = p.parse_args()
    run(
        a.input,
        a.output,
        a.pose_model,
        a.classifier,
        a.onnx_model,
        a.metadata,
        a.report_dir,
    )
