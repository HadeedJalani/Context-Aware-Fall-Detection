from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from .features import extract_features
from .onnx_sequence import ONNXTemporalClassifier


def run_webcam(
    camera: int = 0,
    pose_model: str = "yolo11n-pose.pt",
    onnx_model: str = "models/context_aware_fall_lstm.onnx",
    metadata: str = "models/model_metadata.json",
    conf: float = 0.35,
    imgsz: int = 640,
):
    """Run the context-aware fall detector on a local webcam."""
    if not Path(onnx_model).exists():
        raise FileNotFoundError(
            f"ONNX classifier not found: {onnx_model}. "
            "Copy context_aware_fall_lstm.onnx into models/ or pass --onnx-model."
        )
    if not Path(metadata).exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata}. "
            "Copy model_metadata.json into models/ or pass --metadata."
        )

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera}")

    pose = YOLO(pose_model)
    classifier = ONNXTemporalClassifier(
        onnx_model, metadata_path=metadata, window=30
    )

    previous = {}
    previous_previous = {}

    print("Context-Aware Fall Detection — webcam test")
    print("Press Q or ESC to stop.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read a frame from the webcam.")
            break

        result = pose.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )[0]

        active_ids = []

        if result.keypoints is not None and result.boxes is not None:
            ids = result.boxes.id
            ids = [] if ids is None else ids.int().cpu().tolist()

            for i, track_id in enumerate(ids):
                keypoints = result.keypoints.data[i].cpu().numpy()
                features = extract_features(
                    keypoints,
                    previous.get(track_id),
                    previous_previous.get(track_id),
                    1.0 / 30.0,
                )

                previous_previous[track_id] = previous.get(track_id, keypoints)
                previous[track_id] = keypoints
                active_ids.append(track_id)

                ready = classifier.append(track_id, features)
                if ready:
                    label, confidence, probs = classifier.predict(track_id)
                else:
                    label = "WARMING UP"
                    confidence = len(classifier.buffers[track_id]) / 30.0
                    probs = None

                box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box.tolist()

                if label == "FALLEN":
                    color = (0, 0, 255)
                elif label == "FALLING":
                    color = (0, 215, 255)
                elif label == "NORMAL":
                    color = (0, 200, 0)
                else:
                    color = (255, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {track_id} | {label} | {confidence:.0%}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                if probs is not None:
                    cv2.putText(
                        frame,
                        f"N {probs[0]:.0%}  F {probs[1]:.0%}  D {probs[2]:.0%}",
                        (x1, min(frame.shape[0] - 10, y2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        classifier.clear_missing(active_ids)

        cv2.putText(
            frame,
            "Context-Aware Fall Detection | Q/ESC: quit",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Context-Aware Fall Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Context-Aware Fall Detection using a local webcam"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument(
        "--onnx-model", default="models/context_aware_fall_lstm.onnx"
    )
    parser.add_argument(
        "--metadata", default="models/model_metadata.json"
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    run_webcam(
        camera=args.camera,
        pose_model=args.pose_model,
        onnx_model=args.onnx_model,
        metadata=args.metadata,
        conf=args.conf,
        imgsz=args.imgsz,
    )
