from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

# MediaPipe BlazePose-33 -> the COCO-17 joints consumed by src.features.
MP_TO_COCO = {0: 0, 5: 11, 6: 12, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}
# UP-Fall: 1-5 are falls; 6-11 are ADLs in the improved skeleton release.
FALL_ACTIVITIES = {1, 2, 3, 4, 5}
ADL_ACTIVITIES = {6, 7, 8, 9, 10, 11}


def load_pose(df: pd.DataFrame) -> np.ndarray:
    kp = np.zeros((len(df), 17, 3), dtype=np.float32)
    for coco, mp in MP_TO_COCO.items():
        kp[:, coco, 0] = df[f"Joint{mp + 1}_X"].to_numpy(np.float32)
        kp[:, coco, 1] = df[f"Joint{mp + 1}_Y"].to_numpy(np.float32)
        # The supplied skeleton CSVs do not expose BlazePose confidence per joint.
        # Use finite coordinates as a conservative visibility proxy.
        kp[:, coco, 2] = np.isfinite(kp[:, coco, 0]) & np.isfinite(kp[:, coco, 1])
    return kp


def derive_labels(activity: int, n: int, impact: np.ndarray, falling_frames: int) -> np.ndarray:
    if activity in ADL_ACTIVITIES:
        return np.zeros(n, dtype=np.int64)  # NORMAL
    if activity not in FALL_ACTIVITIES:
        raise ValueError(f"Unsupported activity {activity}")
    impact_idx = np.flatnonzero(impact > 0)
    if len(impact_idx) == 0:
        # Fall file without an impact marker: keep it as NORMAL rather than inventing a transition.
        return np.zeros(n, dtype=np.int64)
    first = int(impact_idx[0])
    y = np.zeros(n, dtype=np.int64)
    y[max(0, first - falling_frames):first] = 1  # FALLING
    y[first:] = 2  # FALLEN
    return y


def main(args):
    rows = []
    for f in Path(args.input).rglob("C*.csv"):
        m = re.search(r"C(\d+)S(\d+)A(\d+)T(\d+)", f.name.replace("_", ""))
        if not m:
            continue
        camera, subject, activity, trial = map(int, m.groups())
        df = pd.read_csv(f)
        label_col = "LABEL" if "LABEL" in df.columns else ("LLABEL" if "LLABEL" in df.columns else None)
        impact = df[label_col].to_numpy() if label_col else np.zeros(len(df), dtype=np.int64)
        y = derive_labels(activity, len(df), impact, args.falling_frames)
        # Store raw 17-joint pose; feature extraction remains centralized in src.features.
        pose = load_pose(df)
        for frame in range(len(df)):
            rows.append({
                "video_id": f"C{camera}S{subject}A{activity}T{trial}",
                "track_id": 1,
                "subject_id": subject,
                "camera_id": camera,
                "activity_id": activity,
                "frame": frame,
                "label": int(y[frame]),
                "keypoints": pose[frame].reshape(-1).tolist(),
            })
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out, orient="records", lines=True)
    print(f"Wrote {len(rows):,} frames to {out}")
    if rows:
        labels = pd.Series([r["label"] for r in rows]).value_counts().sort_index().to_dict()
        print(f"labels: {labels}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare full 11-activity UP-Fall skeleton data")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/processed/upfall_frames.jsonl")
    p.add_argument("--falling-frames", type=int, default=10)
    main(p.parse_args())
