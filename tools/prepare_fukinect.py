"""Prepare FUKinect 20-joint Kinect skeleton MAT files for temporal training.

The utility keeps dataset-specific parsing separate from the live YOLO/pose
pipeline. It extracts the nine temporal features used by the FUKinect Phase B.2
experiment and creates video-disjoint temporal windows.

Usage:
    python tools/prepare_fukinect.py --input data/raw/fukinect/skeleton \
        --output data/processed/fukinect_sequences.npz
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

ACTIVITIES = {"walking": 0, "bending": 0, "sitting": 0, "squatting": 0,
              "lying": 0, "falling": None}
FEATURES = {
    5: [0, 1, 2, 3, 8],
    7: [0, 1, 2, 3, 4, 5, 8],
    9: list(range(9)),
}


def joint_angle(a, b, c):
    u, v = a - b, c - b
    den = np.linalg.norm(u) * np.linalg.norm(v)
    if den < 1e-8:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / den, -1, 1))))


def features(s):
    out = np.zeros((len(s), 9), dtype=np.float32)
    for t, q in enumerate(s):
        shoulder = (q[4] + q[8]) / 2
        hip = (q[12] + q[16]) / 2
        center = (shoulder + hip) / 2
        feet = (q[15] + q[19]) / 2
        scale = max(np.linalg.norm(shoulder - hip) + np.linalg.norm(hip - feet), 1e-3)
        torso = hip - shoulder
        angle = abs(math.degrees(math.atan2(torso[0], torso[1])))
        relpos = (center[1] - feet[1]) / scale
        width = q[:, 0].max() - q[:, 0].min()
        height = q[:, 1].max() - q[:, 1].min()
        shape = width / max(height, 1e-4)
        angles = [
            joint_angle(q[4], q[12], q[13]),
            joint_angle(q[8], q[16], q[17]),
            joint_angle(q[12], q[13], q[14]),
            joint_angle(q[16], q[17], q[18]),
        ]
        if t == 0:
            vel = acc = motion = depth_vel = 0.0
        else:
            p = s[t - 1]
            ps = (p[4] + p[8]) / 2
            ph = (p[12] + p[16]) / 2
            pc = (ps + ph) / 2
            pscale = max(np.linalg.norm(ps - ph) + np.linalg.norm(ph - (p[15] + p[19]) / 2), 1e-3)
            vel = (center[1] - pc[1]) / pscale * 30.0
            motion = np.linalg.norm(center - pc) / pscale * 30.0
            depth_vel = (center[2] - pc[2]) / pscale * 30.0
            acc = 0.0 if t < 2 else (vel - out[t - 1, 3]) * 30.0
        out[t] = [angle, relpos, shape, vel, acc, np.mean(angles), np.std(angles), motion, depth_vel]
    return out


def impact_index(s):
    shoulder = (s[:, 4, :2] + s[:, 8, :2]) / 2
    hip = (s[:, 12, :2] + s[:, 16, :2]) / 2
    v = hip - shoulder
    angle = np.degrees(np.arctan2(np.abs(v[:, 0]), np.abs(v[:, 1]) + 1e-8))
    smooth = gaussian_filter1d(angle, 1.2)
    lo, hi = max(10, int(.15 * len(s))), min(len(s) - 5, int(.9 * len(s)))
    return lo + int(np.argmax(smooth[lo:hi]))


def windows(s, activity):
    feat = features(s)
    n = len(feat)
    if activity != "falling":
        return [(feat[e - 29:e + 1], 0) for e in range(29, n, 10)]
    impact = impact_index(s)
    out = []
    for e in (impact - 15, impact - 5):
        if 29 <= e < n:
            out.append((feat[e - 29:e + 1], 0))
    for e in range(max(29, impact - 10), min(n, impact + 6), 3):
        out.append((feat[e - 29:e + 1], 1))
    for e in range(max(29, impact + 8), n, 3):
        out.append((feat[e - 29:e + 1], 2))
    if n >= 30 and impact + 8 < n:
        out.append((feat[-30:], 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.input)
    records = []
    for path in sorted(root.rglob("*.mat")):
        activity = path.parent.parent.name
        subject = int(path.parent.name)
        skeleton = loadmat(path)["iskelet"].astype(np.float32).reshape(-1, 20, 3)
        for seq, label in windows(skeleton, activity):
            records.append((subject, activity, path.stem, seq, label))
    videos = sorted({(r[0], r[1], r[2]) for r in records})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(videos)
    n = len(videos)
    train_v = set(videos[:int(.70 * n)])
    val_v = set(videos[int(.70 * n):int(.85 * n)])
    test_v = set(videos[int(.85 * n):])
    result = {}
    for width, cols in FEATURES.items():
        split = {k: [] for k in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")}
        for subject, activity, stem, seq, label in records:
            key = (subject, activity, stem)
            name = "train" if key in train_v else "val" if key in val_v else "test"
            split[name + "_x"].append(seq[:, cols])
            split[name + "_y"].append(label)
        result[width] = {
            k: np.stack(v).astype(np.float32) if k.endswith("_x") else np.asarray(v, dtype=np.int64)
            for k, v in split.items()
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{f"{w}_{k}": v for w, d in result.items() for k, v in d.items()})
    print(f"Prepared {len(records)} windows from {len(videos)} videos -> {out}")


if __name__ == "__main__":
    main()
