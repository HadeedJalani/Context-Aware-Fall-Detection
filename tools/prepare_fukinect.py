"""Prepare FUKinect 20-joint Kinect skeleton MAT files for temporal training.

The dataset adapter is deliberately separate from the live YOLO/pose pipeline.
It converts Kinect skeletons into the same nine temporal feature families used
by the project and creates 30-frame windows with subject-disjoint splits.

Labels:
    0 NORMAL   - everyday ADL activity or stable pre-fall context
    1 FALLING  - transition immediately before the estimated impact
    2 FALLEN   - post-impact state
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

ACTIVITIES = {"walking": 0, "bending": 0, "sitting": 0, "squatting": 0, "lying": 0, "falling": None}
FEATURES = {5: [0, 1, 2, 3, 8], 7: [0, 1, 2, 3, 4, 5, 8], 9: list(range(9))}


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
        angles = [joint_angle(q[4], q[12], q[13]), joint_angle(q[8], q[16], q[17]),
                  joint_angle(q[12], q[13], q[14]), joint_angle(q[16], q[17], q[18])]
        if t == 0:
            vel = acc = motion = depth_vel = 0.0
        else:
            p = s[t - 1]
            ps, ph = (p[4] + p[8]) / 2, (p[12] + p[16]) / 2
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
    if n < 30:
        return []
    if activity != "falling":
        # ADLs are explicitly NORMAL.  We subsample windows to avoid a single
        # long recording dominating the class distribution.
        return [(feat[e - 29:e + 1], 0) for e in range(29, n, 10)]

    impact = impact_index(s)
    out = []
    # Pre-impact context stays NORMAL until the fall transition begins.
    for e in range(max(29, impact - 24), max(29, impact - 11), 4):
        out.append((feat[e - 29:e + 1], 0))
    # Falling windows are centered on the transition and terminate at impact.
    for e in range(max(29, impact - 10), impact + 1, 2):
        out.append((feat[e - 29:e + 1], 1))
    # Post-impact windows represent the fallen state.
    for e in range(impact + 5, n, 4):
        out.append((feat[e - 29:e + 1], 2))
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
        activity = path.parent.parent.name.lower()
        try:
            subject = int(path.parent.name)
        except ValueError:
            continue
        if activity not in ACTIVITIES:
            continue
        skeleton = loadmat(path)["iskelet"].astype(np.float32).reshape(-1, 20, 3)
        for seq, label in windows(skeleton, activity):
            records.append((subject, activity, path.stem, seq, label))

    subjects = sorted({r[0] for r in records})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)
    n = len(subjects)
    train_s = set(subjects[:int(.70 * n)])
    val_s = set(subjects[int(.70 * n):int(.85 * n)])
    test_s = set(subjects[int(.85 * n):])

    result = {}
    for width, cols in FEATURES.items():
        split = {k: [] for k in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")}
        for subject, activity, stem, seq, label in records:
            name = "train" if subject in train_s else "val" if subject in val_s else "test"
            split[name + "_x"].append(seq[:, cols])
            split[name + "_y"].append(label)
        result[width] = {
            k: np.stack(v).astype(np.float32) if k.endswith("_x") else np.asarray(v, dtype=np.int64)
            for k, v in split.items() if v
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"{w}_{k}": v for w, d in result.items() for k, v in d.items()}
    payload.update({"train_subjects": np.asarray(sorted(train_s)), "val_subjects": np.asarray(sorted(val_s)), "test_subjects": np.asarray(sorted(test_s))})
    np.savez_compressed(out, **payload)
    print(f"Prepared {len(records)} windows from {len(subjects)} subjects -> {out}")
    print(f"Subject split: train={sorted(train_s)}, val={sorted(val_s)}, test={sorted(test_s)}")


if __name__ == "__main__":
    main()
