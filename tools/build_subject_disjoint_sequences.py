from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import extract_features

FEATURES = ["body_angle","body_height","body_shape","vertical_velocity","vertical_acceleration","joint_angle_mean","joint_angle_variability","immobility","pose_confidence"]


def main(args):
    records = [json.loads(x) for x in Path(args.frames).read_text().splitlines() if x.strip()]
    groups = defaultdict(list)
    for r in records: groups[(r["subject_id"], r["video_id"])].append(r)
    sequences = []
    for (subject, video), rs in groups.items():
        rs.sort(key=lambda r: r["frame"])
        poses = [np.asarray(r["keypoints"], np.float32).reshape(17,3) for r in rs]
        feats=[]
        for i,kp in enumerate(poses):
            feats.append(extract_features(kp, poses[i-1] if i else None, poses[i-2] if i>1 else None))
        feats=np.asarray(feats,np.float32)
        labels=np.asarray([r["label"] for r in rs],np.int64)
        for end in range(args.length-1,len(rs),args.stride):
            sequences.append((subject, video, feats[end-args.length+1:end+1], int(labels[end])))
    subjects=sorted({s for s,_,_,_ in sequences})
    if len(subjects)<3: raise RuntimeError("Need at least 3 subjects for subject-disjoint train/validation/test")
    test_subjects=set(args.test_subjects or [subjects[-1]])
    val_subjects=set(args.val_subjects or [subjects[-2]])
    train_subjects=set(subjects)-test_subjects-val_subjects
    if not train_subjects: raise RuntimeError("No training subjects remain")
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    payload={"features":FEATURES,"length":args.length,"stride":args.stride,"train_subjects":sorted(train_subjects),"val_subjects":sorted(val_subjects),"test_subjects":sorted(test_subjects)}
    for split,allowed in (("train",train_subjects),("val",val_subjects),("test",test_subjects)):
        xs=[x for s,_,x,y in sequences if s in allowed]; ys=[y for s,_,x,y in sequences if s in allowed]
        payload[split+"_x"]=np.asarray(xs,np.float32).tolist(); payload[split+"_y"]=np.asarray(ys,np.int64).tolist()
    # JSON keeps the artifact inspectable; convert to NPZ for training.
    (out/"split_manifest.json").write_text(json.dumps({k:v for k,v in payload.items() if not k.endswith("_x") and not k.endswith("_y")},indent=2))
    np.savez_compressed(out/"sequences.npz", **{k:np.asarray(v) for k,v in payload.items() if k.endswith("_x") or k.endswith("_y")})
    print(f"subjects={subjects}; train={sorted(train_subjects)} val={sorted(val_subjects)} test={sorted(test_subjects)}")
    for split in ("train","val","test"): print(split, len(payload[split+"_y"]))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--frames",required=True); p.add_argument("--output",default="data/processed/subject_disjoint"); p.add_argument("--length",type=int,default=30); p.add_argument("--stride",type=int,default=3); p.add_argument("--test-subjects",type=int,nargs="*"); p.add_argument("--val-subjects",type=int,nargs="*"); main(p.parse_args())
