from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

LABELS={"NORMAL":0,"FALLING":1,"FALLEN":2}
FEATURES=["body_angle","body_height","body_shape","vertical_velocity","vertical_acceleration","joint_angle_mean","joint_angle_variability","immobility","pose_confidence"]

def make(args):
    df=pd.read_csv(args.csv).sort_values(["video_id","track_id","frame"])
    required=["video_id","track_id","frame","label",*FEATURES]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    X=[]; y=[]
    for (_,tid),g in df.groupby(["video_id","track_id"]):
        values=g[FEATURES].to_numpy(np.float32); labels=g.label.map(LABELS).to_numpy()
        for end in range(args.length-1,len(g)):
            start=end-args.length+1; X.append(values[start:end+1]); y.append(labels[end])
    X=np.asarray(X,np.float32); y=np.asarray(y,np.int64)
    if len(X)<3: raise ValueError("Not enough sequences")
    rng=np.random.default_rng(args.seed); idx=rng.permutation(len(X)); n=len(X); a=int(.70*n); b=int(.85*n)
    np.savez_compressed(args.output,train_x=X[idx[:a]],train_y=y[idx[:a]],val_x=X[idx[a:b]],val_y=y[idx[a:b]],test_x=X[idx[b:]],test_y=y[idx[b:]])
    print(f"saved {len(X)} sequences -> {args.output}")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--csv",required=True); p.add_argument("--output",required=True); p.add_argument("--length",type=int,default=30); p.add_argument("--seed",type=int,default=42); make(p.parse_args())
