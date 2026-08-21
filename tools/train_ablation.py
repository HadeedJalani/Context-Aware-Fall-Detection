from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from src.model import FallLSTM

FEATURE_SETS={"5f":[0,1,2,3,8],"7f":[0,1,2,3,4,5,8],"9f":list(range(9))}
LABELS=("NORMAL","FALLING","FALLEN")

def score(y,p):
    cm=np.zeros((3,3),int)
    for a,b in zip(y,p): cm[a,b]+=1
    out=[]
    for c in range(3):
        tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c].sum()-tp; pr=tp/max(tp+fp,1); re=tp/max(tp+fn,1); out.append((pr,re,2*pr*re/max(pr+re,1e-9)))
    return {"accuracy":float(np.trace(cm)/max(cm.sum(),1)),"macro_f1":float(np.mean([x[2] for x in out])),"precision":[x[0] for x in out],"recall":[x[1] for x in out],"f1":[x[2] for x in out],"confusion_matrix":cm.tolist()}

def run(a):
    d=np.load(a.data); device=torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu"); results={}
    for name,cols in FEATURE_SETS.items():
        X=d["train_x"][:,:,cols].astype(np.float32); y=d["train_y"].astype(np.int64); V=d["val_x"][:,:,cols].astype(np.float32); vy=d["val_y"].astype(np.int64); T=d["test_x"][:,:,cols].astype(np.float32); ty=d["test_y"].astype(np.int64)
        mean=X.reshape(-1,len(cols)).mean(0); std=np.maximum(X.reshape(-1,len(cols)).std(0),1e-6); X=(X-mean)/std; V=(V-mean)/std; T=(T-mean)/std
        counts=np.bincount(y,minlength=3); cw=np.minimum(counts.sum()/np.maximum(counts,1),8.0); sw=torch.tensor(cw[y],dtype=torch.double)
        loader=DataLoader(TensorDataset(torch.tensor(X),torch.tensor(y)),batch_size=a.batch_size,sampler=WeightedRandomSampler(sw,len(sw),replacement=True))
        model=FallLSTM(input_size=len(cols),hidden_size=a.hidden,num_layers=2).to(device); loss=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32,device=device),label_smoothing=.03); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); best=(-1,None)
        for _ in range(a.epochs):
            model.train()
            for xb,yb in loader:
                xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); z=model(xb); l=loss(z,yb); l.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            model.eval(); pred=[]
            with torch.no_grad():
                for xb,_ in DataLoader(TensorDataset(torch.tensor(V),torch.tensor(vy)),batch_size=256): pred.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            s=score(vy,np.asarray(pred))["macro_f1"]
            if s>best[0]: best=(s,{k:v.cpu().clone() for k,v in model.state_dict().items()})
        model.load_state_dict(best[1]); model.eval(); pred=[]
        with torch.no_grad():
            for xb,_ in DataLoader(TensorDataset(torch.tensor(T),torch.tensor(ty)),batch_size=256): pred.extend(model(xb.to(device)).argmax(1).cpu().numpy())
        results[name]=score(ty,np.asarray(pred)); results[name]["feature_count"]=len(cols); results[name]["feature_mean"]=mean.tolist(); results[name]["feature_std"]=std.tolist()
        print(name,results[name]["accuracy"],results[name]["macro_f1"],results[name]["recall"])
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(results,indent=2)); print(f"saved {out}")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--output",default="outputs/ablation_results.json"); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--hidden",type=int,default=128); p.add_argument("--lr",type=float,default=8e-4); p.add_argument("--cpu",action="store_true"); run(p.parse_args())
