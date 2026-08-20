from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .dataset import SequenceDataset,feature_stats
from .model import FallLSTM
LABELS=("NORMAL","FALLING","FALLEN")

def metrics(y_true,y_pred,n=3):
    cm=np.zeros((n,n),dtype=np.int64)
    for a,b in zip(y_true,y_pred): cm[int(a),int(b)]+=1
    f1=[]; recall=[]
    for c in range(n):
        tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp; p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); recall.append(float(r)); f1.append(float(2*p*r/max(p+r,1e-12)))
    return {"accuracy":float(np.trace(cm)/max(cm.sum(),1)),"macro_f1":float(np.mean(f1)),"recall":recall,"confusion_matrix":cm.tolist()}

def run(a):
    device=torch.device(a.device if a.device=="cpu" or torch.cuda.is_available() else "cpu")
    tr,va,te=SequenceDataset(a.data,"train"),SequenceDataset(a.data,"val"),SequenceDataset(a.data,"test")
    mean,std=feature_stats(tr.x); model=FallLSTM(input_size=tr.x.shape[-1],hidden_size=a.hidden,num_layers=a.layers).to(device)
    counts=np.bincount(tr.y,minlength=3).astype(np.float32); weights=counts.sum()/np.maximum(counts,1); weights/=weights.mean()
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device)); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    best=-1.; wait=0; out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    mmean=torch.tensor(mean,device=device); mstd=torch.tensor(std,device=device)
    for epoch in range(1,a.epochs+1):
        model.train(); losses=[]
        for x,y in DataLoader(tr,batch_size=a.batch_size,shuffle=True):
            x,y=x.to(device),y.to(device); x=(x-mmean)/mstd; opt.zero_grad(set_to_none=True); loss=criterion(model(x),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(loss.item())
        model.eval(); yt=[]; yp=[]
        with torch.inference_mode():
            for x,y in DataLoader(va,batch_size=a.batch_size): yt.extend(y.numpy()); yp.extend(model((x.to(device)-mmean)/mstd).argmax(1).cpu().numpy())
        score=metrics(yt,yp)["macro_f1"]; print(f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_f1={score:.4f}")
        if score>best:
            best=score; wait=0; torch.save({"model_state":model.state_dict(),"model_config":{"input_size":tr.x.shape[-1],"hidden_size":a.hidden,"num_layers":a.layers},"feature_mean":mean.tolist(),"feature_std":std.tolist(),"labels":LABELS},out)
        else:
            wait+=1
            if wait>=a.patience: break
    model,ckpt=FallLSTM.from_checkpoint(out,device); mmean=torch.tensor(ckpt["feature_mean"],device=device); mstd=torch.tensor(ckpt["feature_std"],device=device); yt=[]; yp=[]
    with torch.inference_mode():
        for x,y in DataLoader(te,batch_size=a.batch_size): yt.extend(y.numpy()); yp.extend(model((x.to(device)-mmean)/mstd).argmax(1).cpu().numpy())
    report=metrics(yt,yp); report["labels"]=LABELS; Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--output",default="models/fall_lstm.pt"); p.add_argument("--report",default="outputs/test_metrics.json"); p.add_argument("--epochs",type=int,default=40); p.add_argument("--patience",type=int,default=7); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--hidden",type=int,default=128); p.add_argument("--layers",type=int,default=2); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--device",default="cuda"); run(p.parse_args())
