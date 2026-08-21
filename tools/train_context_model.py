"""Train the context-aware temporal classifier on prepared NPZ sequences.

This recipe is tuned for the supplied UP-Fall-derived artifact: macro-F1
matters more than raw accuracy, while moderate class weighting avoids making
the classifier predict rare classes everywhere.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

class ContextGRU(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, classes=3, dropout=0.20):
        super().__init__()
        self.gru=nn.GRU(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout if num_layers>1 else 0.0)
        self.head=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,classes))
    def forward(self,x):
        z,_=self.gru(x); return self.head(z[:,-1])

def main(args):
    d=np.load(args.data); X,y=d['train_x'],d['train_y']; Xv,yv=d['val_x'],d['val_y']; Xt,yt=d['test_x'],d['test_y']
    mean=X.reshape(-1,X.shape[-1]).mean(0); std=np.maximum(X.reshape(-1,X.shape[-1]).std(0),1e-6)
    X=(X-mean)/std; Xv=(Xv-mean)/std; Xt=(Xt-mean)/std
    tr=DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(y)),128,shuffle=True)
    va=DataLoader(TensorDataset(torch.tensor(Xv,dtype=torch.float32),torch.tensor(yv)),256)
    te=DataLoader(TensorDataset(torch.tensor(Xt,dtype=torch.float32),torch.tensor(yt)),256)
    counts=np.bincount(y,minlength=3).astype(np.float32)
    weights=(counts.sum()/np.maximum(counts,1))**args.class_weight_power; weights/=weights.mean()
    weight=torch.tensor(weights,dtype=torch.float32,device=args.device)
    model=ContextGRU().to(args.device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=-1.; best_state=None; wait=0
    for epoch in range(args.epochs):
        model.train()
        for xb,yb in tr:
            xb,yb=xb.to(args.device),yb.to(args.device); opt.zero_grad(); loss=nn.functional.cross_entropy(model(xb),yb,weight=weight)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); yp=[]; yy=[]
        with torch.inference_mode():
            for xb,yb in va: yp.extend(model(xb.to(args.device)).argmax(1).cpu().numpy()); yy.extend(yb.numpy())
        score=f1_score(yy,yp,average='macro'); print(f'epoch={epoch+1:03d} val_macro_f1={score:.4f}')
        if score>best: best=score; wait=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=args.patience: break
    model.load_state_dict(best_state); model.eval(); yp=[]; yy=[]
    with torch.inference_mode():
        for xb,yb in te: yp.extend(model(xb.to(args.device)).argmax(1).cpu().numpy()); yy.extend(yb.numpy())
    p,r,f,_=precision_recall_fscore_support(yy,yp,labels=[0,1,2],zero_division=0)
    report={'accuracy':accuracy_score(yy,yp),'macro_f1':f1_score(yy,yp,average='macro'),'labels':['NORMAL','FALLING','FALLEN'],'precision':p.tolist(),'recall':r.tolist(),'f1':f.tolist(),'confusion_matrix':confusion_matrix(yy,yp,labels=[0,1,2]).tolist(),'class_counts':{'train':np.bincount(y,minlength=3).tolist(),'test':np.bincount(yt,minlength=3).tolist()}}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model_state':model.state_dict(),'model_config':{'input_size':9,'hidden_size':64,'num_layers':2,'architecture':'GRU'},'feature_mean':mean.tolist(),'feature_std':std.tolist(),'labels':report['labels'],'training':{'class_weight_power':args.class_weight_power}},out)
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',default='models/fall_gru_context_9f.pt'); p.add_argument('--report',default='outputs/development_metrics_upfall.json'); p.add_argument('--epochs',type=int,default=35); p.add_argument('--patience',type=int,default=8); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--class-weight-power',type=float,default=0.3); p.add_argument('--device',default='cpu'); main(p.parse_args())
