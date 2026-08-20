"""Train the context-aware temporal classifier on a prepared NPZ sequence set.

The script intentionally optimizes macro-F1 rather than raw accuracy because
FALLING is the safety-critical transition and is usually the minority class.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

class BiGRUAttention(nn.Module):
    def __init__(self, input_size=9, hidden=48, classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, 1, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, 32),
            nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, classes)
        )
    def forward(self, x):
        z, _ = self.gru(x)
        a = torch.softmax(self.attention(z).squeeze(-1), dim=1).unsqueeze(-1)
        return self.head((z * a).sum(dim=1))

class FocalLoss(nn.Module):
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, reduction="none", label_smoothing=0.01)
        return (((1 - torch.exp(-ce)) ** 1.5) * ce).mean()

def main(args):
    d = np.load(args.data)
    X, y = d["train_x"], d["train_y"]
    Xv, yv = d["val_x"], d["val_y"]
    Xt, yt = d["test_x"], d["test_y"]
    mean = X.reshape(-1, X.shape[-1]).mean(0); std = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    X, Xv, Xt = (X-mean)/std, (Xv-mean)/std, (Xt-mean)/std
    counts = np.bincount(y, minlength=3); sample_w = 1/np.maximum(counts, 1)
    sampler = WeightedRandomSampler(torch.tensor(sample_w[y], dtype=torch.double), len(y), replacement=True)
    tr = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y)), 128, sampler=sampler)
    va = DataLoader(TensorDataset(torch.tensor(Xv, dtype=torch.float32), torch.tensor(yv)), 256)
    te = DataLoader(TensorDataset(torch.tensor(Xt, dtype=torch.float32), torch.tensor(yt)), 256)
    model = BiGRUAttention().to(args.device); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = FocalLoss(); best = -1; best_state = None
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(args.device), yb.to(args.device); opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); yp=[]; yy=[]
        with torch.inference_mode():
            for xb,yb in va: yp.extend(model(xb.to(args.device)).argmax(1).cpu().numpy()); yy.extend(yb.numpy())
        score=f1_score(yy,yp,average="macro")
        if score>best: best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"epoch={epoch+1:03d} val_macro_f1={score:.4f}")
    model.load_state_dict(best_state); model.eval(); yp=[]; yy=[]
    with torch.inference_mode():
        for xb,yb in te: yp.extend(model(xb.to(args.device)).argmax(1).cpu().numpy()); yy.extend(yb.numpy())
    p,r,f,_=precision_recall_fscore_support(yy,yp,labels=[0,1,2],zero_division=0)
    report={"accuracy":accuracy_score(yy,yp),"macro_f1":f1_score(yy,yp,average="macro"),"labels":["NORMAL","FALLING","FALLEN"],"precision":p.tolist(),"recall":r.tolist(),"f1":f.tolist(),"confusion_matrix":confusion_matrix(yy,yp,labels=[0,1,2]).tolist()}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"model_config":{"input_size":9,"hidden_size":48,"num_layers":1,"architecture":"BiGRU-Attention"},"feature_mean":mean.tolist(),"feature_std":std.tolist(),"labels":report["labels"]},out)
    Path(args.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--output",default="models/fall_lstm_attention.pt"); p.add_argument("--report",default="outputs/development_metrics_upfall.json"); p.add_argument("--epochs",type=int,default=20); p.add_argument("--device",default="cpu"); main(p.parse_args())
