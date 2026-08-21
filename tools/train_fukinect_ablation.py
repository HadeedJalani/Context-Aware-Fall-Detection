"""Train the FUKinect 5/7/9-feature ablation using the project LSTM family."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LABELS = ("NORMAL", "FALLING", "FALLEN")


def metrics(y, p):
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(y, p): cm[int(a), int(b)] += 1
    f1, recall = [], []
    for c in range(3):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        precision = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        recall.append(r); f1.append(2 * precision * r / max(precision + r, 1e-9))
    return {"accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
            "macro_f1": float(np.mean(f1)), "recall": recall,
            "confusion_matrix": cm.tolist()}


def train(x, y, xv, yv, xt, yt, epochs=12, batch=512, hidden=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = x.reshape(-1, x.shape[-1]).mean(0).astype(np.float32)
    std = np.maximum(x.reshape(-1, x.shape[-1]).std(0), 1e-6).astype(np.float32)
    class M(nn.Module):
        def __init__(self, n):
            super().__init__(); self.norm = nn.LayerNorm(n)
            self.lstm = nn.LSTM(n, hidden, 1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32, 3))
        def forward(self, z): return self.head(self.lstm(self.norm(z))[0][:, -1])
    model = M(x.shape[-1]).to(device)
    counts = torch.bincount(torch.from_numpy(y), minlength=3).float()
    weights = counts.sum() / counts.clamp_min(1); weights /= weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch, shuffle=True)
    tx, ty = torch.from_numpy(xv).to(device), torch.from_numpy(yv).to(device)
    best, state, wait = -1, None, 0
    mu, sig = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device); xb = (xb - mu) / sig
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.inference_mode(): pv = model((tx - mu) / sig).argmax(1).cpu().numpy()
        score = metrics(yv, pv)["macro_f1"]
        if score > best:
            best, wait = score, 0; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: wait += 1
        if wait >= 3: break
    model.load_state_dict(state); model.eval()
    with torch.inference_mode():
        pred = model((torch.from_numpy(xt).to(device) - mu) / sig).argmax(1).cpu().numpy()
    report = metrics(yt, pred); report["best_validation_macro_f1"] = best
    return model, mean, std, report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--output", required=True)
    args = ap.parse_args(); data = np.load(args.data); out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for width in (5, 7, 9):
        model, mean, std, report = train(data[f"{width}_train_x"], data[f"{width}_train_y"], data[f"{width}_val_x"], data[f"{width}_val_y"], data[f"{width}_test_x"], data[f"{width}_test_y"])
        results[str(width)] = report
        torch.save({"model_state": model.state_dict(), "model_config": {"input_size": width, "hidden_size": 64, "num_layers": 1}, "feature_mean": mean.tolist(), "feature_std": std.tolist(), "labels": LABELS}, out.parent / f"fukinect_lstm_{width}f.pt")
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
