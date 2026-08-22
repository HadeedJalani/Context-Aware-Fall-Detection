"""Train the production-oriented v2 context LSTM on prepared FUKinect windows.

This script intentionally works from the prepared NPZ rather than touching the
raw dataset. It is therefore deterministic, easy to rerun, and avoids the
fragile all-in-one notebook pipeline.

Main changes versus v1:
- 9-feature LSTM is retained as the project input contract.
- Bidirectional temporal context + attention are used inside the LSTM.
- WeightedRandomSampler gives FALLING windows enough training exposure.
- Mild class-weighted cross entropy avoids the extreme gradients of inverse
  frequency weighting.
- Early stopping is based on validation macro-F1.
- The selected checkpoint is exported directly to ONNX with preprocessing
  statistics and feature metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.context_lstm_v2 import ContextLSTMv2, LABELS


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def metrics(y_true, y_pred, n=3):
    cm = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        cm[int(a), int(b)] += 1
    precision, recall, f1 = [], [], []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precision.append(float(p))
        recall.append(float(r))
        f1.append(float(2 * p * r / max(p + r, 1e-12)))
    normal_fpr = (cm[0, 1] + cm[0, 2]) / max(cm[0].sum(), 1)
    return {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_f1": float(np.mean(f1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normal_false_positive_rate": float(normal_fpr),
        "confusion_matrix": cm.tolist(),
    }


def load_width(npz, width=9):
    return (
        npz[f"{width}_train_x"], npz[f"{width}_train_y"],
        npz[f"{width}_val_x"], npz[f"{width}_val_y"],
        npz[f"{width}_test_x"], npz[f"{width}_test_y"],
    )


def normalize(train_x, val_x, test_x):
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(0).astype(np.float32)
    std = train_x.reshape(-1, train_x.shape[-1]).std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((train_x - mean) / std, (val_x - mean) / std,
            (test_x - mean) / std, mean, std)


def evaluate(model, loader, mean, std, device):
    model.eval()
    yt, yp = [], []
    mean_t = torch.as_tensor(mean, device=device)
    std_t = torch.as_tensor(std, device=device)
    with torch.inference_mode():
        for x, y in loader:
            x = (x.to(device) - mean_t) / std_t
            pred = model(x).argmax(1).cpu().numpy()
            yp.extend(pred.tolist())
            yt.extend(y.numpy().tolist())
    return metrics(yt, yp)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    data = np.load(args.data)
    train_x, train_y, val_x, val_y, test_x, test_y = load_width(data, 9)
    train_x, val_x, test_x, mean, std = normalize(train_x, val_x, test_x)

    train_ds = ArrayDataset(train_x, train_y)
    val_ds = ArrayDataset(val_x, val_y)
    test_ds = ArrayDataset(test_x, test_y)

    counts = np.bincount(train_y, minlength=3).astype(np.float64)
    class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    class_weights /= class_weights.mean()
    sample_weights = class_weights[train_y]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ContextLSTMv2(
        input_size=9, hidden_size=args.hidden, num_layers=2, dropout=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_state = None
    wait = 0
    history = []
    mean_t = torch.as_tensor(mean, device=device)
    std_t = torch.as_tensor(std, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = (x.to(device) - mean_t) / std_t
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(y)

        val = evaluate(model, val_loader, mean, std, device)
        scheduler.step(val["macro_f1"])
        row = {"epoch": epoch, "train_loss": running / len(train_ds), **{f"val_{k}": v for k, v in val.items() if k != "confusion_matrix"}}
        history.append(row)
        print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} val_macro_f1={val['macro_f1']:.4f} falling_recall={val['recall'][1]:.4f}")

        if val["macro_f1"] > best_score:
            best_score = val["macro_f1"]
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, mean, std, device)
    payload = {
        "model_state": model.state_dict(),
        "model_config": {"input_size": 9, "hidden_size": args.hidden, "num_layers": 2, "dropout": args.dropout},
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "feature_indices": list(range(9)),
        "window_size": int(train_x.shape[1]),
        "labels": LABELS,
        "validation_macro_f1": best_score,
        "test_metrics": test,
        "train_class_counts": counts.astype(int).tolist(),
    }
    torch.save(payload, out)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(test, indent=2))
    Path(args.history).write_text(json.dumps(history, indent=2))

    model.eval()
    dummy = torch.zeros(1, train_x.shape[1], 9, dtype=torch.float32, device=device)
    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(onnx_path), opset_version=17,
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
    )

    metadata = {
        "window_size": int(train_x.shape[1]),
        "feature_count": 9,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "labels": LABELS,
        "test_metrics": test,
        "validation_macro_f1": best_score,
    }
    Path(args.metadata).write_text(json.dumps(metadata, indent=2))
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="models/context_aware_fall_lstm_v2.pt")
    p.add_argument("--onnx", default="models/context_aware_fall_lstm_v2.onnx")
    p.add_argument("--report", default="outputs/fukinect_v2_metrics.json")
    p.add_argument("--history", default="outputs/fukinect_v2_history.json")
    p.add_argument("--metadata", default="outputs/fukinect_v2_metadata.json")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    run(p.parse_args())
