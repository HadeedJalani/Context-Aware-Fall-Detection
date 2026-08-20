from __future__ import annotations
import torch
from torch import nn

LABELS = ("NORMAL", "FALLING", "FALLEN")

class FallLSTM(nn.Module):
    """Temporal classifier for the context-aware 9-feature sequence."""
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, num_classes=3, dropout=0.25):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size//2, num_classes))
    def forward(self, x):
        out, _ = self.lstm(self.input_norm(x))
        return self.head(out[:, -1, :])
    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ckpt=torch.load(path,map_location=device,weights_only=False)
        model=cls(**ckpt.get("model_config",{})); model.load_state_dict(ckpt["model_state"]); model.to(device).eval()
        return model,ckpt
