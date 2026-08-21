from __future__ import annotations
import torch
from torch import nn

LABELS = ("NORMAL", "FALLING", "FALLEN")

class ContextGRU(nn.Module):
    """Development classifier used for the UP-Fall experiment."""
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, num_classes=3, dropout=0.20):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1])

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint.get("model_config", {})
        config.pop("architecture", None)
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state"])
        return model.to(device).eval(), checkpoint
