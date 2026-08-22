from __future__ import annotations

import torch
from torch import nn

LABELS = ("NORMAL", "FALLING", "FALLEN")


class ContextLSTMv2(nn.Module):
    """Robust temporal classifier for the 9 engineered fall-context features.

    v2 keeps the project LSTM design but adds bidirectional temporal context and
    lightweight temporal attention. The output remains a 3-class classifier so
    it can replace the existing LSTM without changing the downstream state
    machine.
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 96,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        width = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(width, 48),
            nn.Tanh(),
            nn.Linear(48, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(self.input_norm(x))
        scores = self.attention(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(sequence * weights, dim=1)
        return self.head(context)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(**checkpoint.get("model_config", {}))
        model.load_state_dict(checkpoint["model_state"])
        return model.to(device).eval(), checkpoint
