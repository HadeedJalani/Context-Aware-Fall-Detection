import torch
from torch import nn

class ContextLSTM(nn.Module):
    """Temporal classifier for the expanded context-aware feature vector."""
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, num_classes=5, dropout=.3):
        super().__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout)
        self.norm=nn.LayerNorm(hidden_size)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        y,_=self.lstm(x)
        y=self.norm(y[:,-1])
        return self.fc(self.dropout(y))
