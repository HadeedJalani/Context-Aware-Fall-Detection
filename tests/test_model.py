import torch
from src.model import FallLSTM

def test_lstm_shape():
    m=FallLSTM(input_size=9)
    assert m(torch.randn(4,30,9)).shape==(4,3)
