from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self,npz_path,split="train"):
        data=np.load(Path(npz_path)); self.x=data[f"{split}_x"].astype(np.float32); self.y=data[f"{split}_y"].astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return torch.from_numpy(self.x[i]),torch.tensor(self.y[i])

def feature_stats(x):
    mean=x.reshape(-1,x.shape[-1]).mean(axis=0); std=x.reshape(-1,x.shape[-1]).std(axis=0)
    return mean.astype(np.float32),np.maximum(std,1e-6).astype(np.float32)
