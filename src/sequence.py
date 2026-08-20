from __future__ import annotations
from collections import deque
import numpy as np
import torch

class TemporalBuffer:
    """Maintains one fixed-length feature history per BoT-SORT track ID."""
    def __init__(self,length=30,feature_dim=9): self.length=length; self.feature_dim=feature_dim; self.buffers={}
    def append(self,track_id,features):
        buf=self.buffers.setdefault(int(track_id),deque(maxlen=self.length)); buf.append(np.asarray(features,dtype=np.float32)); return len(buf)==self.length
    def tensor(self,track_id,mean=None,std=None):
        x=np.stack(self.buffers[int(track_id)])
        if mean is not None and std is not None: x=(x-np.asarray(mean))/np.maximum(np.asarray(std),1e-6)
        return torch.from_numpy(x).unsqueeze(0).float()
    def clear_missing(self,active_ids):
        active={int(x) for x in active_ids}
        for tid in list(self.buffers):
            if tid not in active: del self.buffers[tid]

def predict(model,buffer,track_id,mean=None,std=None,device="cpu"):
    x=buffer.tensor(track_id,mean,std).to(device)
    with torch.inference_mode(): probs=torch.softmax(model(x),dim=-1)[0]
    idx=int(torch.argmax(probs)); return ("NORMAL","FALLING","FALLEN")[idx],float(probs[idx]),probs.cpu().numpy()
