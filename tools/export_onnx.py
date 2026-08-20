from __future__ import annotations
import argparse
from pathlib import Path
import torch
from src.model import FallLSTM

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=model
    def forward(self,x): return self.model(x)

def main(a):
    model, ckpt=FallLSTM.from_checkpoint(a.checkpoint,"cpu")
    wrapper=Wrapper(model).eval(); dummy=torch.randn(1,a.sequence_length,len(ckpt["feature_mean"]))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(wrapper,dummy,a.output,input_names=["features"],output_names=["logits"],dynamic_axes={"features":{0:"batch",1:"time"},"logits":{0:"batch"}},opset_version=17)
    print(f"exported {a.output}")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--output",default="models/fall_lstm.onnx"); p.add_argument("--sequence-length",type=int,default=30); main(p.parse_args())
