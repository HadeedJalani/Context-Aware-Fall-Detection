from __future__ import annotations

from collections import deque
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort

LABELS = ("NORMAL", "FALLING", "FALLEN")


class ONNXTemporalClassifier:
    """Per-track 30-frame ONNX inference with the project's normalization metadata."""

    def __init__(self, model_path, metadata_path=None, providers=None, window=30):
        model_path = str(model_path)
        self.session = ort.InferenceSession(
            model_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.window = int(window)
        self.buffers: dict[int, deque] = {}
        self.mean = None
        self.std = None
        self.feature_columns = list(range(9))

        if metadata_path and Path(metadata_path).exists():
            meta = json.loads(Path(metadata_path).read_text())
            self.mean = np.asarray(meta.get("normalization_mean"), dtype=np.float32)
            self.std = np.asarray(meta.get("normalization_std"), dtype=np.float32)
            self.feature_columns = meta.get("feature_columns", self.feature_columns)

    def append(self, track_id, features):
        tid = int(track_id)
        buf = self.buffers.setdefault(tid, deque(maxlen=self.window))
        buf.append(np.asarray(features, dtype=np.float32))
        return len(buf) == self.window

    def predict(self, track_id):
        tid = int(track_id)
        if tid not in self.buffers or len(self.buffers[tid]) < self.window:
            return "NORMAL", 0.0, np.zeros(3, dtype=np.float32)

        x = np.stack(self.buffers[tid]).astype(np.float32)[None, ...]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / np.maximum(self.std, 1e-6)

        logits = self.session.run([self.output_name], {self.input_name: x})[0][0]
        logits = logits.astype(np.float32)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= max(probs.sum(), 1e-8)
        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx]), probs

    def clear_missing(self, active_ids):
        active = {int(x) for x in active_ids}
        for tid in list(self.buffers):
            if tid not in active:
                del self.buffers[tid]
