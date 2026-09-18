"""Buffer dimensions of the official realtime engine for any startup configuration.

`OfficialDims` instantiates the unchanged official `RVCStreamEngine` once with the real model, then makes
`rtrvc.RVC` return that same loaded RVC so further engines (one per sample rate / block / crossfade / extra)
run the official `__init__` without reloading weights. The formant and F0-method dependent values come
from `export_onnx.dims_from_engine` (copied from `rtrvc.RVC.infer`).
"""
import importlib.util
import os
import sys

from export_onnx import MODEL_PTH, RVC_ROOT, dims_from_engine


class OfficialDims:
    def __init__(self):
        os.environ["RVC_CUDA_GRAPH"] = "0"
        sys.argv = [sys.argv[0]]
        path = RVC_ROOT / "RVCRealtimeVST" / "worker" / "rvc_worker.py"
        spec = importlib.util.spec_from_file_location("rvc_worker", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        first = self.engine(48000, 130, 80, 2000)
        self.rvc = first.rvc
        from infer import rtrvc

        rtrvc.RVC = lambda *a, **k: first.rvc

    def engine(self, sr, block_ms, crossfade_ms, extra_ms):
        return self.module.RVCStreamEngine({
            "rvc_root": str(RVC_ROOT), "sample_rate": sr, "block_ms": block_ms, "crossfade_ms": crossfade_ms,
            "extra_ms": extra_ms, "model_path": str(MODEL_PTH), "index_path": "",
        })

    def dims(self, sr, block_ms, crossfade_ms, extra_ms, formant, f0, engine=None):
        import torch

        e = engine or self.engine(sr, block_ms, crossfade_ms, extra_ms)
        d = dims_from_engine(e, {"device_sr": sr, "formant": formant, "f0": f0})
        d["feats_frames"] = int(self.rvc.model._get_feat_extract_output_lengths(torch.tensor(d["n16"])))
        return d
