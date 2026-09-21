"""Fixed-shape GPU normal-RMVPE seams; FFT itself is Rust/cuFFT, not ONNX STFT.

Dev-time only. Existing assets and TG source are read-only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--audio-assets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.out.resolve().is_relative_to(a.reference.resolve()):
        p.error('destination must be new and outside reference')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.reference.resolve() / 'server'))
    import numpy as np
    import torch
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    from voice_changer.common.rmvpe.rmvpe import RMVPE, E2E
    torch.set_num_threads(1)
    a.out.mkdir(parents=True)
    # Keep the salience model inside this immutable bundle, never select an older
    # folded graph from the product's asset directory.
    model = E2E(4, 1, (2, 2))
    model.load_state_dict(torch.load(a.reference/'server/pretrain/pitch_extractor/rmvpe.pt', map_location='cpu', weights_only=True), strict=True)
    model = model.eval().half().cuda()
    model_path = a.out/'rmvpe-salience.onnx'
    with torch.inference_mode():
        torch.onnx.export(model, (torch.zeros(1, 128, 32, dtype=torch.float16, device='cuda'),),
                          str(model_path), input_names=['mel'], output_names=['salience'],
                          dynamic_axes={'mel': {2: 'mel_frames'}, 'salience': {1: 'mel_frames'}},
                          opset_version=18, dynamo=False, do_constant_folding=False)
    onnx.checker.check_model(onnx.load(model_path))

    def save(path, nodes, inputs, outputs, constants):
        graph = h.make_graph(nodes, path.stem, inputs, outputs,
                             [nh.from_array(v, k) for k, v in constants.items()])
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 18)], ir_version=9)
        onnx.checker.check_model(model)
        onnx.save(model, path)

    class Decode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ref = RMVPE.__new__(RMVPE)
            self.ref.idx = torch.arange(360)[None, None, :]
            self.ref.idx_cents = self.ref.idx * 20 + 1997.3794084376191

        def forward(self, salience, trim_index):
            return self.ref.decode(salience[:, trim_index], .05)

    # One graph family handles all startup lengths; Rust fixes these dimensions.
    for root in [a.out]:
        save(root/'frame.onnx', [h.make_node('Cast', ['audio'], ['single'], to=T.FLOAT),
             h.make_node('Gather', ['single', 'index'], ['framed'], axis=0),
             h.make_node('Mul', ['framed', 'hann'], ['windowed'])],
             [h.make_tensor_value_info('audio', T.FLOAT16, ['audio_len']),
              h.make_tensor_value_info('index', T.INT64, ['real_frames', 1024])],
             [h.make_tensor_value_info('windowed', T.FLOAT, ['real_frames', 1024])],
             {'hann': np.fromfile(a.audio_assets/'hann.f32', dtype=np.float32)})
        save(root/'mel.onnx', [
            h.make_node('Gather', ['spectrum', 'real_index'], ['real'], axis=2),
            h.make_node('Gather', ['spectrum', 'imag_index'], ['imag'], axis=2),
            h.make_node('Mul', ['real', 'real'], ['r2']), h.make_node('Mul', ['imag', 'imag'], ['i2']),
            h.make_node('Add', ['r2', 'i2'], ['power']), h.make_node('Sqrt', ['power'], ['mag']),
            h.make_node('Transpose', ['mag'], ['mag_t'], perm=[1, 0]),
            h.make_node('MatMul', ['basis', 'mag_t'], ['mel32']),
            h.make_node('Cast', ['mel32'], ['mel16'], to=T.FLOAT16),
            h.make_node('Max', ['mel16', 'floor'], ['clamped']),
            h.make_node('Log', ['clamped'], ['logmel']),
            h.make_node('Gather', ['logmel', 'pad_index'], ['padded'], axis=1),
            h.make_node('Unsqueeze', ['padded', 'batch_axis'], ['mel'])],
            [h.make_tensor_value_info('spectrum', T.FLOAT, ['real_frames', 513, 2]),
             h.make_tensor_value_info('pad_index', T.INT64, ['mel_frames'])],
            [h.make_tensor_value_info('mel', T.FLOAT16, [1, 128, 'mel_frames'])],
            {'real_index': np.array(0, np.int64), 'imag_index': np.array(1, np.int64),
             'basis': np.fromfile(a.audio_assets/'mel-basis.f32', dtype=np.float32).reshape(128, 513),
             'floor': np.array(1e-5, np.float16),
             'batch_axis': np.array([0], np.int64)})
        with torch.inference_mode():
            torch.onnx.export(Decode(), (torch.zeros(1, 32, 360, dtype=torch.float16), torch.arange(20)),
                              str(root/'decode.onnx'), input_names=['salience', 'trim_index'], output_names=['f0'],
                              dynamic_axes={'salience': {1: 'mel_frames'}, 'trim_index': {0: 'real_frames'}, 'f0': {1: 'real_frames'}},
                              opset_version=18, dynamo=False)
        onnx.checker.check_model(onnx.load(root/'decode.onnx'))
        files = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                 for name in ['frame.onnx', 'mel.onnx', 'decode.onnx']}
        files['rmvpe-salience.onnx'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        (root/'contract.json').write_text(json.dumps(dict(schema=2, algorithm='tg-cufft-rmvpe-v2',
            model_export='unfused-conv-bn', files=files), indent=2))
        print('exported dynamic normal RMVPE GPU seams', flush=True)


if __name__ == '__main__':
    main()
