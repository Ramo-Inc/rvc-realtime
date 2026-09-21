"""Export normal TG RMVPE salience plus numeric mel coefficients; dev-time only.

Existing Deiteris generator templates/weights are not rewritten. The destination
must be new; the reference distribution and old product assets are read-only.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from tg_fast_reference import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--legacy-assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.reference.resolve(), args.out.resolve()
    if out.exists() or out.is_relative_to(root):
        parser.error('destination must be new and outside reference')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / 'server'))
    import torch
    import onnx
    from voice_changer.common.rmvpe.rmvpe import E2E, MelSpectrogram
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True)
    model = E2E(4, 1, (2, 2))
    checkpoint = root / 'server/pretrain/pitch_extractor/rmvpe.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model = model.eval().half().cuda()
    with torch.no_grad():
        torch.onnx.export(model, (torch.zeros(1, 128, 32, device='cuda', dtype=torch.float16),),
                          str(out / 'rmvpe-salience.onnx'), input_names=['mel'], output_names=['salience'],
                          dynamic_axes={'mel': {2: 'mel_frames'}, 'salience': {1: 'mel_frames'}},
                          opset_version=18, dynamo=False)
    onnx.checker.check_model(onnx.load(out / 'rmvpe-salience.onnx'))
    mel = MelSpectrogram(True, 128, 16000, 1024, 160, None, 30, 8000)
    mel.mel_basis.numpy().tofile(out / 'mel-basis.f32')
    torch.hann_window(1024).numpy().tofile(out / 'hann.f32')
    for name in ['contentvec.onnx', 'post.onnx']:
        shutil.copyfile(args.legacy_assets / name, out / name)
    files = ['rmvpe-salience.onnx', 'mel-basis.f32', 'hann.f32', 'contentvec.onnx', 'post.onnx']
    post = onnx.load(out / 'post.onnx')
    type_name = {onnx.TensorProto.FLOAT: 'float32', onnx.TensorProto.FLOAT16: 'float16',
                 onnx.TensorProto.INT64: 'int64', onnx.TensorProto.BOOL: 'bool'}
    post_ports = [f'{v.name}:{type_name[v.type.tensor_type.elem_type]}' for v in [*post.graph.input, *post.graph.output]]
    manifest = {'schema': 1, 'algorithm': 'tg-fast-v1', 'generator_kind': 'deiteris-onnx-v1',
                'reference': (root / 'server/version.txt').read_text().strip(),
                'source_sha256': sha256(root / 'server/voice_changer/common/rmvpe/rmvpe.py'),
                'weights_sha256': sha256(checkpoint), 'tf32': False,
                'ports': {'contentvec': ['audio:float16', 'units9:float16', 'unit12:float16', 'unit12s:float16'],
                          'rmvpe': ['mel:float16', 'salience:float16'], 'post': post_ports},
                'files': {name: sha256(out / name) for name in files}}
    (out / 'audio-contract.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
