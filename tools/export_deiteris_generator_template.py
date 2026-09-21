"""Dynamic Deiteris ONNX generator templates; Python/Torch are export-time only.

uv run --project PoC/tools python PoC/tools/export_deiteris_generator_template.py --out PoC/assets/app/deiteris

Uses pinned upstream modules unchanged on disk. Shape-only export substitutions
remove Python int/branch specialization. FP16 scalar opmath follows the verified
fixed Deiteris exporter. All voice weights use the existing Rust template format.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch.onnx import symbolic_helper, symbolic_opset9

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'PoC/assets/research-20260919/deiteris'
sys.path.insert(0, str(UPSTREAM / 'server'))
from voice_changer.RVC.inferencer.rvc_models.infer_pack import attentions
from voice_changer.RVC.inferencer.rvc_models.infer_pack.models import SynthesizerTrnMs768NSFsid


def patch_attention():
    def get_rel(self, rel, length):
        # Negative Pad is symmetric cropping for short sequences, equivalent to
        # upstream max/pad/slice; positive Pad handles long sequences.
        pad = length - (self.window_size + 1)
        return F.pad(rel, [0, 0, pad, pad, 0, 0])

    def rel_to_abs(self, x):
        b, h, n, _ = x.shape
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0]).reshape(b, h, n * 2 * n)
        x = F.pad(x, [0, n - 1, 0, 0, 0, 0])
        return x.reshape(b, h, n + 1, 2 * n - 1)[:, :, :n, n - 1:]

    def abs_to_rel(self, x):
        b, h, n, _ = x.shape
        x = F.pad(x, [0, n - 1, 0, 0, 0, 0, 0, 0]).reshape(b, h, n*n+n*(n-1))
        return F.pad(x, [n, 0, 0, 0, 0, 0]).reshape(b, h, n, 2*n)[:, :, :, 1:]

    attentions.MultiHeadAttention._get_relative_embeddings = get_rel
    attentions.MultiHeadAttention._relative_position_to_absolute_position = rel_to_abs
    attentions.MultiHeadAttention._absolute_position_to_relative_position = abs_to_rel


def scalar_half(graph, left, right, divide=False):
    scalar = symbolic_helper._maybe_get_scalar(right)
    if left.type().scalarType() == 'Half' and isinstance(scalar, torch.Tensor) and scalar.ndim == 0:
        number = float(scalar.item())
        factor = float(np.float32(1) / np.float32(number)) if divide else number
        constant = graph.op('Constant', value_t=torch.tensor(factor, dtype=torch.float32))
        return graph.op('Cast', graph.op('Mul', graph.op('Cast', left, to_i=1), constant), to_i=10)
    return None


def half_mul(graph, left, right):
    result = scalar_half(graph, left, right)
    return result if result is not None else symbolic_opset9.mul(graph, left, right)


def half_div(graph, left, right, *args):
    result = scalar_half(graph, left, right, True) if not args else None
    return result if result is not None else symbolic_opset9.div(graph, left, right, *args)


class Generator(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.model = net

    def forward(self, features, pitch, pitchf, rnd, sine_noise, n_res, n_head):
        net = self.model
        enc, dec = net.enc_p, net.dec
        flow_head = features.shape[1] - rnd.shape[2]
        dec_head = n_head.shape[1]
        head = flow_head + dec_head
        length = sine_noise.shape[1] // dec.upp
        length2 = n_res.shape[1]
        sid = torch.zeros(1, dtype=torch.int64, device=features.device)
        g = net.emb_g(sid).unsqueeze(-1)
        x = enc.emb_phone(features) + enc.emb_pitch(pitch)
        x = enc.lrelu(x * enc.sqrt_hidden_channels).transpose(1, -1)
        mask = torch.ones_like(x[:, :1, :])
        x = enc.encoder(x * mask, mask)[:, :, flow_head:]
        mask = mask[:, :, flow_head:]
        m, logs = torch.split(enc.proj(x) * mask, enc.out_channels, dim=1)
        z = net.flow((m + torch.exp(logs) * rnd * 0.66666) * mask, mask, g=g, reverse=True)
        x = z[:, :, dec_head:dec_head+length] * mask[:, :, dec_head:dec_head+length]
        f0 = pitchf[:, head:head+length]
        original_like, original_rand = torch.randn_like, torch.rand
        torch.randn_like = lambda *args, **kwargs: sine_noise
        # harmonic_num=0: upstream overwrites its only initial phase with zero.
        torch.rand = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
        try:
            har, _, _ = dec.m_source(f0, dec.upp)
        finally:
            torch.randn_like, torch.rand = original_like, original_rand
        har = F.interpolate(har.transpose(1, 2), size=length2 * dec.upp, mode='linear')
        x = F.interpolate(x, size=length2, mode='linear')
        x = dec.conv_pre(x) + dec.cond(g)
        for i, (ups, noise_conv) in enumerate(zip(dec.ups, dec.noise_convs)):
            x = ups(F.leaky_relu(x, dec.lrelu_slope)) + noise_conv(har)
            xs = dec.resblocks[i * dec.num_kernels](x)
            for j in range(1, dec.num_kernels):
                xs = xs + dec.resblocks[i * dec.num_kernels + j](x)
            x = xs / dec.num_kernels
        x = torch.tanh(dec.conv_post(F.leaky_relu(x)))
        return torch.clamp(x[0, 0], -1, 1).float()


def export(rate, destination):
    template = json.loads((ROOT / f'PoC/assets/app/templates/{rate}/template.json').read_text())
    config = list(template['config'])
    config[15] = 1
    torch.manual_seed(731)
    net = SynthesizerTrnMs768NSFsid(*config, is_half=True)
    del net.enc_q
    raw_keys = set(net.state_dict())
    # Distinct placeholder values prevent exporter deduplication of equal biases
    # and LayerNorm scales. Every real voice weight must remain replaceable.
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.normal_(0, 0.02)
    net = net.float().eval().cuda()
    net.remove_weight_norm()
    net = net.half()
    module = Generator(net).eval()
    upp = int(net.dec.upp)
    args = (torch.randn(1, 67, 768, device='cuda').half(),
            torch.ones(1, 67, device='cuda', dtype=torch.int64),
            torch.full((1, 67), 220., device='cuda').half(),
            torch.randn(1, 192, 41, device='cuda').half(),
            torch.randn(1, 17 * upp, 1, device='cuda').half(),
            torch.zeros(1, 18, device='cuda'), torch.zeros(1, 24, device='cuda'))
    folder = destination / 'templates' / rate
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / 'generator.onnx'
    with torch.inference_mode():
        torch.onnx.export(module, args, str(filename), opset_version=18, dynamo=False,
                          do_constant_folding=False,
                          input_names=['features', 'pitch', 'pitchf', 'rnd', 'sine_noise', 'n_res', 'n_head'],
                          output_names=['generated'],
                          dynamic_axes={'features': {1: 'p_len'}, 'pitch': {1: 'p_len'},
                                        'pitchf': {1: 'p_len'}, 'rnd': {2: 'flow_len'},
                                        'sine_noise': {1: 'sine_len'}, 'n_res': {1: 'ret2_len'},
                                        'n_head': {1: 'dec_head'}, 'generated': {0: 'audio_len'}})
    graph = onnx.load(filename)
    state = net.state_dict()
    entries, offset = [], 0
    for tensor in graph.graph.initializer:
        key = tensor.name.removeprefix('model.')
        if key not in state:
            continue
        shape = list(tensor.dims)
        size = int(np.prod(shape)) * 2
        assert tensor.data_type == onnx.TensorProto.FLOAT16
        tensor.ClearField('raw_data')
        tensor.data_location = onnx.TensorProto.EXTERNAL
        del tensor.external_data[:]
        for name, value in [('location', 'generator.weights'), ('offset', str(offset)), ('length', str(size))]:
            entry = tensor.external_data.add()
            entry.key, entry.value = name, value
        entry = {'key': key, 'shape': shape, 'offset': offset}
        if key + '_g' in raw_keys:
            entry['weight_norm'] = [key + '_g', key + '_v']
        if key == 'emb_g.weight':
            entry['rows'] = 1
        entries.append(entry)
        offset += size
    assert set(state) == {w['key'] for w in entries}, set(state) - {w['key'] for w in entries}
    template.update(weights=entries, weights_len=offset)
    template['model']['generator_kind'] = 'deiteris-onnx-v1'
    onnx.save(graph, filename)
    (folder / 'template.json').write_text(json.dumps(template, indent=2))
    print(json.dumps({'rate': rate, 'weights': len(entries), 'weight_bytes': offset}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--only', choices=['32k', '40k', '48k'])
    args = parser.parse_args()
    from deiteris_reference import _verify_upstream
    _verify_upstream(UPSTREAM)
    patch_attention()
    torch.onnx.register_custom_op_symbolic('aten::mul', half_mul, 18)
    torch.onnx.register_custom_op_symbolic('aten::div', half_div, 18)
    try:
        for rate in ([args.only] if args.only else ['32k', '40k', '48k']):
            export(rate, args.out.resolve())
    finally:
        torch.onnx.unregister_custom_op_symbolic('aten::mul', 18)
        torch.onnx.unregister_custom_op_symbolic('aten::div', 18)


if __name__ == '__main__':
    main()
