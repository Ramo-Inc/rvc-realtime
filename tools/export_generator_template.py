"""Export generator ONNX templates for the app's Rust-only voice model conversion.

One template per official training config (v2 32k, v1 40k used by v2 40k models, v2 48k). A template is the
trial 2 shape-agnostic generator graph with every voice model weight moved to external data, so a converter
(rvc-engine voice_model) only has to write one weights file at the offsets in template.json:

  <out>/templates/<name>/generator.onnx   graph; weights live in generator.weights next to it (external data)
  <out>/templates/<name>/template.json    config list as in the official .pth (speaker count null), model
                                          (model.json values without files), and per weight: key, offset, shape,
                                          weight_norm (g/v keys to combine), rows (emb_g: first row only, sid 0)
  <out>/reference/<name>/                 the same graph with generator.weights written by torch: the weights of
                                          --model for 40k, random weights for the others; --check runs it on ORT

Weights follow the official load (rtrvc.get_synthesizer): fp32, remove_weight_norm, then half.

Usage:
  uv run --project tools python tools/export_generator_template.py --out assets/app --model assets/app/voices/default_v2_40k.pth [--check]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import OPSET, RVC_ROOT, snr_db  # noqa: E402
from export_onnx_dynamic import patch_attention  # noqa: E402

TEMPLATES = {"32k": "configs/v2/32k.json", "40k": "configs/v1/40k.json", "48k": "configs/v2/48k.json"}


def config_list(path):
    """Same list as the official train/process_ckpt.py savee."""
    h = json.loads((RVC_ROOT / path).read_text(encoding="utf-8"))
    d, m = h["data"], h["model"]
    return [d["filter_length"] // 2 + 1, 32, m["inter_channels"], m["hidden_channels"], m["filter_channels"], m["n_heads"],
            m["n_layers"], m["kernel_size"], m["p_dropout"], m["resblock"], m["resblock_kernel_sizes"],
            m["resblock_dilation_sizes"], m["upsample_rates"], m["upsample_initial_channel"], m["upsample_kernel_sizes"],
            m["spk_embed_dim"], m["gin_channels"], d["sampling_rate"]]


def build(name, model_path):
    import torch

    from infer.module.models import SynthesizerTrnMs768NSFsid

    cfg = config_list(TEMPLATES[name])
    if model_path:
        cpt = torch.load(model_path, map_location="cpu")
        weight = cpt["weight"]
        assert cpt["config"][:15] + cpt["config"][16:] == cfg[:15] + cfg[16:], (cpt["config"], cfg)
        cfg[15] = weight["emb_g.weight"].shape[0]
    net_g = SynthesizerTrnMs768NSFsid(*cfg, is_half=False)
    del net_g.enc_q
    raw_keys = list(net_g.state_dict().keys())
    if model_path:
        net_g.load_state_dict(weight, strict=False)
    else:
        torch.manual_seed(0)
        with torch.no_grad():
            for p in net_g.parameters():
                p.normal_(0, 0.02)
    net_g = net_g.float().eval().to("cuda")
    net_g.remove_weight_norm()
    net_g = net_g.half()
    # sid is always 0: keep only row 0 so any speaker count fits the template
    net_g.emb_g.weight = torch.nn.Parameter(net_g.emb_g.weight[:1].clone())
    return net_g, cfg, raw_keys


def generator_module(net_g):
    """export_onnx_dynamic.py Generator, unchanged except that it closes over the given net_g."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    enc, dec = net_g.enc_p, net_g.dec
    upp = int(dec.upp)

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net_g = net_g

        def forward(self, feats, pitch, pitchf, rnd, sine_noise, n_res):
            p_len = feats.shape[1]
            flow_head = p_len - rnd.shape[2]
            head = flow_head + 24
            length = sine_noise.shape[1] // upp
            length2 = n_res.shape[1]
            sid = torch.zeros(1, dtype=torch.long, device=feats.device)
            g = net_g.emb_g(sid).unsqueeze(-1)
            x = enc.emb_phone(feats) + enc.emb_pitch(pitch)
            x = x * math.sqrt(enc.hidden_channels)
            x = enc.lrelu(x)
            x = torch.transpose(x, 1, -1)
            x_mask = torch.ones_like(x[:, :1, :])
            x = enc.encoder(x * x_mask, x_mask)
            x = x[:, :, flow_head:]
            x_mask = x_mask[:, :, flow_head:]
            stats = enc.proj(x) * x_mask
            m_p, logs_p = torch.split(stats, enc.out_channels, dim=1)
            z_p = (m_p + torch.exp(logs_p) * rnd.to(m_p.dtype) * 0.66666) * x_mask
            z = net_g.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, 24 : 24 + length]
            x_mask = x_mask[:, :, 24 : 24 + length]
            nsff0 = pitchf[:, head : head + length]
            queue = [sine_noise]
            orig_randn_like, orig_rand = torch.randn_like, torch.rand
            torch.randn_like = lambda t, *a, **k: queue.pop(0).to(t.dtype)
            torch.rand = lambda *size, device=None, **k: torch.zeros(*size, device=device)
            try:
                har_source, _, _ = dec.m_source(nsff0, dec.upp)
            finally:
                torch.randn_like, torch.rand = orig_randn_like, orig_rand
            har_source = har_source.transpose(1, 2)
            har_source = F.interpolate(har_source, size=length2 * dec.upp, mode="linear")
            x = F.interpolate(z * x_mask, size=length2, mode="linear")
            x = dec.conv_pre(x)
            x = x + dec.cond(g)
            for i, (ups, noise_convs) in enumerate(zip(dec.ups, dec.noise_convs)):
                x = F.leaky_relu(x, dec.lrelu_slope)
                x = ups(x)
                x = x + noise_convs(har_source)
                xs = None
                for j in range(dec.num_kernels):
                    r = dec.resblocks[i * dec.num_kernels + j](x)
                    xs = r if xs is None else xs + r
                x = xs / dec.num_kernels
            x = F.leaky_relu(x)
            x = dec.conv_post(x)
            return torch.tanh(x).float()

    return Generator().eval(), upp


def trace_inputs(net_g, upp, P=300, flow_len=250, R=40, R2=41):
    import torch

    dev = "cuda"
    feats = torch.randn(1, P, 768, device=dev).half()
    pitchf = (torch.rand(1, P, device=dev) * 200 + 100).float()
    pitch = torch.clamp(torch.round(1127 * torch.log(1 + pitchf / 700)), 1, 255).long()
    rnd = torch.randn(1, net_g.inter_channels, flow_len, device=dev).half()
    sine_noise = torch.randn(1, R * upp, 1, device=dev)
    n_res = torch.zeros(1, R2, device=dev)
    return feats, pitch, pitchf, rnd, sine_noise, n_res


def export_template(name, out, model_path, check):
    import onnx
    import torch

    net_g, cfg, raw_keys = build(name, model_path)
    gen, upp = generator_module(net_g)
    d = out / "templates" / name
    ref_dir = out / "reference" / name
    d.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    tmp = d / "generator.tmp.onnx"
    ins = trace_inputs(net_g, upp)
    names = ["feats", "pitch", "pitchf", "rnd", "sine_noise", "n_res"]
    with torch.no_grad():
        torch.onnx.export(gen, ins, str(tmp), input_names=names, output_names=["audio"], opset_version=OPSET,
                          do_constant_folding=False, dynamo=False,
                          dynamic_axes={"feats": {1: "p_len"}, "pitch": {1: "p_len"}, "pitchf": {1: "p_len"},
                                        "rnd": {2: "flow_len"}, "sine_noise": {1: "sine_len"}, "n_res": {1: "ret2_len"},
                                        "audio": {2: "audio_len"}})
    model = onnx.load(str(tmp))
    tmp.unlink()

    # every initializer must be a voice model weight (named by state_dict key) or a weight-independent constant
    sd = {k: v for k, v in net_g.state_dict().items()}
    weights, offset = [], 0
    with open(ref_dir / "generator.weights", "wb") as f:
        for t in model.graph.initializer:
            key = t.name[len("net_g."):] if t.name.startswith("net_g.") else None
            if key not in sd:
                continue
            a = sd[key].detach().cpu().numpy()
            assert a.dtype == np.float16, (key, a.dtype)
            data = a.tobytes()
            f.write(data)
            t.ClearField("raw_data")
            t.data_location = onnx.TensorProto.EXTERNAL
            del t.external_data[:]
            for k, v in (("location", "generator.weights"), ("offset", str(offset)), ("length", str(len(data)))):
                e = t.external_data.add()
                e.key, e.value = k, v
            entry = {"key": key, "offset": offset, "shape": list(a.shape)}
            if key + "_g" in raw_keys:
                entry["weight_norm"] = [key + "_g", key + "_v"]
            if key == "emb_g.weight":
                entry["rows"] = 1
            weights.append(entry)
            offset += len(data)
    inline = [(t.name, list(t.dims)) for t in model.graph.initializer if t.data_location != onnx.TensorProto.EXTERNAL]
    missing = sorted(set(sd) - {w["key"] for w in weights})
    assert not missing, f"weights not in the graph: {missing}"
    onnx.save(model, str(d / "generator.onnx"))
    onnx.save(model, str(ref_dir / "generator.onnx"))
    # model.json values (export_onnx_dynamic.py); f0_min / f0_max / cache_pitch length are constants of rtrvc.RVC
    template = {"config": cfg[:15] + [None] + cfg[16:],
                "model": {"model_sr": cfg[-1], "upp": upp, "half": True, "f0_min": 50.0, "f0_max": 1100.0,
                          "pitch_cache_len": 1024, "inter_channels": cfg[2]},
                "weights_len": offset, "weights": weights}
    (d / "template.json").write_text(json.dumps(template, indent=1), encoding="utf-8")
    print(json.dumps({"template": name, "weights": len(weights), "bytes": offset, "inline_constants": inline}))

    if check:
        import onnxruntime as ort
        from export_onnx import ort_session

        ort_session(ref_dir / "generator.onnx")
        feats, pitch, pitchf, rnd, sine_noise, n_res = ins
        P, R, R2 = feats.shape[1], sine_noise.shape[1] // upp, n_res.shape[1]
        with torch.no_grad():
            ref = gen(*ins).cpu().numpy()
        so = ort.SessionOptions()
        for k, v in {"p_len": P, "flow_len": rnd.shape[2], "sine_len": R * upp, "ret2_len": R2}.items():
            so.add_free_dimension_override_by_name(k, int(v))
        s = ort.InferenceSession(str(ref_dir / "generator.onnx"), so, providers=["CUDAExecutionProvider"])
        o = s.run(None, dict(zip(names, [t.cpu().numpy() for t in ins])))[0]
        print(json.dumps({"template": name, "generator_snr_db": snr_db(ref, o)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", help="official .pth whose weights fill the 40k weights file")
    ap.add_argument("--only", choices=list(TEMPLATES))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out, model = args.out.resolve(), args.model and str(Path(args.model).resolve())
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(RVC_ROOT))
    patch_attention()
    for name in TEMPLATES:
        if args.only and name != args.only:
            continue
        export_template(name, out, model if name == "40k" else None, args.check)


if __name__ == "__main__":
    main()
