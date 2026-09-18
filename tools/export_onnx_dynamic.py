"""Export shape-agnostic ONNX models once per voice model (trial 02). Lengths are named dimensions that the
Rust engine fixes at session creation (ORT free dimension overrides), so block / crossfade / extra /
formant / sample rate become startup parameters instead of export-time constants.

Named dimensions:
  contentvec.onnx  audio [1, n16]            -> feats [1, feats_frames, 768]
  rmvpe.onnx       mag   [1, 513, f0_frames] -> hidden [1, f0_frames, 360]
  fcpe.onnx        mag   [1, 513, f0_frames] -> f0 [1, f0_frames, 1]
  generator.onnx   feats [1, p_len, 768], pitch/pitchf [1, p_len], rnd [1, 192, flow_len],
                   sine_noise [1, sine_len, 1] (sine_len = return_length * upp), n_res [1, ret2_len]
                   -> audio [1, 1, ret2_len * upp]
The generator derives flow_head = p_len - flow_len, skip_head = flow_head + 24 (extra >= 500 ms keeps
skip_head >= 50), return_length = sine_len / upp and return_length2 = ret2_len from input shapes.

Official code that turns lengths into Python ints (and would bake them into the trace) is replaced during
export only by the same arithmetic on traced sizes: SynthesizerTrnMs256NSFsid.infer / TextEncoder skip
slicing, GeneratorNSF n_res branches (always interpolate; same-size linear resize is the identity),
MultiHeadAttention relative-position helpers (max()/int() on the length).

Usage:
  uv run --project tools python tools/export_onnx_dynamic.py [--check]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import POC, export, fcpe_mag, ort_session, rmvpe_mag, snr_db  # noqa: E402
from official_dims import OfficialDims  # noqa: E402

OUT = POC / "assets" / "onnx-dynamic"

# (sample rate, block, crossfade, extra, formant): trace shapes first, then check sets covering range ends
TRACE = (48000, 130, 80, 2000, -0.2)
CHECKS = [
    (48000, 130, 80, 2000, -0.2),
    (48000, 60, 80, 1000, -0.2),
    (44100, 20, 10, 500, 12.0),
    (96000, 1000, 100, 3000, -12.0),
    (88200, 37, 41, 777, 0.0),
    (48000, 300, 45, 1500, 3.33),   # RMVPE 64 frames
    (44100, 600, 100, 2500, -5.5),  # RMVPE 96 frames
]


def patch_attention():
    import torch.nn.functional as F
    from infer.module import attentions

    mha = attentions.MultiHeadAttention

    def get_rel(self, rel, length):
        # official: max(length - (w + 1), 0) / max((w + 1) - length, 0); lengths here are always > w + 1
        pad = length - (self.window_size + 1)
        padded = F.pad(rel, [0, 0, pad, pad, 0, 0])
        return padded[:, 0 : 2 * length - 1]

    def rel_to_abs(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, [0, length - 1, 0, 0, 0, 0])
        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]

    def abs_to_rel(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length * length + length * (length - 1)])
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

    mha._get_relative_embeddings = get_rel
    mha._relative_position_to_absolute_position = rel_to_abs
    mha._absolute_position_to_relative_position = abs_to_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    od = OfficialDims()
    rvc = od.rvc
    dev = torch.device(rvc.device)
    half = bool(rvc.is_half)
    ftype = torch.float16 if half else torch.float32
    net_g = rvc.net_g
    upp = int(net_g.dec.upp)
    assert net_g.dec.m_source.l_sin_gen.dim == 1, "rand_ini is only all-zero when harmonic_num == 0"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    d0 = od.dims(TRACE[0], TRACE[1], TRACE[2], TRACE[3], TRACE[4], "rmvpe")

    # ---- ContentVec
    class ContentVec(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, audio):
            return self.m(input_values=audio, attention_mask=None, output_hidden_states=False,
                          return_dict=True).last_hidden_state

    cv = ContentVec(rvc.model).eval()
    # SDPA without `scale` exports 1/sqrt(head_dim) as Shape->Cast(fp16)->Sqrt, which ORT cannot constant-fold
    # (no fp16 CPU Sqrt kernel) and which blocks its attention fusions; pass the same value as a constant.
    head_dim = rvc.model.config.hidden_size // rvc.model.config.num_attention_heads
    orig_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = lambda *a, **k: orig_sdpa(*a, **{**k, "scale": k.get("scale") or head_dim ** -0.5})
    # 3-D nn.Linear exports as MatMul+Add; the fixed-shape export (after onnxslim) runs 2-D Gemm. Export the same
    # 2-D Gemm on a flattened view so ORT picks the same fp16 kernels.
    orig_linear = nn.Linear.forward
    nn.Linear.forward = lambda self, x: (F.linear(x.reshape(-1, self.in_features), self.weight, self.bias)
                                         .reshape(x.shape[0], x.shape[1], self.out_features) if x.dim() == 3 else orig_linear(self, x))
    try:
        export(cv, ((torch.randn(1, d0["n16"], device=dev) * 0.1).to(ftype),), OUT / "contentvec.onnx", ["audio"], ["feats"],
               dynamic_axes={"audio": {1: "n16"}, "feats": {1: "feats_frames"}})
    finally:
        F.scaled_dot_product_attention = orig_sdpa
        nn.Linear.forward = orig_linear

    # ---- RMVPE
    from infer.rmvpe import RMVPE

    rm = RMVPE(str(POC / "assets" / "official" / "rvc" / "assets" / "rmvpe" / "rmvpe.pt"), is_half=half, device=str(dev))

    class Rmvpe(nn.Module):
        def __init__(self, rm):
            super().__init__()
            self.register_buffer("basis", rm.mel_extractor.mel_basis.clone())
            self.model = rm.model
            self.half_ = rm.is_half

        def forward(self, mag):
            mel = torch.matmul(self.basis, mag)
            if self.half_:
                mel = mel.half()
            x = torch.log(torch.clamp(mel, min=1e-5))
            n = mag.shape[-1]
            x = F.pad(x, (0, 32 * ((n - 1) // 32 + 1) - n), mode="constant")
            return self.model(x)[:, :n].float()

    rmvpe_m = Rmvpe(rm).eval()
    mag = rmvpe_mag(torch.randn(1, d0["f0_extractor_frame"], device=dev) * 0.1)
    export(rmvpe_m, (mag,), OUT / "rmvpe.onnx", ["mag"], ["hidden"],
           dynamic_axes={"mag": {2: "f0_frames"}, "hidden": {1: "f0_frames"}})

    # ---- FCPE (fp32, like the official)
    from infer.fcpe import FCPEInfer

    fc = FCPEInfer(dev)

    class Fcpe(nn.Module):
        def __init__(self, fc):
            super().__init__()
            self.fc = fc
            self.register_buffer("basis", fc.infer_model.wav2mel.mel_extractor.mel_basis.clone())
            self.model = fc.infer_model.model

        def forward(self, mag):
            mel = torch.log(torch.clamp(torch.matmul(self.basis, mag), min=1e-5)).transpose(-1, -2)
            return self.fc._graphable_model_infer(mel, "local_argmax", 0.006).float()

    fcpe_m = Fcpe(fc).eval()
    d0f = od.dims(TRACE[0], TRACE[1], TRACE[2], TRACE[3], TRACE[4], "fcpe")
    mag = fcpe_mag(torch.randn(1, d0f["f0_extractor_frame"], device=dev) * 0.1)
    export(fcpe_m, (mag,), OUT / "fcpe.onnx", ["mag"], ["f0"],
           dynamic_axes={"mag": {2: "f0_frames"}, "f0": {1: "f0_frames"}})

    # ---- Generator: lengths from input shapes
    patch_attention()
    enc, dec = net_g.enc_p, net_g.dec
    queue = []

    def with_noise(fn):
        orig_randn_like, orig_rand = torch.randn_like, torch.rand
        torch.randn_like = lambda t, *a, **k: queue.pop(0).to(t.dtype)
        torch.rand = lambda *size, device=None, **k: torch.zeros(*size, device=device)
        try:
            return fn()
        finally:
            torch.randn_like, torch.rand = orig_randn_like, orig_rand

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
            # TextEncoder.forward with lengths == p_len (sequence_mask is all ones) and skip_head = flow_head
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
            # SynthesizerTrnMs256NSFsid.infer (dec_head = head - flow_head = 24)
            z_p = (m_p + torch.exp(logs_p) * rnd.to(m_p.dtype) * 0.66666) * x_mask
            z = net_g.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, 24 : 24 + length]
            x_mask = x_mask[:, :, 24 : 24 + length]
            nsff0 = pitchf[:, head : head + length]
            # GeneratorNSF.forward with n_res = length2 (interpolations always present)
            queue[:] = [sine_noise]
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

    def gen_inputs(d):
        P, R, R2 = d["p_len"], d["return_length"], d["return_length2"]
        feats = torch.randn(1, P, 768, device=dev).to(ftype)
        pitchf = (torch.rand(1, P, device=dev) * 200 + 100).float()
        pitch = torch.clamp(torch.round(1127 * torch.log(1 + pitchf / 700)), 1, 255).long()
        rnd = torch.randn(1, net_g.inter_channels, P - d["flow_head"], device=dev).to(ftype)
        sine_noise = torch.randn(1, R * upp, 1, device=dev)
        n_res = torch.zeros(1, R2, device=dev)
        return feats, pitch, pitchf, rnd, sine_noise, n_res

    gen = Generator().eval()
    export(gen, gen_inputs(d0), OUT / "generator.onnx", ["feats", "pitch", "pitchf", "rnd", "sine_noise", "n_res"], ["audio"],
           torch_folding=False,
           dynamic_axes={"feats": {1: "p_len"}, "pitch": {1: "p_len"}, "pitchf": {1: "p_len"}, "rnd": {2: "flow_len"},
                         "sine_noise": {1: "sine_len"}, "n_res": {1: "ret2_len"}, "audio": {2: "audio_len"}})

    model = {
        "model_sr": int(rvc.tgt_sr), "upp": upp, "half": half, "f0_min": float(rvc.f0_min), "f0_max": float(rvc.f0_max),
        "pitch_cache_len": int(rvc.cache_pitch.shape[0]), "inter_channels": int(net_g.inter_channels),
        "files": {"contentvec": "contentvec.onnx", "rmvpe": "rmvpe.onnx", "fcpe": "fcpe.onnx", "generator": "generator.onnx"},
    }
    (OUT / "model.json").write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(json.dumps(model))

    if not args.check:
        return

    def session(path, overrides):
        import onnxruntime as ort

        ort_session(path)  # loads CUDA DLLs, asserts CUDA EP
        so = ort.SessionOptions()
        for k, v in overrides.items():
            so.add_free_dimension_override_by_name(k, int(v))
        s = ort.InferenceSession(str(path), so, providers=["CUDAExecutionProvider"])
        assert s.get_providers()[0] == "CUDAExecutionProvider"
        return s

    from infer.hubert import extract_hubert_features

    failed = []
    for sr, block, cf, extra, formant in CHECKS:
        for f0 in ("rmvpe", "fcpe"):
            d = od.dims(sr, block, cf, extra, formant, f0)
            rep = {}
            if f0 == "rmvpe":
                audio16 = (torch.randn(1, d["n16"], device=dev) * 0.1).to(ftype)
                with torch.no_grad():
                    ref = extract_hubert_features(rvc.model, audio16, "v2").float().cpu().numpy()
                out = session(OUT / "contentvec.onnx", {"n16": d["n16"]}).run(None, {"audio": audio16.cpu().numpy()})[0]
                assert out.shape[1] == d["feats_frames"], (out.shape, d["feats_frames"])
                rep["contentvec_snr_db"] = snr_db(ref, out.astype(np.float32))
                a = torch.randn(1, d["f0_extractor_frame"], device=dev) * 0.1
                m = rmvpe_mag(a)
                with torch.no_grad():
                    ref = rm.mel2hidden(rm.extract_mel(a)).float().cpu().numpy()
                out = session(OUT / "rmvpe.onnx", {"f0_frames": m.shape[-1]}).run(None, {"mag": m.cpu().numpy()})[0]
                rep["f0_snr_db"] = snr_db(ref, out)
                # generator does not depend on the F0 method
                P, S, R, R2 = d["p_len"], d["skip_head"], d["return_length"], d["return_length2"]
                ins = gen_inputs(d)
                feats, pitch, pitchf, rnd, sine_noise, _ = ins

                def official():
                    queue[:] = [rnd, sine_noise]
                    p_len_t = torch.tensor([P], device=dev, dtype=torch.long)
                    sid = torch.tensor([0], device=dev, dtype=torch.long)
                    return net_g.infer(feats, p_len_t, pitch, pitchf, sid, S, R, R2)[0].float()

                with torch.no_grad():
                    ref = with_noise(official).cpu().numpy()
                ov = {"p_len": P, "flow_len": P - d["flow_head"], "sine_len": R * upp, "ret2_len": R2}
                out = session(OUT / "generator.onnx", ov).run(None, dict(zip(
                    ["feats", "pitch", "pitchf", "rnd", "sine_noise", "n_res"], [t.cpu().numpy() for t in ins])))[0]
                assert out.shape == ref.shape, (out.shape, ref.shape)
                rep["generator_snr_db"] = snr_db(ref, out)
            else:
                a = torch.randn(1, d["f0_extractor_frame"], device=dev) * 0.1
                m = fcpe_mag(a)
                with torch.no_grad():
                    ref = fc.infer_model.infer(a.float(), sr=16000, decoder_mode="local_argmax", threshold=0.006).float().cpu().numpy()
                out = session(OUT / "fcpe.onnx", {"f0_frames": m.shape[-1]}).run(None, {"mag": m.cpu().numpy()})[0]
                rep["f0_snr_db"] = snr_db(ref, out)
            limits = {"generator_snr_db": 25 if half else 40}
            bad = {k: v for k, v in rep.items() if v < limits.get(k, 40)}
            print(json.dumps({"sr": sr, "block": block, "crossfade": cf, "extra": extra, "formant": formant, "f0": f0,
                              "p_len": d["p_len"], "skip_head": d["skip_head"], "R": d["return_length"], "R2": d["return_length2"], **rep}))
            if bad:
                failed.append(((sr, block, cf, extra, formant, f0), bad))
    if failed:
        raise SystemExit(f"dynamic ONNX check failed: {failed}")
    print("dynamic ONNX check passed")


if __name__ == "__main__":
    main()
