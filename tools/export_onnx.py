"""Export fixed-shape ONNX models + manifest.json for one realtime config, built from the official modules.

Dimensions are read from an instantiated official RVCStreamEngine (not re-derived), so the ONNX
shapes and the Rust engine follow exactly the buffers the official realtime path uses.

Usage:
  uv run --project tools python tools/export_onnx.py --config tools/configs/base.json [--check] \
      [--dsp-fixture crates/rvc-engine/tests/fixtures/dsp.json] [--fcpe-fp16]
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

POC = Path(__file__).resolve().parents[1]
RVC_ROOT = POC / "assets" / "official" / "rvc"
MODEL_PTH = POC / "assets" / "app" / "voices" / "default_v2_40k.pth"
OPSET = 17


def load_engine(cfg):
    os.environ["RVC_CUDA_GRAPH"] = "0"
    sys.argv = [sys.argv[0]]
    path = RVC_ROOT / "RVCRealtimeVST" / "worker" / "rvc_worker.py"
    spec = importlib.util.spec_from_file_location("rvc_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RVCStreamEngine(
        {
            "rvc_root": str(RVC_ROOT),
            "sample_rate": cfg["device_sr"],
            "block_ms": cfg["block_ms"],
            "crossfade_ms": cfg["crossfade_ms"],
            "extra_ms": cfg["extra_ms"],
            "model_path": str(MODEL_PTH),
            "index_path": "",
        }
    )


def dims_from_engine(engine, cfg):
    rvc = engine.rvc
    n16 = int(engine.input_wav_res.shape[0])
    p_len = n16 // 160
    factor = 2 ** (cfg["formant"] / 12)
    return_length = int(engine.return_length)
    return_length2 = int(math.ceil(return_length * factor))  # rtrvc.infer
    upp_res = int(np.floor(factor * rvc.tgt_sr // 100))  # rtrvc.infer (same precedence)
    skip_head = int(engine.skip_head)
    flow_head = max(skip_head - 24, 0)  # models.infer
    f0_frame = engine.block_frame_16k + 800  # rtrvc.infer
    if cfg["f0"] == "rmvpe":
        f0_frame = 5120 * ((f0_frame - 1) // 5120 + 1) - 160
    return {
        "device_sr": cfg["device_sr"],
        "model_sr": int(rvc.tgt_sr),
        "zc": int(engine.zc),
        "block_frame": int(engine.block_frame),
        "block_frame_16k": int(engine.block_frame_16k),
        "crossfade_frame": int(engine.crossfade_frame),
        "sola_buffer_frame": int(engine.sola_buffer_frame),
        "sola_search_frame": int(engine.sola_search_frame),
        "extra_frame": int(engine.extra_frame),
        "input_wav_len": int(engine.input_wav.shape[0]),
        "n16": n16,
        "p_len": p_len,
        "skip_head": skip_head,
        "return_length": return_length,
        "return_length2": return_length2,
        "upp_res": upp_res,
        "flow_head": flow_head,
        "f0_extractor_frame": f0_frame,
        "f0_frames": f0_frame // 160 + 1,
        "f0_min": float(rvc.f0_min),
        "f0_max": float(rvc.f0_max),
        "pitch_cache_len": int(rvc.cache_pitch.shape[0]),
    }


def export(module, args, path, input_names, output_names, torch_folding=True, dynamic_axes=None):
    import onnx
    import onnxslim
    import torch

    tmp = str(path) + ".tmp"
    with torch.no_grad():
        # torch's own folding pass fails on the generator (mixed cpu/cuda constants); onnxslim folds instead.
        torch.onnx.export(module, args, tmp, input_names=input_names, output_names=output_names,
                          opset_version=OPSET, do_constant_folding=torch_folding, dynamo=False,
                          dynamic_axes=dynamic_axes)
    model = onnxslim.slim(onnx.load(tmp))
    onnx.save(model, str(path))
    os.remove(tmp)
    return path


def snr_db(ref, test):
    ref, test = np.asarray(ref, np.float64), np.asarray(test, np.float64)
    finite = np.isfinite(ref) & np.isfinite(test)
    err = np.sum((ref[finite] - test[finite]) ** 2)
    return float(10 * np.log10(np.sum(ref[finite] ** 2) / max(err, 1e-30)))


def ort_session(path):
    import torch  # noqa: F401  (loads CUDA/cuDNN DLLs that onnxruntime-gpu uses)

    torch_lib = Path(torch.__file__).parent / "lib"
    os.add_dll_directory(str(torch_lib))
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
    assert s.get_providers()[0] == "CUDAExecutionProvider", s.get_providers()
    return s


def rmvpe_mag(audio):
    """|STFT| exactly as infer/rmvpe.py MelSpectrogram.forward (center=True, reflect, periodic hann)."""
    import torch

    fft = torch.stft(audio, n_fft=1024, hop_length=160, win_length=1024,
                     window=torch.hann_window(1024, device=audio.device), center=True, return_complex=True)
    return torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))


def fcpe_mag(audio):
    """|STFT| exactly as torchfcpe MelModule (pad (win-hop)//2 reflect/constant, center=False, +1e-9),
    followed by Wav2MelModule's frame-count fix (duplicate last frame up to T//160+1)."""
    import torch

    win, hop = 1024, 160
    pad_left = (win - hop) // 2
    pad_right = max((win - hop + 1) // 2, win - audio.size(-1) - pad_left)
    mode = "reflect" if pad_right < audio.size(-1) else "constant"
    y = torch.nn.functional.pad(audio.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)
    spec = torch.stft(y, win, hop_length=hop, win_length=win, window=torch.hann_window(win, device=audio.device),
                      center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    n_frames = audio.shape[-1] // hop + 1
    if n_frames > mag.shape[-1]:
        mag = torch.cat((mag, mag[..., -1:]), -1)
    return mag[..., :n_frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dsp-fixture", default=None)
    ap.add_argument("--fcpe-fp16", action="store_true", help="export fcpe in fp16 (VRAM variant)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = POC / "assets" / "onnx" / cfg["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = Path(args.dsp_fixture).resolve() if args.dsp_fixture else None

    import torch
    import torch.nn as nn

    engine = load_engine(cfg)
    rvc = engine.rvc
    dev = torch.device(rvc.device)
    half = bool(rvc.is_half)
    d = dims_from_engine(engine, cfg)
    ftype = torch.float16 if half else torch.float32
    report = {}
    torch.manual_seed(0)

    # ---- ContentVec (infer/hubert.py extract_hubert_features, v2 -> last_hidden_state)
    from infer.hubert import extract_hubert_features

    class ContentVec(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, audio):
            return self.m(input_values=audio, attention_mask=None, output_hidden_states=False,
                          return_dict=True).last_hidden_state

    audio16 = (torch.randn(1, d["n16"], device=dev) * 0.1).to(ftype)
    cv_path = export(ContentVec(rvc.model).eval(), (audio16,), out_dir / "contentvec.onnx", ["audio"], ["feats"])
    with torch.no_grad():
        feats_t = extract_hubert_features(rvc.model, audio16, "v2")
    d["feats_frames"] = int(feats_t.shape[1])

    # ---- F0
    if cfg["f0"] == "rmvpe":
        from infer.rmvpe import RMVPE

        rm = RMVPE(str(RVC_ROOT / "assets" / "rmvpe" / "rmvpe.pt"), is_half=half, device=str(dev))

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
                x = nn.functional.pad(x, (0, 32 * ((n - 1) // 32 + 1) - n), mode="constant")
                return self.model(x)[:, :n].float()

        f0_audio = torch.randn(1, d["f0_extractor_frame"], device=dev) * 0.1
        mag = rmvpe_mag(f0_audio)
        f0_path = export(Rmvpe(rm).eval(), (mag,), out_dir / "rmvpe.onnx", ["mag"], ["hidden"])
        with torch.no_grad():
            f0_t = rm.mel2hidden(rm.extract_mel(f0_audio)).float()
        f0_io = {"input": "mag", "output": "hidden", "frames": int(mag.shape[-1])}
    else:
        from infer.fcpe import FCPEInfer

        fc = FCPEInfer(dev)
        fp16 = args.fcpe_fp16

        class Fcpe(nn.Module):
            def __init__(self, fc):
                super().__init__()
                self.fc = fc
                self.register_buffer("basis", fc.infer_model.wav2mel.mel_extractor.mel_basis.clone())
                self.model = fc.infer_model.model

            def forward(self, mag):
                mel = torch.log(torch.clamp(torch.matmul(self.basis, mag), min=1e-5)).transpose(-1, -2)
                if fp16:
                    mel = mel.half()
                return self.fc._graphable_model_infer(mel, "local_argmax", 0.006).float()

        if fp16:
            fc.infer_model.model.half()
            fc.infer_model.model.cent_table = fc.infer_model.model.cent_table.half()
        f0_audio = torch.randn(1, d["f0_extractor_frame"], device=dev) * 0.1
        mag = fcpe_mag(f0_audio)
        name = "fcpe_fp16.onnx" if fp16 else "fcpe.onnx"
        f0_path = export(Fcpe(fc).eval(), (mag,), out_dir / name, ["mag"], ["f0"])
        if fp16:
            fc = FCPEInfer(dev)  # torch reference stays the official fp32 path
        with torch.no_grad():
            f0_t = fc.infer_model.infer(f0_audio.float(), sr=16000, decoder_mode="local_argmax", threshold=0.006).float()
        f0_io = {"input": "mag", "output": "f0", "frames": int(mag.shape[-1])}

    # ---- Generator (models.py SynthesizerTrnMs768NSFsid.infer with the two randn_like as inputs)
    net_g = rvc.net_g
    assert net_g.dec.m_source.l_sin_gen.dim == 1, "rand_ini is only all-zero when harmonic_num == 0"
    P, S, R, R2 = d["p_len"], d["skip_head"], d["return_length"], d["return_length2"]
    noise_queue = []

    class Generator(nn.Module):
        def __init__(self, net_g):
            super().__init__()
            self.net_g = net_g

        def forward(self, feats, pitch, pitchf, rnd, sine_noise):
            noise_queue[:] = [rnd, sine_noise]
            orig_randn_like, orig_rand = torch.randn_like, torch.rand
            torch.randn_like = lambda t, *a, **k: noise_queue.pop(0).to(t.dtype)
            torch.rand = lambda *size, device=None, **k: torch.zeros(*size, device=device)
            try:
                p_len = torch.tensor([P], device=feats.device, dtype=torch.long)
                sid = torch.tensor([0], device=feats.device, dtype=torch.long)
                out = self.net_g.infer(feats, p_len, pitch, pitchf, sid, S, R, R2)[0]
            finally:
                torch.randn_like, torch.rand = orig_randn_like, orig_rand
            return out.float()

    feats = torch.randn(1, P, 768, device=dev).to(ftype)
    pitchf = (torch.rand(1, P, device=dev) * 200 + 100).float()
    pitch = torch.clamp(torch.round(1127 * torch.log(1 + pitchf / 700)), 1, 255).long()
    rnd = torch.randn(1, net_g.inter_channels, P - d["flow_head"], device=dev).to(ftype)
    sine_noise = torch.randn(1, R * 400, 1, device=dev)
    gen = Generator(net_g).eval()
    gen_path = export(gen, (feats, pitch, pitchf, rnd, sine_noise), out_dir / "generator.onnx",
                      ["feats", "pitch", "pitchf", "rnd", "sine_noise"], ["audio"], torch_folding=False)
    with torch.no_grad():
        audio_t = gen(feats, pitch, pitchf, rnd, sine_noise)

    manifest = {
        "config": cfg,
        "dims": d,
        "half": half,
        "files": {"contentvec": cv_path.name, "f0": f0_path.name, "generator": gen_path.name},
        "io": {
            "contentvec": {"audio": [1, d["n16"]], "feats": [1, d["feats_frames"], 768], "dtype": "f16" if half else "f32"},
            "f0": {**f0_io, "shape": [1, 513, f0_io["frames"]]},
            "generator": {"feats": [1, P, 768], "pitch": [1, P], "pitchf": [1, P], "rnd": list(rnd.shape),
                          "sine_noise": list(sine_noise.shape), "audio": list(audio_t.shape)},
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest["dims"]))

    if args.check:
        s = ort_session(cv_path)
        cv_o = s.run(None, {"audio": audio16.cpu().numpy()})[0]
        report["contentvec_snr_db"] = snr_db(feats_t.float().cpu().numpy(), cv_o.astype(np.float32))
        s = ort_session(f0_path)
        f0_o = s.run(None, {"mag": mag.cpu().numpy()})[0]
        report["f0_snr_db"] = snr_db(f0_t.cpu().numpy(), f0_o)
        s = ort_session(gen_path)
        g_o = s.run(None, {"feats": feats.cpu().numpy(), "pitch": pitch.cpu().numpy(), "pitchf": pitchf.cpu().numpy(),
                           "rnd": rnd.cpu().numpy(), "sine_noise": sine_noise.cpu().numpy()})[0]
        report["generator_snr_db"] = snr_db(audio_t.cpu().numpy(), g_o)
        print(json.dumps({"check": report}))
        # fp16 generator: flow + NSF decoder amplify fp16 kernel differences (fp32 export measured 49 dB,
        # fp16 30-39 dB on random inputs); a structural error lands near 0 dB.
        limits = {"generator_snr_db": 25 if half else 40}
        if cfg["f0"] == "fcpe" and args.fcpe_fp16:
            limits["f0_snr_db"] = 25  # fp16 variant is compared with the official fp32 FCPE
        bad = {k: v for k, v in report.items() if v < limits.get(k, 40)}
        if bad:
            raise SystemExit(f"ONNX check failed (< 40 dB): {bad}")

    if fixture:
        write_dsp_fixture(fixture)


def write_dsp_fixture(path):
    """Reference values for the Rust dsp port (torchaudio Resample, librosa rms, linear interp, STFT mags)."""
    import librosa
    import torch
    import torch.nn.functional as F
    from torchaudio.transforms import Resample

    rng = np.random.default_rng(7)

    def signal(n, sr):
        t = np.arange(n) / sr
        return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1330 * t)
                + 0.05 * rng.standard_normal(n)).astype(np.float32)

    cases = {"resample": [], "rms": [], "interp": [], "stft_rmvpe": [], "stft_fcpe": []}
    for orig, new, n in [(48000, 16000, 7200), (40000, 48000, 7200), (395, 400, 7110),
                         # trial 02 startup ranges: 44.1 / 88.2 / 96 kHz devices, formant +12 / +3.33 / -12 (upp_res 800 / 484 / 200)
                         (44100, 16000, 7200), (40000, 44100, 7200), (88200, 16000, 7200), (40000, 88200, 7200),
                         (96000, 16000, 7200), (40000, 96000, 7200), (800, 400, 7200), (484, 400, 7260), (200, 400, 7200)]:
        x = signal(n, max(orig, 16000))
        y = Resample(orig_freq=orig, new_freq=new, dtype=torch.float32)(torch.from_numpy(x)).numpy()
        cases["resample"].append({"orig": orig, "new": new, "input": x.tolist(), "output": y.tolist()})
    x = signal(8640, 48000)
    r = librosa.feature.rms(y=x, frame_length=1920, hop_length=480)
    cases["rms"].append({"frame": 1920, "hop": 480, "input": x.tolist(), "output": r[0].tolist()})
    it = F.interpolate(torch.from_numpy(r).unsqueeze(0), size=8641, mode="linear", align_corners=True)[0, 0, :-1]
    cases["interp"].append({"input": r[0].tolist(), "size": 8640, "output": it.numpy().tolist()})
    x = signal(4960, 16000)
    cases["stft_rmvpe"].append({"input": x.tolist(), "output": rmvpe_mag(torch.from_numpy(x)[None])[0].numpy().T.tolist()})
    x = signal(2880, 16000)
    cases["stft_fcpe"].append({"input": x.tolist(), "output": fcpe_mag(torch.from_numpy(x)[None])[0].numpy().T.tolist()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
