"""Reference table for the Rust `Dims::compute` port (trial 02), produced by the unchanged official
`RVCStreamEngine.__init__` (see official_dims.py) plus the rtrvc.infer expressions in export_onnx.dims_from_engine.

- sweeps: every integer block 20..1000, crossfade 10..100 and extra 500..3000 ms (others at the VST defaults
  130 / 80 / 2000) for each sample rate, with formant 0 and both F0 methods
- cross: a grid mixing all axes and formants
- formant: return_length2 for every formant step -12.00..12.00 (0.01) and every return_length seen in the
  sweeps, and upp_res per formant step

Usage: uv run --project PoC/tools python PoC/tools/make_dims_fixture.py --out PoC/02-rust-ort-dynamic-shape/tests/fixtures/dims.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_dims import OfficialDims  # noqa: E402

RATES = [44100, 48000, 88200, 96000]
KEYS = ["zc", "block_frame", "block_frame_16k", "crossfade_frame", "sola_buffer_frame", "sola_search_frame", "extra_frame",
        "input_wav_len", "n16", "p_len", "skip_head", "return_length", "return_length2", "upp_res", "flow_head",
        "f0_extractor_frame", "f0_frames", "feats_frames"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()  # before OfficialDims: the official engine chdirs into its root
    od = OfficialDims()
    cases = []

    def add(sr, block, cf, extra, formant):
        e = od.engine(sr, block, cf, extra)
        for f0 in ("rmvpe", "fcpe"):
            d = od.dims(sr, block, cf, extra, formant, f0, engine=e)
            cases.append([sr, block, cf, extra, formant, f0] + [d[k] for k in KEYS])
        return d

    returns = set()
    for sr in RATES:
        for block in range(20, 1001):
            returns.add(add(sr, block, 80, 2000, 0.0)["return_length"])
        for cf in range(10, 101):
            returns.add(add(sr, 130, cf, 2000, 0.0)["return_length"])
        for extra in range(500, 3001):
            add(sr, 130, 80, extra, 0.0)
        print(f"sweeps done for {sr}", flush=True)
    for sr in RATES:
        for block in (20, 37, 60, 130, 555, 1000):
            for cf in (10, 41, 100):
                for extra in (500, 1234, 3000):
                    for formant in (-12.0, -0.2, 3.33, 12.0):
                        returns.add(add(sr, block, cf, extra, formant)["return_length"])

    # formant-only expressions (rtrvc.infer), evaluated exactly as in dims_from_engine
    tgt_sr = int(od.rvc.tgt_sr)
    formants = [k / 100 for k in range(-1200, 1201)]
    returns = sorted(returns)
    return_length2 = [[int(math.ceil(r * 2 ** (f / 12))) for r in returns] for f in formants]
    upp_res = [int(np.floor(2 ** (f / 12) * tgt_sr // 100)) for f in formants]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": {"model_sr": tgt_sr, "f0_min": float(od.rvc.f0_min), "f0_max": float(od.rvc.f0_max),
                  "pitch_cache_len": int(od.rvc.cache_pitch.shape[0])},
        "columns": ["sr", "block_ms", "crossfade_ms", "extra_ms", "formant", "f0"] + KEYS,
        "cases": cases,
        "formant": {"formants": formants, "return_lengths": returns, "return_length2": return_length2, "upp_res": upp_res},
    }), encoding="utf-8")
    print(f"wrote {out}: {len(cases)} cases, {len(formants)} x {len(returns)} formant entries")


if __name__ == "__main__":
    main()
