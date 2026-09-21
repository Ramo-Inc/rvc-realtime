//! dsp ports vs values produced by the originals (torchaudio Resample, librosa rms, torch interpolate,
//! torch.stft magnitudes) — fixture written by `PoC/tools/export_onnx.py --dsp-fixture`.

use rvc_engine::dsp;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn max_rel_err(expected: &[f32], got: &[f32]) -> f32 {
    assert_eq!(expected.len(), got.len(), "length");
    let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    expected.iter().zip(got).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}

#[test]
fn dsp_matches_official_ops() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsp.json")).unwrap();
    let fx: Value = serde_json::from_str(&text).unwrap();
    let stft = dsp::Stft::new();
    let mut rows: Vec<(String, f32)> = Vec::new();

    for c in fx["resample"].as_array().unwrap() {
        let r = dsp::Resampler::new(c["orig"].as_u64().unwrap() as usize, c["new"].as_u64().unwrap() as usize);
        rows.push((format!("resample {}->{}", c["orig"], c["new"]), max_rel_err(&floats(&c["output"]), &r.process(&floats(&c["input"])))));
    }
    let c = &fx["rms"][0];
    rows.push(("rms".into(), max_rel_err(&floats(&c["output"]), &dsp::rms(&floats(&c["input"]), 1920, 480))));
    let c = &fx["interp"][0];
    let got = dsp::interp_linear_align_corners(&floats(&c["input"]), 8641);
    rows.push(("interp".into(), max_rel_err(&floats(&c["output"]), &got[..8640])));
    for (name, rmvpe) in [("stft_rmvpe", true), ("stft_fcpe", false)] {
        let c = &fx[name][0];
        let x = floats(&c["input"]);
        let (mag, frames) = if rmvpe { stft.rmvpe(&x) } else { stft.fcpe(&x) };
        // fixture layout: [frame][bin]; ours: [bin][frame]
        let exp: Vec<Vec<f32>> = c["output"].as_array().unwrap().iter().map(floats).collect();
        assert_eq!(exp.len(), frames, "{name} frames");
        let mut got = Vec::new();
        let mut want = Vec::new();
        for (f, row) in exp.iter().enumerate() {
            for (b, v) in row.iter().enumerate() {
                want.push(*v);
                got.push(mag[b * frames + f]);
            }
        }
        rows.push((name.into(), max_rel_err(&want, &got)));
    }

    for (name, err) in &rows {
        println!("{name}: max rel err {err:.2e}");
    }
    let bad: Vec<_> = rows.iter().filter(|(_, e)| *e > 1e-4).collect();
    assert!(bad.is_empty(), "mismatch: {bad:?}");
}
