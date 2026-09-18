//! `Dims::compute` vs the official engine: every block / crossfade / extra step per sample rate, a mixed grid,
//! and every formant step for every return_length — fixture written by `tools/make_dims_fixture.py`.

use rvc_engine::config::{self, Dims, F0Method, Model, ModelFiles, Startup};
use serde_json::Value;

#[test]
fn dims_match_official_engine() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dims.json")).unwrap();
    let fx: Value = serde_json::from_str(&text).unwrap();
    let fm = &fx["model"];
    let model = Model {
        model_sr: fm["model_sr"].as_u64().unwrap() as usize,
        upp: 400,
        half: true,
        f0_min: fm["f0_min"].as_f64().unwrap() as f32,
        f0_max: fm["f0_max"].as_f64().unwrap() as f32,
        pitch_cache_len: fm["pitch_cache_len"].as_u64().unwrap() as usize,
        inter_channels: 192,
        files: ModelFiles { contentvec: String::new(), rmvpe: String::new(), fcpe: String::new(), generator: String::new() },
        dir: Default::default(),
    };
    let cols: Vec<&str> = fx["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let mut mismatches = Vec::new();
    let cases = fx["cases"].as_array().unwrap();
    for c in cases {
        let c = c.as_array().unwrap();
        let s = Startup {
            sample_rate: c[0].as_u64().unwrap() as u32,
            block_ms: c[1].as_f64().unwrap(),
            crossfade_ms: c[2].as_f64().unwrap(),
            extra_ms: c[3].as_f64().unwrap(),
            formant: c[4].as_f64().unwrap(),
            f0: if c[5].as_str().unwrap() == "rmvpe" { F0Method::Rmvpe } else { F0Method::Fcpe },
            variant: Default::default(),
        };
        let d = Dims::compute(&model, &s).unwrap();
        let got = [d.zc, d.block_frame, d.block_frame_16k, d.crossfade_frame, d.sola_buffer_frame, d.sola_search_frame, d.extra_frame,
                   d.input_wav_len, d.n16, d.p_len, d.skip_head, d.return_length, d.return_length2, d.upp_res, d.flow_head,
                   d.f0_extractor_frame, d.f0_frames, d.feats_frames];
        for (i, g) in got.iter().enumerate() {
            let want = c[6 + i].as_u64().unwrap() as usize;
            if *g != want && mismatches.len() < 20 {
                mismatches.push(format!("{s:?} {}: rust {g} official {want}", cols[6 + i]));
            }
        }
    }

    let f = &fx["formant"];
    let formants: Vec<f64> = f["formants"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let returns: Vec<usize> = f["return_lengths"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    for (i, &formant) in formants.iter().enumerate() {
        let want = f["upp_res"][i].as_u64().unwrap() as usize;
        let got = config::upp_res(model.model_sr, formant);
        if got != want && mismatches.len() < 20 {
            mismatches.push(format!("formant {formant} upp_res: rust {got} official {want}"));
        }
        for (j, &r) in returns.iter().enumerate() {
            let want = f["return_length2"][i][j].as_u64().unwrap() as usize;
            let got = config::return_length2(r, formant);
            if got != want && mismatches.len() < 20 {
                mismatches.push(format!("formant {formant} return_length {r}: rust {got} official {want}"));
            }
        }
    }
    println!("{} engine cases, {} formant x {} return_length entries", cases.len(), formants.len(), returns.len());
    assert!(mismatches.is_empty(), "mismatches:\n{}", mismatches.join("\n"));
}

/// Context, crossfade and block share the pitch cache (1024 frames): what fits computes, what does not is
/// refused at startup instead of running past the cache.
#[test]
fn pitch_cache_limits_the_context() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dims.json")).unwrap();
    let fx: Value = serde_json::from_str(&text).unwrap();
    let model = Model {
        model_sr: 40000,
        upp: 400,
        half: true,
        f0_min: 50.0,
        f0_max: 1100.0,
        pitch_cache_len: fx["model"]["pitch_cache_len"].as_u64().unwrap() as usize,
        inter_channels: 192,
        files: ModelFiles { contentvec: String::new(), rmvpe: String::new(), fcpe: String::new(), generator: String::new() },
        dir: Default::default(),
    };
    let startup = |block_ms, crossfade_ms, extra_ms| Startup {
        sample_rate: 48000,
        block_ms,
        crossfade_ms,
        extra_ms,
        formant: 0.0,
        f0: F0Method::Fcpe,
        variant: Default::default(),
    };
    // 1024 frames of 10 ms, minus block, crossfade and the 10 ms SOLA search
    assert!(Dims::compute(&model, &startup(60.0, 80.0, 10_000.0)).is_ok());
    assert!(Dims::compute(&model, &startup(200.0, 80.0, 10_000.0)).is_err());
    assert!(Dims::compute(&model, &startup(200.0, 80.0, 9_950.0)).is_ok());
}
