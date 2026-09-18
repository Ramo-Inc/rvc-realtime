//! Voice model conversion against the weights torch writes for the same model
//! (`tools/export_generator_template.py`, reference/40k from the default voice).

use std::collections::HashMap;
use std::path::PathBuf;

use half::f16;
use rvc_engine::voice_model::{self, Unsupported};
use rvc_engine::Error;

fn assets(rel: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/")).join(rel)
}

#[test]
fn converts_like_official_load() {
    let app = assets("app");
    let reference = std::fs::read(app.join("reference/40k/generator.weights")).unwrap();
    let template: serde_json::Value = serde_json::from_slice(&std::fs::read(app.join("templates/40k/template.json")).unwrap()).unwrap();
    for (name, src) in [("pth", "app/voices/default_v2_40k.pth")] {
        let out = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/target/voice-model-test")).join(name);
        voice_model::convert(&assets(src), &app, &out).unwrap();
        let got = std::fs::read(out.join("generator.weights")).unwrap();
        assert_eq!(got.len(), reference.len(), "{name}");
        // weight-norm weights: the reference computes the norm on the GPU, so allow one fp16 step
        for w in template["weights"].as_array().unwrap() {
            let (off, len) = (w["offset"].as_u64().unwrap() as usize, w["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).product::<usize>() * 2);
            let step = if w.get("weight_norm").is_some() { 1 } else { 0 };
            for (a, b) in got[off..off + len].chunks_exact(2).zip(reference[off..off + len].chunks_exact(2)) {
                let (a, b) = (u16::from_le_bytes([a[0], a[1]]), u16::from_le_bytes([b[0], b[1]]));
                // neighbouring fp16 values of the same sign differ by one in their bit pattern
                assert!(a.abs_diff(b) <= step, "{name} {} differs: {} vs {}", w["key"], f16::from_bits(a), f16::from_bits(b));
            }
        }
    }

    // a v1 model is reported as unsupported before any weight is read
    let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/target/voice-model-test"));
    let v1 = dir.join("v1.safetensors");
    let data = [0u8; 2];
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![1], &data).unwrap();
    let meta: HashMap<String, String> = [("version", "v1"), ("f0", "1"), ("config", "[1025, 32, 192, 192, 768, 2, 6, 3, 0, \"1\", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]")]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    safetensors::serialize_to_file([("x", view)], Some(meta), &v1).unwrap();
    assert!(matches!(voice_model::check(&v1, &app), Err(Error::Unsupported(Unsupported::Version(_)))));
}
