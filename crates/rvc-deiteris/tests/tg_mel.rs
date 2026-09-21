#![cfg(feature="onnx")]
use half::f16;
#[test]
fn rust_mel_matches_normal_rmvpe_center_padding_and_half_log() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../PoC/assets/app/deiteris/tg-fast-v1");
    let fixture: serde_json::Value = serde_json::from_str(include_str!("fixtures/tg-fast-mel-v1.json")).unwrap();
    let tolerance = fixture["max_abs"].as_f64().unwrap() as f32;
    let mut mel = rvc_deiteris::mel::Mel::new(&root).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let len = case["length"].as_u64().unwrap() as usize;
        let audio: Vec<_> = (0..len).map(|i| f16::from_f32(((i*37 %257) as i32-128) as f32/256.)).collect();
        let (actual, frames, padded) = mel.extract(&audio).unwrap();
        assert_eq!(frames, case["frames"].as_u64().unwrap() as usize);
        assert_eq!(padded, case["padded"].as_u64().unwrap() as usize);
        let expected = case["mel"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual.to_f32()-expected.as_f64().unwrap() as f32).abs() <= tolerance, "length {len} index {index}: {actual} != {expected}");
        }
    }
}
