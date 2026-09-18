//! The slot loudness the silence rules use.

#[test]
fn slot_db_is_absolute() {
    let sine: Vec<f32> = (0..160).map(|i| 0.1 * (i as f32 * 0.3).sin()).collect();
    let db = rvc_engine::noise::slot_db(&sine);
    // 0.1 amplitude sine: rms = 0.1 / sqrt(2) -> about -23 dB
    assert!((db + 23.0).abs() < 0.5, "{db}");
    assert!(rvc_engine::noise::slot_db(&[0.0; 160]) < -90.0);
    let quiet: Vec<f32> = sine.iter().map(|v| v * 0.01).collect();
    assert!((rvc_engine::noise::slot_db(&quiet) - (db - 40.0)).abs() < 0.1);
}
