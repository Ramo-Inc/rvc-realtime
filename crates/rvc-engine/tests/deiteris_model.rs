//! Deiteris graph selection and cache separation, no GPU/audio devices.
use rvc_engine::voice_model;
use std::path::PathBuf;

#[test]
fn concurrent_first_conversion_keeps_published_cache() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let voice = root.join("assets/app/voices/default_v2_40k.pth");
    let assets = root.join("assets/app/deiteris");
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let cache = root.join(format!("target/deiteris-concurrent-{}-{nonce}", std::process::id()));
    let barrier = std::sync::Barrier::new(2);
    let conversions = std::sync::atomic::AtomicUsize::new(0);
    let paths = std::thread::scope(|scope| {
        let run = || {
            barrier.wait();
            voice_model::prepare_deiteris_with_progress(&voice, &assets, &cache, || {
                conversions.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }).unwrap()
        };
        let a = scope.spawn(run);
        let b = scope.spawn(run);
        (a.join().unwrap(), b.join().unwrap())
    });
    assert_eq!(paths.0, paths.1);
    assert_eq!(conversions.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(paths.0.join("generator.onnx").is_file());
    assert!(paths.0.join("generator.weights").is_file());
    assert!(!paths.0.with_extension("tmp").exists());
}

#[test]
fn deiteris_cache_is_distinct_and_preserves_model_weights() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let voice = root.join("assets/app/voices/default_v2_40k.pth");
    let assets = root.join("assets/app");
    let cache = root.join("target/deiteris-model-test");
    let first = voice_model::prepare_deiteris(&voice, &assets.join("deiteris"), &cache).unwrap();
    let second = voice_model::prepare_deiteris(&voice, &assets.join("deiteris"), &cache).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.parent().unwrap().file_name().unwrap(),
        "deiteris-onnx-v1"
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(first.join("model.json")).unwrap()).unwrap();
    assert_eq!(metadata["generator_kind"], "deiteris-onnx-v1");
    assert_eq!(metadata["model_sr"], 40000);
    assert!(voice_model::prepare_deiteris(&voice, &assets, &cache).is_err());
    // Compare by keys, because initializer ordering is an export detail.
    let old = cache.join("legacy");
    voice_model::convert(&voice, &assets, &old).unwrap();
    let a: serde_json::Value =
        serde_json::from_slice(&std::fs::read(assets.join("templates/40k/template.json")).unwrap())
            .unwrap();
    let b: serde_json::Value = serde_json::from_slice(
        &std::fs::read(assets.join("deiteris/templates/40k/template.json")).unwrap(),
    )
    .unwrap();
    let old_weights = std::fs::read(old.join("generator.weights")).unwrap();
    let new_weights = std::fs::read(first.join("generator.weights")).unwrap();
    assert_eq!(
        a["weights"].as_array().unwrap().len(),
        b["weights"].as_array().unwrap().len()
    );
    for w in a["weights"].as_array().unwrap() {
        let v = b["weights"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["key"] == w["key"])
            .unwrap();
        assert_eq!(w["shape"], v["shape"]);
        let size = w["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .product::<usize>()
            * 2;
        let x = w["offset"].as_u64().unwrap() as usize;
        let y = v["offset"].as_u64().unwrap() as usize;
        assert_eq!(
            &old_weights[x..x + size],
            &new_weights[y..y + size],
            "{}",
            w["key"]
        );
    }
}
