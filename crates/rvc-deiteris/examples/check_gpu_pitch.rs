//! Device-free acceptance against actual TG CUDA tensors; strict gate, not a benchmark.
use anyhow::{ensure, Result};
use half::f16;
use rvc_deiteris::{gpu_pitch::GpuPitch, pitch::PitchState};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

fn halves(path: &Path) -> Result<Vec<f16>> {
    let b = std::fs::read(path)?;
    ensure!(b.len() % 2 == 0, "invalid half fixture");
    Ok(b.chunks_exact(2)
        .map(|v| f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
        .collect())
}
fn integers(path: &Path) -> Result<Vec<i64>> {
    let b = std::fs::read(path)?;
    ensure!(b.len() % 8 == 0, "invalid integer fixture");
    Ok(b.chunks_exact(8)
        .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn delta(a: &[f32], b: &[f32]) -> Result<f32> {
    ensure!(
        !a.is_empty() && a.len() == b.len() && a.iter().chain(b).all(|v| v.is_finite()),
        "shape/finite mismatch"
    );
    Ok(a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max))
}
fn half_delta(a: &[f16], b: &[f16]) -> Result<f32> {
    delta(
        &a.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
        &b.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
    )
}
fn report(path: &Path, rows: &[Value], complete: bool, pass: bool) -> Result<()> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&json!({
        "scope":"normal RMVPE component only; not whole voice conversion or device/noise acceptance",
        "complete": complete, "pass": pass, "rows": rows}))?,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 5,
        "usage: check_gpu_pitch GPU_ASSETS FIXTURES REPORT RUNTIME"
    );
    let assets = Path::new(&args[1]);
    let fixtures = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    ensure!(!output.exists(), "report must be new");
    let runtime = std::path::absolute(&args[4])?;
    let old = std::env::var_os("PATH").unwrap_or_default();
    std::env::set_var(
        "PATH",
        std::env::join_paths(std::iter::once(runtime.clone()).chain(std::env::split_paths(&old)))?,
    );
    let manifest: Value = serde_json::from_slice(&std::fs::read(fixtures.join("manifest.json"))?)?;
    let cases = manifest["cases"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing cases"))?;
    ensure!(cases.len() >= 36, "incomplete fixture");
    let mut rows = Vec::new();
    let mut first = BTreeMap::<(usize, String), Vec<f32>>::new();
    report(output, &rows, false, false)?;
    for length in [3040, 10560] {
        for graph in [false, true] {
            // Same startup is explicitly destroyed/recreated, not just different shapes.
            for generation in 0..2 {
                let mut engine = GpuPitch::new(&runtime, assets, length, graph)?;
                let mut cache = PitchState::new(256)?;
                let mut reference_cache = PitchState::new(256)?;
                for case in cases
                    .iter()
                    .filter(|c| c["length"].as_u64() == Some(length as u64))
                {
                    let name = case["case"]
                        .as_str()
                        .ok_or_else(|| anyhow::anyhow!("missing case name"))?;
                    let path = fixtures.join(length.to_string()).join(name);
                    let audio = halves(&path.join("audio.bin"))?;
                    let start = std::time::Instant::now();
                    let raw = engine.process(&audio)?.to_vec();
                    let ms = start.elapsed().as_secs_f64() * 1000.;
                    // Process has already completed; observation cannot repair scheduling.
                    let observed = engine.observe()?;
                    let mel_max = half_delta(&observed.mel, &halves(&path.join("mel.bin"))?)?;
                    let salience_max =
                        half_delta(&observed.salience, &halves(&path.join("salience.bin"))?)?;
                    let bytes = std::fs::read(path.join("f0.bin"))?;
                    ensure!(bytes.len() % 4 == 0, "invalid F0 fixture");
                    let reference: Vec<_> = bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    let f0_max = delta(&raw, &reference)?;
                    let uv = raw
                        .iter()
                        .zip(&reference)
                        .filter(|(a, b)| (**a > 0.) != (**b > 0.))
                        .count();
                    let mut max_cents = 0f32;
                    for (&a, &b) in raw.iter().zip(&reference) {
                        if a > 0. && b > 0. {
                            max_cents = max_cents.max((1200. * (a / b).log2()).abs());
                        }
                    }
                    let mut coarse_mismatches = 0;
                    for (tune, formant, file) in
                        [(0., 0., "coarse.bin"), (14., -0.2, "coarse_shifted.bin")]
                    {
                        let mut pitch = PitchState::new(raw.len())?;
                        pitch.update(&raw, tune, formant)?;
                        let expected = integers(&path.join(file))?;
                        ensure!(expected.len() == raw.len(), "coarse shape mismatch");
                        coarse_mismatches += pitch.coarse()[1..]
                            .iter()
                            .zip(expected)
                            .filter(|(a, b)| **a != *b)
                            .count();
                    }
                    // Propagation through existing Rust cache, not independent GPU-cache acceptance.
                    cache.update(&raw, 14., -0.2)?;
                    reference_cache.update(&reference, 14., -0.2)?;
                    let cache_mismatches = cache
                        .coarse()
                        .iter()
                        .zip(reference_cache.coarse())
                        .filter(|(a, b)| a != b)
                        .count();
                    let key = (length, name.to_owned());
                    let graph_recreate_delta = match first.get(&key) {
                        Some(previous) => delta(&raw, previous)?,
                        None => {
                            first.insert(key, raw.clone());
                            0.
                        }
                    };
                    rows.push(json!({"length":length,"graph":graph,"generation":generation,"case":name,
                        "mel_max":mel_max,"salience_max":salience_max,"f0_max":f0_max,"uv_mismatches":uv,
                        "coarse_mismatches":coarse_mismatches,"cache_mismatches":cache_mismatches,
                        "graph_recreate_delta":graph_recreate_delta,"max_cents":max_cents,"ms":ms}));
                    report(output, &rows, false, false)?;
                }
                drop(engine);
            }
        }
    }
    let pass = rows.len() == cases.len() * 4
        && rows.iter().all(|r| {
            r["mel_max"] == 0.0
                && r["salience_max"] == 0.0
                && r["f0_max"] == 0.0
                && r["coarse_mismatches"] == 0
                && r["cache_mismatches"] == 0
                && r["graph_recreate_delta"] == 0.0
        });
    report(output, &rows, true, pass)?;
    ensure!(
        pass,
        "strict TG CUDA pitch parity failed; see {}",
        output.display()
    );
    Ok(())
}
