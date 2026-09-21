//! Development-only stage reproducibility probe. No devices or GUI.
use std::{collections::BTreeMap, path::PathBuf};
use clap::Parser;
use half::f16;
#[derive(Parser)]
struct Args {
    #[arg(long)] model: PathBuf,
    #[arg(long)] assets: PathBuf,
    #[arg(long)] runtime: PathBuf,
    #[arg(long)] input: PathBuf,
    #[arg(long)] report: PathBuf,
    /// Use the engine's integrated GPU CV/RMVPE path; assets is still tg-fast-v1.
    #[arg(long)] gpu_assets: Option<PathBuf>,
    /// Compare reference CPU-DSP adapter with the GPU engine, not two recreations.
    #[arg(long, requires = "gpu_assets")] compare_reference: bool,
    /// Reset and replay one session instead of recreating it (isolates autotuning).
    #[arg(long, conflicts_with = "compare_reference")] same_session: bool,
    /// Record a full input's pitch stages once; diagnostic only, not a timing run.
    #[arg(long, requires = "gpu_assets", conflicts_with_all = ["compare_reference", "same_session"])] trace_pitch: bool,
    #[arg(long, default_value_t = 19)] chunk: usize,
    #[arg(long, default_value_t = 500.0)] extra_ms: f64,
    #[arg(long, default_value_t = 100.0)] crossfade_ms: f64,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    if a.report.exists() { return Err("report must be new".into()); }
    let previous = std::env::var_os("PATH").unwrap_or_default();
    std::env::set_var("PATH", format!("{};{}", a.runtime.canonicalize()?.display(), previous.to_string_lossy()));
    let mut reader = hound::WavReader::open(a.input)?;
    if reader.spec().channels != 1 || reader.spec().sample_format != hound::SampleFormat::Float {
        return Err("requires mono float32 WAV".into());
    }
    let rate = reader.spec().sample_rate as usize;
    let samples = reader.samples::<f32>().collect::<Result<Vec<_>,_>>()?;
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) { return Err("invalid input".into()); }
    let startup = rvc_deiteris::startup::Startup {sample_rate: rate, chunk: a.chunk, block_frames: None, extra_ms: a.extra_ms, crossfade_ms: a.crossfade_ms, formant: -0.2};
    let params = rvc_deiteris::engine::Params {pitch: 14., threshold_db: -90.};
    if a.trace_pitch {
        let mut engine = rvc_deiteris::tg::Engine::new_fast_v2(&a.model, &a.assets,
            a.gpu_assets.as_ref().unwrap(), &a.runtime.canonicalize()?, startup.clone(), true)?;
        engine.prepare(1)?;
        let mut rows = Vec::new();
        let mut input = vec![0.; engine.block_frames()];
        for (block, samples) in samples.chunks(input.len()).enumerate() {
            input.fill(0.);
            input[..samples.len()].copy_from_slice(samples);
            let mut values = BTreeMap::<String, Vec<f32>>::new();
            engine.process_observed(&input, &params, &mut |name, value| {
                match name {
                    "rawf0" => { values.insert(name.into(), value.try_extract_tensor::<f32>()?.1.to_vec()); }
                    "pitchf" => { values.insert(name.into(), value.try_extract_tensor::<f16>()?.1.iter().map(|v| v.to_f32()).collect()); }
                    _ => {}
                }
                Ok(())
            })?;
            rows.push(serde_json::json!({"block":block,"values":values}));
        }
        std::fs::write(a.report, serde_json::to_vec_pretty(&serde_json::json!({
            "scope":"C1 pitch trace, full recorded input; zero-pad final block; observation timings are not performance evidence",
            "startup":startup,"dimensions":engine.dimensions(),"rows":rows
        }))?)?;
        return Ok(());
    }
    let mut reference = BTreeMap::<(usize,String),Vec<f32>>::new();
    let mut differences = Vec::new();
    let mut parallel = Vec::new();
    let mut retained = None;
    for pass in 0..2 {
        let mut engine = if let Some(engine) = retained.take() { engine }
        else if let Some(gpu) = a.gpu_assets.as_ref().filter(|_| !a.compare_reference || pass > 0) {
            rvc_deiteris::tg::Engine::new_fast_v2(&a.model, &a.assets, gpu, &a.runtime.canonicalize()?, startup.clone(), true)?
        } else { rvc_deiteris::tg::Engine::new(&a.model, &a.assets, startup.clone(), true)? };
        engine.prepare(1)?;
        for (block, input) in samples.chunks_exact(engine.block_frames()).take(8).enumerate() {
            engine.process_observed(input, &params, &mut |name, value| {
                let data = if let Ok((_,v)) = value.try_extract_tensor::<f32>() { v.to_vec() }
                    else if let Ok((_,v)) = value.try_extract_tensor::<f16>() { v.iter().map(|v|v.to_f32()).collect() }
                    else if let Ok((_,v)) = value.try_extract_tensor::<i64>() { v.iter().map(|&v|v as f32).collect() }
                    else { return Ok(()) };
                let key = (block, name.to_string());
                if data.iter().any(|v| !v.is_finite()) { return Err(std::io::Error::other(format!("nonfinite stage: {name}")).into()); }
                if name == "parallel_gpu_ms" {
                    parallel.push(serde_json::json!({"pass":pass,"block":block,"pitch_ms":data[0],"cv_ms":data[1],"overlap_ms":data[2]}));
                    return Ok(());
                }
                if pass == 0 { reference.insert(key, data); }
                else {
                    let expected = &reference[&key];
                    // Reference salience includes reflect-padded frames. Compare
                    // only real frames supplied by GPU decode, not padding.
                    let padded_salience = a.compare_reference && name == "salience"
                        && data.len()%360 == 0 && expected.len() == (data.len()/360).div_ceil(32)*32*360;
                    if data.len() != expected.len() && !padded_salience { return Err(std::io::Error::other(format!("stage shape mismatch: {name}")).into()); }
                    let max = data.iter().zip(expected).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
                    let different = data.iter().zip(expected).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
                    if matches!(name, "audio16" | "next_convert_buffer") && different != 0 {
                        return Err(std::io::Error::other(format!("input timeline changed: {name}")).into());
                    }
                    differences.push(serde_json::json!({"block":block,"stage":name,"max_abs":max,"different":different,"len":data.len()}));
                }
                Ok(())
            })?;
        }
        if a.same_session { retained = Some(engine); }
    }
    std::fs::write(a.report, serde_json::to_vec_pretty(&serde_json::json!({
        "differences":differences,"parallel":parallel,"same_session":a.same_session,
        "gates":"finite, shape, unchanged input timeline, device overlap; other numeric differences are measurements, NOT quality acceptance"
    }))?)?;
    if a.gpu_assets.is_some() && !parallel.iter().any(|p| p["overlap_ms"].as_f64().is_some_and(|ms| ms>0.)) {
        return Err("no device overlap observed".into());
    }
    if a.same_session && differences.iter().any(|row| row["different"].as_u64() != Some(0)) {
        return Err("same-session reset/replay changed numerical output; report retained".into());
    }
    Ok(())
}
