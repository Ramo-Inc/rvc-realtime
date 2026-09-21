//! Headless full-GPU vs hybrid arithmetic with identical explicit generator noise.
use anyhow::{ensure, Result};
use clap::Parser;
use half::f16;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    assets: PathBuf,
    #[arg(long)]
    pipeline_assets: PathBuf,
    #[arg(long)]
    runtime: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    uncaptured: bool,
    #[arg(long, default_value_t = 90)]
    max_blocks: usize,
    #[arg(long, default_value_t = 19)]
    chunk: usize,
    #[arg(long, default_value_t = 500.)]
    extra_ms: f64,
    #[arg(long, default_value_t = 100.)]
    crossfade_ms: f64,
}
fn samples(v: &ort::value::DynValue) -> Result<Vec<f32>> {
    if let Ok((_, x)) = v.try_extract_tensor::<f32>() {
        return Ok(x.to_vec());
    }
    if let Ok((_, x)) = v.try_extract_tensor::<f16>() {
        return Ok(x.iter().map(|v| v.to_f32()).collect());
    }
    if let Ok((_, x)) = v.try_extract_tensor::<i64>() {
        return Ok(x.iter().map(|&v| v as f32).collect());
    }
    anyhow::bail!("unsupported diagnostic tensor")
}
fn metric(reference: &[f32], candidate: &[f32]) -> Result<serde_json::Value> {
    ensure!(
        reference.len() == candidate.len()
            && reference.iter().chain(candidate).all(|v| v.is_finite()),
        "shape/finite mismatch"
    );
    let signal = reference.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
    let error = reference
        .iter()
        .zip(candidate)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>();
    let exact = reference
        .iter()
        .zip(candidate)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let snr = if signal > 0. && error > 0. {
        Some(10. * (signal / error).log10())
    } else {
        None
    };
    Ok(serde_json::json!({"samples":reference.len(),"exact":exact,
        "max_abs":reference.iter().zip(candidate).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max),
        "signal":signal,"error":error,"snr_db":snr,
        "pass_25db":error==0. || snr.is_some_and(|v|v>=25.)}))
}

fn replay(
    gpu: &mut rvc_deiteris::gpu_pipeline::GpuPipeline,
    input: &[f32],
    observe: bool,
) -> Result<(Vec<f32>, BTreeMap<String, Vec<f32>>)> {
    let b = gpu.block_frames();
    let mut result = Vec::new();
    let mut stages: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for part in input.chunks(b).take(12) {
        let mut block = vec![0.; b];
        block[..part.len()].copy_from_slice(part);
        result.extend_from_slice(gpu.process(
            &block,
            &rvc_deiteris::engine::Params {
                pitch: 14.,
                threshold_db: -90.,
            },
        )?);
        if observe {
            for (name, value) in gpu.observe()? {
                stages.entry(name).or_default().extend(samples(&value)?);
            }
        }
    }
    Ok((result, stages))
}

fn lifecycle(
    gpu: &mut rvc_deiteris::gpu_pipeline::GpuPipeline,
    input: &[f32],
    rnd_len: usize,
    sine_len: usize,
) -> Result<serde_json::Value> {
    gpu.reset_stream_state()?;
    gpu.seed(2026)?;
    let (baseline, baseline_stages) = replay(gpu, input, true)?;
    gpu.reset_stream_state()?;
    gpu.seed(2026)?;
    let (observed, observed_stages) = replay(gpu, input, true)?;
    let repeat = metric(&baseline, &observed)?;
    let mut stage_repeat = BTreeMap::new();
    for (name, values) in &baseline_stages {
        stage_repeat.insert(name.clone(), metric(values, &observed_stages[name])?);
    }
    gpu.reset_stream_state()?;
    gpu.seed(2026)?;
    let (unobserved, _) = replay(gpu, input, false)?;
    let observer_invariant = metric(&baseline, &unobserved)?;
    gpu.reset_stream_state()?;
    gpu.seed(2026)?;
    let zero = vec![0.; gpu.block_frames()];
    let muted = gpu
        .process_with_noise(
            &zero,
            &rvc_deiteris::engine::Params {
                pitch: 14.,
                threshold_db: 0.,
            },
            &vec![0.; rnd_len],
            &vec![0.; sine_len],
        )?
        .to_vec();
    let state = gpu.observe()?;
    let mute_zero = muted.iter().all(|v| *v == 0.)
        && samples(&state["next_sola_buffer"])?
            .iter()
            .all(|v| *v == 0.);
    gpu.reset_stream_state()?; // Deliberately do not reseed after diagnostic noise.
    let (after_diagnostic, after_stages) = replay(gpu, input, true)?;
    let noise_invariant = metric(&baseline, &after_diagnostic)?;
    let mut rng_repeat = BTreeMap::new();
    for name in ["rnd", "sine_noise"] {
        rng_repeat.insert(name, metric(&baseline_stages[name], &after_stages[name])?);
    }
    let pass = repeat["exact"] == true
        && observer_invariant["exact"] == true
        && noise_invariant["exact"] == true
        && rng_repeat.values().all(|value| value["exact"] == true)
        && stage_repeat.values().all(|value| value["exact"] == true)
        && mute_zero;
    Ok(
        serde_json::json!({"pass":pass,"reset_seed":repeat,"observer_invariant":observer_invariant,"stage_repeat":stage_repeat,"diagnostic_noise_output_invariant":noise_invariant,"rng_tensor_invariant":rng_repeat,"mute_output_and_state_zero":mute_zero}),
    )
}
fn main() -> Result<()> {
    let a = Args::parse();
    let report = a.out.with_extension("json");
    ensure!(
        !a.out.exists() && !report.exists(),
        "output/report must be new"
    );
    let runtime = a.runtime.canonicalize()?;
    let old = std::env::var_os("PATH").unwrap_or_default();
    std::env::set_var(
        "PATH",
        std::env::join_paths(std::iter::once(runtime.clone()).chain(std::env::split_paths(&old)))?,
    );
    let mut reader = hound::WavReader::open(&a.input)?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1 && spec.sample_format == hound::SampleFormat::Float,
        "requires mono float WAV"
    );
    let input = reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?;
    ensure!(
        !input.is_empty() && a.max_blocks > 0,
        "requires nonempty input and positive block limit"
    );
    let startup = rvc_deiteris::startup::Startup {
        sample_rate: spec.sample_rate as usize,
        chunk: a.chunk,
        block_frames: None,
        extra_ms: a.extra_ms,
        crossfade_ms: a.crossfade_ms,
        formant: -0.2,
    };
    let mut reference = rvc_deiteris::tg::Engine::new_fast_v2(
        &a.model,
        &a.assets.join("tg-fast-v1"),
        &a.assets.join("tg-gpu-pitch-v2"),
        &runtime,
        startup.clone(),
        !a.uncaptured,
    )?;
    reference.prepare(7)?;
    let mut gpu = rvc_deiteris::gpu_pipeline::GpuPipeline::new(
        &a.model,
        &a.assets,
        &a.pipeline_assets,
        &runtime,
        startup.clone(),
        !a.uncaptured,
    )?;
    gpu.prepare(7)?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(a.model.join("model.json"))?)?;
    let rate = metadata["model_sr"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("missing rate"))? as usize;
    let d = startup.tg_dims(rate)?;
    let mut rng = StdRng::seed_from_u64(7);
    let mut rows = Vec::new();
    let mut expected_wave = Vec::new();
    let mut actual_wave = Vec::new();
    let mut blocks_pass = true;
    for (i, part) in input.chunks(d.block).take(a.max_blocks).enumerate() {
        let mut block = vec![0.; d.block];
        block[..part.len()].copy_from_slice(part);
        let params = rvc_deiteris::engine::Params {
            pitch: if i < 24 { 14. } else { 7. },
            threshold_db: -90.,
        };
        let rnd: Vec<f32> = (0..192 * (d.features - d.skip.saturating_sub(24)))
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let sine: Vec<f32> = (0..d.ret * (rate / 100))
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let mut observed = BTreeMap::new();
        let expected = reference
            .process_with_noise(
                &block,
                &params,
                &rnd.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>(),
                &sine.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>(),
                &mut |name, v| {
                    observed.insert(name.to_string(), samples(v)?);
                    Ok(())
                },
            )?
            .to_vec();
        let t = std::time::Instant::now();
        let actual = gpu
            .process_with_noise(&block, &params, &rnd, &sine)?
            .to_vec();
        let ms = t.elapsed().as_secs_f64() * 1000.;
        let stages = gpu.observe()?;
        let mut differences = BTreeMap::new();
        for (name, v) in stages {
            if let Some(expected) = observed.get(&name) {
                let got = samples(&v)?;
                differences.insert(name, metric(expected, &got)?);
            }
        }
        let wave = metric(&expected, &actual)?;
        blocks_pass &= wave["pass_25db"] == true;
        expected_wave.extend(expected);
        actual_wave.extend(actual);
        rows.push(
            serde_json::json!({"block":i,"gpu_ms_diagnostic":ms,"wave":wave,"stages":differences}),
        );
        std::fs::write(
            &report,
            serde_json::to_vec_pretty(
                &serde_json::json!({"complete":false,"pass":false,"rows":rows}),
            )?,
        )?;
    }
    let total = metric(&expected_wave, &actual_wave)?;
    let lifecycle = lifecycle(
        &mut gpu,
        &input,
        192 * (d.features - d.skip.saturating_sub(24)),
        d.ret * (rate / 100),
    )?;
    let passed = blocks_pass && total["pass_25db"] == true && lifecycle["pass"] == true;
    let mut writer = hound::WavWriter::create(&a.out, spec)?;
    for value in actual_wave {
        writer.write_sample(value)?;
    }
    writer.finalize()?;
    drop(gpu);
    drop(reference);
    std::fs::write(
        report,
        serde_json::to_vec_pretty(
            &serde_json::json!({"scope":"full GPU vs hybrid shared-noise diagnostic; NOT actual TG JIT acceptance", "complete":true,"pass":passed,"cuda_graph":!a.uncaptured,"wave":total,"lifecycle":lifecycle,"rows":rows}),
        )?,
    )?;
    ensure!(passed, "shared-noise waveform gate failed; report retained");
    Ok(())
}
