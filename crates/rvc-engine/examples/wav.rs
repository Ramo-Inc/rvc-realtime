//! Converts a WAV block by block through `Engine` (same block handling as the PoC `rvc-poc wav`).

use std::path::PathBuf;

use clap::Parser;
use rvc_engine::{Engine, EngineOptions, F0Method, F0Window, Model, Params, Startup, Variant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../PoC/assets/onnx-dynamic/model11"))]
    model_dir: PathBuf,
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../PoC/01-rust-ort-official-rt/runtime"))]
    runtime_dir: PathBuf,
    #[arg(long, default_value_t = 60.0)]
    block_ms: f64,
    #[arg(long, default_value_t = 80.0)]
    crossfade_ms: f64,
    #[arg(long, default_value_t = 1000.0)]
    extra_ms: f64,
    #[arg(long, default_value_t = -0.2, allow_hyphen_values = true)]
    formant: f64,
    /// rmvpe | fcpe
    #[arg(long, default_value = "fcpe")]
    f0: String,
    #[arg(long, default_value_t = 14.0, allow_hyphen_values = true)]
    pitch: f32,
    #[arg(long, default_value_t = 0.5)]
    rms_mix: f32,
    /// silence gate in dB; -60 and below is off (official GUI)
    #[arg(long, default_value_t = -60.0, allow_hyphen_values = true)]
    threshold_db: f32,
    /// do not convert while the input is below the threshold
    #[arg(long, default_value_t = false)]
    skip_silence: bool,
    /// keep only speech in the context (silence never enters it)
    #[arg(long, default_value_t = false)]
    drop_silent_context: bool,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// leave unvoiced F0 frames at 0 (older official code, VCClient) instead of interpolating (official 2.3)
    #[arg(long, default_value_t = false)]
    no_f0_interp: bool,
    /// RMVPE voicing threshold (official 0.03, Deiteris VCClient 0.05)
    #[arg(long, default_value_t = 0.03)]
    rmvpe_threshold: f32,
    /// F0 window: official | head (also the crossfaded head of the output) | full (the whole buffer)
    #[arg(long, default_value = "official")]
    f0_window: String,
    /// crossfade the whole crossfade length instead of at most 40 ms
    #[arg(long, default_value_t = false)]
    full_crossfade: bool,
    #[arg(long = "in")]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    let mut reader = hound::WavReader::open(&a.input)?;
    let spec = reader.spec();
    let f0 = match a.f0.as_str() {
        "rmvpe" => F0Method::Rmvpe,
        "fcpe" => F0Method::Fcpe,
        other => return Err(format!("unknown f0 method {other}").into()),
    };
    let f0_window = match a.f0_window.as_str() {
        "official" => F0Window::Official,
        "head" => F0Window::Head,
        "full" => F0Window::Full,
        other => return Err(format!("unknown f0 window {other}").into()),
    };
    let variant = Variant { f0_interp: !a.no_f0_interp, rmvpe_threshold: a.rmvpe_threshold, f0_window, full_crossfade: a.full_crossfade };
    let startup = Startup { sample_rate: spec.sample_rate, block_ms: a.block_ms, crossfade_ms: a.crossfade_ms, extra_ms: a.extra_ms, formant: a.formant, f0, variant };
    let model = Model::open(&a.model_dir)?;
    let mut engine = Engine::new(&model, &startup, &EngineOptions { runtime_dir: a.runtime_dir, seed: a.seed })?;
    let params = Params { pitch: a.pitch, rms_mix: a.rms_mix, threshold_db: a.threshold_db, drop_silent_context: a.drop_silent_context, skip_silence: a.skip_silence };

    let ch = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = raw.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect();
    let bf = engine.block_frames();
    let blocks = mono.len().div_ceil(bf);

    let mut output = Vec::with_capacity(blocks * bf);
    let mut ms = Vec::with_capacity(blocks);
    for i in 0..blocks {
        let mut block = vec![0f32; bf];
        let end = ((i + 1) * bf).min(mono.len());
        block[..end - i * bf].copy_from_slice(&mono[i * bf..end]);
        let t = std::time::Instant::now();
        output.extend_from_slice(engine.process(&block, &params)?);
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    ms.sort_by(f64::total_cmp);
    let pct = |p: f64| ms[((ms.len() - 1) as f64 * p).round() as usize];

    if let Some(parent) = a.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = hound::WavWriter::create(&a.out, hound::WavSpec { channels: 1, sample_rate: spec.sample_rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float })?;
    for s in &output {
        w.write_sample(*s)?;
    }
    w.finalize()?;
    println!("{blocks} blocks of {bf} samples -> {}", a.out.display());
    // back-to-back, not at real-time pace: compare variants with each other, not with live block times
    println!("block_ms p50 {:.2} p95 {:.2}", pct(0.5), pct(0.95));
    Ok(())
}
