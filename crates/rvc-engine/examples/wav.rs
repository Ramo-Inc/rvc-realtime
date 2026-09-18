//! Converts a WAV block by block through `Engine`.

use std::path::PathBuf;

use clap::Parser;
use rvc_engine::{Engine, EngineOptions, F0Method, Model, Params, Startup};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/onnx-dynamic"))]
    model_dir: PathBuf,
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/runtime"))]
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
    let startup = Startup { sample_rate: spec.sample_rate, block_ms: a.block_ms, crossfade_ms: a.crossfade_ms, extra_ms: a.extra_ms, formant: a.formant, f0 };
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
    for i in 0..blocks {
        let mut block = vec![0f32; bf];
        let end = ((i + 1) * bf).min(mono.len());
        block[..end - i * bf].copy_from_slice(&mono[i * bf..end]);
        output.extend_from_slice(engine.process(&block, &params)?);
    }

    if let Some(parent) = a.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = hound::WavWriter::create(&a.out, hound::WavSpec { channels: 1, sample_rate: spec.sample_rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float })?;
    for s in &output {
        w.write_sample(*s)?;
    }
    w.finalize()?;
    println!("{blocks} blocks of {bf} samples -> {}", a.out.display());
    Ok(())
}
