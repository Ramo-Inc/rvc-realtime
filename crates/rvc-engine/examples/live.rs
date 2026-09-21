//! Runs `Realtime` on a WASAPI input/output device pair and prints status and conversion time every second.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use rvc_engine::{list_devices, Devices, EngineOptions, F0Method, Realtime, RealtimeOptions, Startup, Status};

#[derive(Parser)]
struct Args {
    /// print WASAPI device names and exit
    #[arg(long)]
    list: bool,
    #[arg(long, default_value = "")]
    input: String,
    #[arg(long, default_value = "")]
    output: String,
    /// monitor output device (none when omitted)
    #[arg(long)]
    monitor: Option<String>,
    #[arg(long, default_value_t = 1.0)]
    monitor_volume: f32,
    #[arg(long, default_value_t = 60)]
    seconds: u64,
    #[arg(long)]
    exclusive: bool,
    #[arg(long, default_value_t = 48000)]
    sample_rate: u32,
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
    /// sweep the gate: raise the threshold by 1 dB per second while running (live-change check)
    #[arg(long, default_value_t = false)]
    sweep_threshold: bool,
    /// do not convert while the input is below the threshold
    #[arg(long, default_value_t = false)]
    skip_silence: bool,
    /// keep only speech in the context (silence never enters it)
    #[arg(long, default_value_t = false)]
    drop_silent_context: bool,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    if a.list {
        let devices = list_devices(&a.runtime_dir)?;
        println!("inputs:");
        devices.inputs.iter().for_each(|d| println!("  {} ({} Hz)", d.name, d.default_sample_rate));
        println!("outputs:");
        devices.outputs.iter().for_each(|d| println!("  {} ({} Hz)", d.name, d.default_sample_rate));
        return Ok(());
    }
    let f0 = match a.f0.as_str() {
        "rmvpe" => F0Method::Rmvpe,
        "fcpe" => F0Method::Fcpe,
        other => return Err(format!("unknown f0 method {other}").into()),
    };
    let startup = Startup { sample_rate: a.sample_rate, block_ms: a.block_ms, crossfade_ms: a.crossfade_ms, extra_ms: a.extra_ms, formant: a.formant, f0, variant: Default::default() };
    let rt = Realtime::start(
        a.model_dir,
        startup,
        Devices { input: a.input, output: a.output, monitor: a.monitor, wasapi_exclusive: a.exclusive },
        RealtimeOptions { engine: EngineOptions { runtime_dir: a.runtime_dir, seed: a.seed } },
    );
    rt.set_pitch(a.pitch);
    rt.set_rms_mix(a.rms_mix);
    rt.set_threshold_db(a.threshold_db);
    rt.set_drop_silent_context(a.drop_silent_context);
    rt.set_skip_silence(a.skip_silence);
    rt.set_monitor_volume(a.monitor_volume);
    let t = Instant::now();
    loop {
        if a.sweep_threshold {
            rt.set_threshold_db(a.threshold_db + t.elapsed().as_secs_f32());
        }
        println!("{:6.1}s {:?} infer_ms={:.2} {}", t.elapsed().as_secs_f64(), rt.status(), rt.infer_ms(), rt.status_text());
        let (blocks, late, mon, starved, missing, dropped) = rt.stats();
        let (overflow, underflow) = rt.glitches();
        if blocks > 0 {
            println!(
                "       blocks={blocks} late={late} ({:.1}%) monitor_callbacks={mon} dry={starved} ({:.1}%) missing_samples={missing} dropped_samples={dropped} in_overflow={overflow} out_underflow={underflow}",
                100.0 * late as f64 / blocks as f64,
                100.0 * starved as f64 / mon.max(1) as f64
            );
        }
        if rt.status() == Status::Error || t.elapsed() >= Duration::from_secs(a.seconds) {
            break;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    Ok(())
}
