//! Headless product engine check: no window, microphone, PortAudio or audio device.
use std::{path::PathBuf, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};
use clap::Parser;
use rvc_engine::{Conversion, DeiterisStartup, EngineOptions, Params};

#[derive(Parser)]
struct Args {
    #[arg(long)] voice: PathBuf,
    #[arg(long)] assets: PathBuf,
    #[arg(long)] runtime: PathBuf,
    #[arg(long = "in")] input: PathBuf,
    #[arg(long)] out: PathBuf,
    #[arg(long, default_value_t = 19)] chunk: usize,
    #[arg(long, default_value_t = 500.0)] extra_ms: f64,
    #[arg(long, default_value_t = 100.0)] crossfade_ms: f64,
    #[arg(long, default_value_t = -0.2, allow_hyphen_values = true)] formant: f64,
    #[arg(long, default_value_t = 14.0, allow_hyphen_values = true)] pitch: f32,
    #[arg(long, default_value_t = -90.0, allow_hyphen_values = true)] threshold_db: f32,
    #[arg(long, default_value_t = 1)] seed: u64,
    #[arg(long)] uncaptured: bool,
    #[cfg(feature = "quality-evaluation")]
    #[arg(long, conflicts_with_all = ["full_gpu", "compare_direct"])] full_pitch_context: bool,
    #[cfg(feature = "quality-evaluation")]
    #[arg(long, conflicts_with_all = ["full_gpu", "compare_direct"])] serial_analysis: bool,
    #[cfg(feature = "quality-evaluation")]
    #[arg(long, conflicts_with_all = ["full_gpu", "compare_direct", "full_pitch_context", "serial_analysis"])] delayed_input: bool,
    #[cfg(feature = "quality-evaluation")]
    #[arg(long, conflicts_with_all = ["full_gpu", "compare_direct", "full_pitch_context", "serial_analysis", "delayed_input"])] precise_volume: bool,
    #[cfg(feature = "quality-evaluation")]
    #[arg(long, conflicts_with_all = ["full_gpu", "compare_direct", "full_pitch_context", "serial_analysis", "delayed_input", "precise_volume"])] aligned_pitch_history: bool,
    #[cfg(feature = "quality-evaluation")]
    #[arg(long, conflicts_with_all = ["full_gpu", "compare_direct", "full_pitch_context", "serial_analysis", "delayed_input", "precise_volume", "aligned_pitch_history"])] blocking_analysis_wait: bool,
    /// Pre-optimization C1 wait, for regression measurements only.
    #[cfg(feature = "quality-evaluation")]
    #[arg(long, conflicts_with_all = ["full_gpu", "compare_direct", "full_pitch_context", "serial_analysis", "delayed_input", "precise_volume", "aligned_pitch_history", "blocking_analysis_wait"])] stream_analysis_wait: bool,
    /// Evaluate C2 through the same owner; unavailable in normal app builds.
    #[cfg(feature = "quality-evaluation")]
    #[arg(long, conflicts_with = "compare_direct")] full_gpu: bool,
    #[arg(long)] max_blocks: Option<usize>,
    #[arg(long, requires = "pitch_after")] pitch_change_at: Option<usize>,
    #[arg(long, requires = "pitch_change_at", allow_hyphen_values = true)] pitch_after: Option<f32>,
    /// Compare thread handoff with direct ONNX calls under identical seeds; not upstream quality acceptance.
    #[arg(long, conflicts_with = "paced_seconds")] compare_direct: bool,
    /// Process an alternate startup once, drop it, then start the requested configuration.
    #[arg(long, conflicts_with = "paced_seconds")] prime: bool,
    /// Real arrival cadence after 20 measurement-only warmup blocks.
    #[arg(long, conflicts_with_all = ["pitch_change_at", "max_blocks"])] paced_seconds: Option<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    let report = a.out.with_extension("json");
    if a.out.exists() || report.exists() { return Err("output/report must be new".into()); }
    if a.paced_seconds.is_some_and(|v| !(60..=120).contains(&v)) { return Err("paced duration must be 60..120 seconds".into()); }
    let mut reader = hound::WavReader::open(&a.input)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_format != hound::SampleFormat::Float {
        return Err("requires mono float32 WAV".into());
    }
    let samples = reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?;
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) { return Err("invalid input".into()); }
    let startup = DeiterisStartup { sample_rate: spec.sample_rate as usize, chunk: a.chunk,
        block_frames: None,
        extra_ms: a.extra_ms, crossfade_ms: a.crossfade_ms, formant: a.formant };
    let options = EngineOptions { runtime_dir: a.runtime.clone(), seed: a.seed };
    let model_dir = rvc_engine::voice_model::prepare_deiteris(&a.voice, &a.assets,
        &a.out.with_extension("model-cache"))?;
    #[cfg(feature = "quality-evaluation")]
    let model_rate = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(model_dir.join("model.json"))?
    )?["model_sr"].as_u64().ok_or("missing model rate")? as usize;
    let load = |startup: DeiterisStartup| {
        #[cfg(feature = "quality-evaluation")]
        if a.full_pitch_context || a.serial_analysis || a.delayed_input || a.precise_volume || a.aligned_pitch_history || a.blocking_analysis_wait || a.stream_analysis_wait { return Conversion::DeiterisTrial {
            model_dir: model_dir.clone(), assets: a.assets.clone(), startup, cuda_graph: !a.uncaptured,
            trial: rvc_deiteris::tg::Trial { full_pitch_context: a.full_pitch_context, serial_analysis: a.serial_analysis, delayed_input:a.delayed_input, precise_volume:a.precise_volume, aligned_pitch_history:a.aligned_pitch_history, blocking_analysis_wait:a.blocking_analysis_wait },
        }.load(&options); }
        #[cfg(feature = "quality-evaluation")]
        if a.full_gpu { return Conversion::DeiterisGpu {
            model_dir: model_dir.clone(), assets: a.assets.clone(), startup, cuda_graph: !a.uncaptured,
        }.load(&options); }
        Conversion::Deiteris {
        model_dir: model_dir.clone(), assets: a.assets.clone(), startup, cuda_graph: !a.uncaptured,
        }.load(&options)
    };
    let params = |i| Params {
        pitch: if a.pitch_change_at.is_some_and(|at| i >= at) { a.pitch_after.unwrap() } else { a.pitch },
        threshold_db: a.threshold_db, rms_mix: 1.0, skip_silence: false, drop_silent_context: false,
    };
    if a.prime {
        let mut previous = startup.clone();
        previous.chunk = if a.chunk == 17 { 19 } else { 17 };
        #[cfg(feature = "quality-evaluation")]
        if a.aligned_pitch_history {
            // Keep both startup configurations inside this PoC's exact-hop domain.
            previous.chunk = if a.chunk == 15 { 60 } else { 15 };
        }
        previous.formant = 0.4;
        let mut engine = load(previous)?;
        let mut input = vec![0.0; engine.block_frames()];
        let n = input.len().min(samples.len());
        input[..n].copy_from_slice(&samples[..n]);
        engine.process(&input, &params(0))?;
        drop(engine);
    }
    let loading = Instant::now();
    let mut engine = load(startup.clone())?;
    let load_ms = loading.elapsed().as_secs_f64() * 1000.0;
    let frames = engine.block_frames();
    let period = Duration::from_secs_f64(frames as f64 / spec.sample_rate as f64);
    let source_blocks = samples.len().div_ceil(frames);
    let count = a.paced_seconds.map_or_else(
        || a.max_blocks.unwrap_or(source_blocks).min(source_blocks),
        |seconds| (seconds as f64 / period.as_secs_f64()).ceil() as usize,
    );
    if count == 0 { return Err("empty run".into()); }
    let mut input = vec![0.0; frames];
    let fill = |i: usize, input: &mut [f32]| {
        input.fill(0.0);
        let offset = (i % source_blocks) * frames;
        let n = frames.min(samples.len() - offset);
        input[..n].copy_from_slice(&samples[offset..offset+n]);
    };
    let mut warmup = Vec::new();
    if a.paced_seconds.is_some() {
        for i in 0..20 {
            fill(i, &mut input);
            let t = Instant::now();
            engine.process(&input, &params(i))?;
            warmup.push(t.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let mut output = Vec::with_capacity(count * frames);
    let mut times = Vec::with_capacity(count);
    let mut completion = Vec::with_capacity(count);
    let measurement_start_unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
    let epoch = Instant::now();
    for i in 0..count {
        let arrival = epoch + period.mul_f64(i as f64);
        if a.paced_seconds.is_some() {
            if let Some(wait) = arrival.checked_duration_since(Instant::now()) { std::thread::sleep(wait); }
        }
        fill(i, &mut input);
        let t = Instant::now();
        let converted = engine.process(&input, &params(i))?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        completion.push(Instant::now().saturating_duration_since(arrival).as_secs_f64() * 1000.0);
        output.extend_from_slice(converted);
    }
    drop(engine); // owner and singleton released before the direct/restart check
    let mut agreement = serde_json::Value::Null;
    if a.compare_direct {
        let converted = rvc_engine::voice_model::prepare_deiteris(&a.voice, &a.assets,
            &a.out.with_extension("model-cache"))?;
        let mut direct = rvc_deiteris::tg::Engine::new_fast_v2(&converted,
            &a.assets.join("tg-fast-v1"), &a.assets.join("tg-gpu-pitch-v2"),
            &a.runtime.canonicalize()?, startup.clone(), !a.uncaptured)?;
        direct.prepare(a.seed)?;
        let mut signal = 0.0f64;
        let mut error = 0.0f64;
        let mut exact = true;
        for i in 0..count {
            fill(i, &mut input);
            let p = params(i);
            let expected = direct.process(&input, &rvc_deiteris::engine::Params { pitch: p.pitch as f64, threshold_db: p.threshold_db as f64 })?;
            for (&x, &y) in output[i*frames..(i+1)*frames].iter().zip(expected) {
                signal += (y as f64).powi(2);
                error += (x as f64 - y as f64).powi(2);
                exact &= x.to_bits() == y.to_bits();
            }
        }
        let snr = if error == 0.0 { None } else { Some(10.0 * (signal / error).log10()) };
        agreement = serde_json::json!({"exact":exact, "snr_db":snr, "passed":exact || snr.is_some_and(|v| v >= 25.0)});
    }
    let mut writer = hound::WavWriter::create(&a.out, spec)?;
    for &value in &output { writer.write_sample(value)?; }
    writer.finalize()?;
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let q = |p:f64| sorted[((sorted.len()-1) as f64 * p).ceil() as usize];
    let period_ms = period.as_secs_f64() * 1000.0;
    let late = times.iter().filter(|&&t| t > period_ms).count();
    let deadline_misses = completion.iter().filter(|&&t| t > period_ms).count();
    let acceptance = a.paced_seconds.map(|_| late == 0 && deadline_misses == 0);
    let core = "C1: tg-fast-v2";
    #[cfg(feature = "quality-evaluation")]
    let core = if a.full_gpu { "C2: tg-full-gpu" } else { core };
    let mut result = serde_json::json!({
        "core":core,
        "scope":"product Processor, same owner path as Realtime; no GUI or audio devices",
        "voice":a.voice,"input":a.input,"startup":startup,"seed":a.seed,"cuda_graph":!a.uncaptured,
        "pitch":a.pitch,"threshold_db":a.threshold_db,"block_frames":frames,
        "source_samples":samples.len(),"source_blocks":source_blocks,
        "warmup_blocks":warmup.len(),"warmup_ms":warmup,
        "input_policy":"zero-pad final block; paced runs repeat source blocks",
        "prime":a.prime,"pitch_change_at":a.pitch_change_at,"pitch_after":a.pitch_after,
        "frames":output.len(),"blocks":count,"finite":output.iter().all(|v|v.is_finite()),
        "load_ms":load_ms,"first_call_ms":warmup.first().copied().unwrap_or(times[0]),
        "measurement_start_unix":measurement_start_unix,
        "paced_seconds":a.paced_seconds,"period_ms":period_ms,"p50_ms":q(0.5),"p95_ms":q(0.95),"p99_ms":q(0.99),"max_ms":q(1.0),
        "late_blocks":late,"arrival_deadline_misses":if a.paced_seconds.is_some(){Some(deadline_misses)}else{None},
        "max_arrival_completion_ms":if a.paced_seconds.is_some(){completion.iter().copied().reduce(f64::max)}else{None},
        "realtime_passed":acceptance,"direct_owner_agreement":agreement,"times_ms":times
    });
    result["analysis_wait"] = serde_json::json!("blocking_event");
    #[cfg(feature = "quality-evaluation")]
    {
        // Old audio trials deliberately retain their original wait configuration.
        result["analysis_wait"] = serde_json::json!(if a.full_gpu { "gpu_pipeline" }
            else if a.stream_analysis_wait || a.full_pitch_context || a.serial_analysis || a.delayed_input || a.precise_volume || a.aligned_pitch_history { "stream" }
            else { "blocking_event" });
        result["trial"] = serde_json::json!({"full_pitch_context":a.full_pitch_context,"serial_analysis":a.serial_analysis,"delayed_input":a.delayed_input,"precise_volume":a.precise_volume,"aligned_pitch_history":a.aligned_pitch_history,"blocking_analysis_wait":a.blocking_analysis_wait});
        result["analysis_dimensions"] = serde_json::to_value(rvc_deiteris::tg::Trial {
            full_pitch_context:a.full_pitch_context, serial_analysis:a.serial_analysis,
            delayed_input:a.delayed_input,
            precise_volume:a.precise_volume,
            aligned_pitch_history:a.aligned_pitch_history,
            blocking_analysis_wait:a.blocking_analysis_wait,
        }.dimensions(&startup, model_rate)?)?;
    }
    std::fs::write(&report, serde_json::to_vec_pretty(&result)?)?;
    println!("blocks={count} p50={:.3} p99={:.3} late={late} report={}", q(0.5),q(0.99),report.display());
    if acceptance == Some(false) || result["direct_owner_agreement"]["passed"] == false {
        return Err("acceptance failed; WAV and report retained".into());
    }
    Ok(())
}
