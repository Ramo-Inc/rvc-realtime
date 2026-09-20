//! Diagnostic only: paced file input, no GUI or audio devices. Results in target/debug-latency/.
use std::{collections::BTreeMap, path::PathBuf, time::{Duration, Instant}};
use clap::Parser;
use rvc_engine::{Conversion, EngineOptions, F0Method, F0Window, Params, Startup, Variant};

#[derive(Parser)]
struct Args {
    #[arg(long)] mode: String,
    #[arg(long)] voice: PathBuf,
    #[arg(long)] input: PathBuf,
    #[arg(long, default_value_t = 60)] chunk: usize,
    #[arg(long, default_value_t = 100.0)] block_ms: f64,
    #[arg(long, default_value_t = 3000.0)] extra_ms: f64,
    #[arg(long, default_value_t = 100.0)] crossfade_ms: f64,
    #[arg(long, default_value = "rmvpe")] f0: String,
    #[arg(long, default_value_t = 100)] blocks: usize,
    #[arg(long)] label: String,
    /// Alternate one second of silence and three seconds of speech; validate onset telemetry.
    #[arg(long)] onsets: bool,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let assets = root.join("assets/app");
    let runtime = root.join("assets/runtime");
    let dir = root.join("target/debug-latency");
    std::fs::create_dir_all(&dir)?;
    let report = dir.join(format!("{}.json", a.label));
    if report.exists() { return Err("report already exists".into()); }
    let mut reader = hound::WavReader::open(&a.input)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_format != hound::SampleFormat::Float { return Err("requires float mono".into()); }
    let samples = reader.samples::<f32>().collect::<Result<Vec<_>,_>>()?;
    let options = EngineOptions { runtime_dir: runtime.clone(), seed: 7 };
    let params = Params { pitch:14.0, threshold_db:-83.0, rms_mix:0.5, skip_silence:false, drop_silent_context:false };
    let startup = rvc_engine::DeiterisStartup { sample_rate:spec.sample_rate as usize, chunk:a.chunk,
        extra_ms:a.extra_ms, crossfade_ms:a.crossfade_ms, formant:-0.2 };
    let mut product = None;
    let mut direct = None;
    if a.mode == "legacy" {
        let cache = dir.join("legacy-model");
        if !cache.join("model.json").exists() { rvc_engine::voice_model::convert(&a.voice, &assets, &cache)?; }
        product = Some(Conversion::Legacy { model_dir:cache, startup:Startup {
            sample_rate:spec.sample_rate, block_ms:a.block_ms, crossfade_ms:a.crossfade_ms,
            extra_ms:a.extra_ms, formant:-0.2,
            f0:if a.f0 == "fcpe" {F0Method::Fcpe} else {F0Method::Rmvpe},
            variant:Variant {f0_interp:false, rmvpe_threshold:0.05, f0_window:F0Window::Head, full_crossfade:true},
        }}.load(&options)?);
    } else {
        let assets = assets.join("deiteris");
        let model = rvc_engine::voice_model::prepare_deiteris(&a.voice, &assets, &dir.join("deiteris-cache"))?;
        if a.mode == "direct" {
            std::env::set_var("PATH", format!("{};{}",runtime.display(),std::env::var("PATH")?));
            let meta:serde_json::Value = serde_json::from_slice(&std::fs::read(model.join("model.json"))?)?;
            let files = rvc_deiteris::engine::Assets { pre:assets.join("pre.onnx"), prepare:assets.join("prepare.onnx"),
                post:assets.join("post.onnx"), contentvec:assets.join("contentvec.onnx"), rmvpe:assets.join("rmvpe.onnx"),
                template:model.join("generator.onnx"), model_rate:meta["model_sr"].as_u64().unwrap() as usize };
            let mut engine = rvc_deiteris::engine::Engine::new_with_graph(&model,startup,&files,true).map_err(|e|e.to_string())?;
            engine.seed(7).map_err(|e|e.to_string())?;
            direct = Some(engine);
        } else if a.mode == "deiteris" {
            product = Some(Conversion::Deiteris {model_dir:model,assets,startup,cuda_graph:true}.load(&options)?);
        } else { return Err("unknown mode".into()); }
    }
    let frames = product.as_ref().map(|p|p.block_frames()).unwrap_or_else(||direct.as_ref().unwrap().block_frames());
    let period = Duration::from_secs_f64(frames as f64/spec.sample_rate as f64);
    let mut input = vec![0.0;frames];
    let mut times = Vec::new();
    let mut stages:BTreeMap<String,Vec<f64>> = BTreeMap::new();
    let mut epoch = Instant::now();
    let mut measurement_start_unix = 0.0;
    let clock = Instant::now();
    let mut onset = rvc_engine::onset::OnsetLatency::new(spec.sample_rate);
    let mut onset_measurements = Vec::new();
    let mut onset_blocks = Vec::new();
    for i in 0..a.blocks+20 {
        if i == 20 {
            epoch = Instant::now();
            measurement_start_unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
        }
        if i >= 20 {
            let arrival = epoch+period.mul_f64((i-20) as f64);
            if let Some(wait)=arrival.checked_duration_since(Instant::now()) {std::thread::sleep(wait);}
        }
        for (j,x) in input.iter_mut().enumerate() {
            let n=i*frames+j;
            *x=if a.onsets && n % (spec.sample_rate as usize*4) < spec.sample_rate as usize {0.0}
                else {samples[n%samples.len()]};
        }
        let t=Instant::now();
        let input_seen=clock.elapsed();
        if let Some(p)=product.as_mut() {
            let output=p.process(&input,&params)?;
            if a.onsets {
                if let Some(m)=onset.observe(&input,output,input_seen,clock.elapsed()) {
                    if i>=20 {
                        onset_measurements.push(m.elapsed_ms());
                        onset_blocks.push(serde_json::json!({"input_block":m.input_block,"output_block":m.output_block,
                            "input_received_ms":m.input_received.as_secs_f64()*1000.0,
                            "output_written_ms":m.output_written.as_secs_f64()*1000.0,"elapsed_ms":m.elapsed_ms()}));
                    }
                }
            }
        } else {
            let mut last=t;
            direct.as_mut().unwrap().process_observed(&input,&rvc_deiteris::engine::Params {pitch:14.0,threshold_db:-83.0}, &mut |name,_| {
                let stage=match name {"active"=>"pre", "cv_features"=>"contentvec", "rawf0"=>"rmvpe",
                    "features"=>"prepare", "generated"=>"generator", "next_sola_buffer"=>"post", _=>return Ok(())};
                let now=Instant::now();
                if i>=20 {stages.entry(stage.into()).or_default().push(now.duration_since(last).as_secs_f64()*1000.0);}
                last=now;
                Ok(())
            }).map_err(|e|e.to_string())?;
        }
        if i>=20 {times.push(t.elapsed().as_secs_f64()*1000.0);}
    }
    fn stats(v:&[f64])->serde_json::Value {
        let mut s=v.to_vec();s.sort_by(f64::total_cmp);
        let q=|p:f64|s[((s.len()-1) as f64*p).round() as usize];
        serde_json::json!({"p50":q(0.5),"p95":q(0.95),"max":q(1.0)})
    }
    let stage_stats:BTreeMap<_,_>=stages.iter().map(|(k,v)|(k,stats(v))).collect();
    let result=serde_json::json!({"mode":a.mode,"voice":a.voice,"sample_rate":spec.sample_rate,
        "period_ms":period.as_secs_f64()*1000.0,"extra_ms":a.extra_ms,"crossfade_ms":a.crossfade_ms,"f0":a.f0,
        "blocks":a.blocks,"measurement_start_unix":measurement_start_unix,
        "measurement_end_unix":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64(),
        "times":stats(&times),"stages":stage_stats,"times_ms":times,"onset_ms":onset_measurements,"onset_blocks":onset_blocks,
        "late_blocks":times.iter().filter(|&&x|x>period.as_secs_f64()*1000.0).count()});
    std::fs::write(&report,serde_json::to_vec_pretty(&result)?)?;
    println!("{}: {} stages={}",a.label,result["times"],result["stages"]);
    if a.onsets {
        println!("onset_ms={}",result["onset_ms"]);
        if onset_measurements.len()<2 {return Err("onset telemetry did not detect two utterances".into());}
    }
    Ok(())
}
