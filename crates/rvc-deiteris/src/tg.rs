//! TG Fast state machine on Rust DSP and ONNX; old engine remains the reference adapter.
use crate::{engine::{floats, i64s, put_resample, run, Params, Values}, input::InputTimeline,
    mel::Mel, pitch::{decode, PitchState}, startup::{fade_windows, Dims, ResampleInputs, Startup}, Generator};
use anyhow::{ensure, Context, Result};
use half::f16;
use ort::{session::Session, value::{DynValue, Tensor}};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read, path::Path};

const POST_PORTS: &[&str] = &["generated:float32", "volume:float16", "active:bool", "sola_buffer:float32",
    "formant_kernel:float32", "formant_indices:int64", "formant_output_length:int64", "output_kernel:float32",
    "output_indices:int64", "output_length:int64", "fade_in:float32", "fade_out:float32", "gather_range:int64",
    "search_end:int64", "block_length:int64", "audio_model:float32", "pre_sola:float32", "sola_offset:int64",
    "output:float32", "next_sola_buffer:float32"];

pub fn validate_assets(root: &Path) -> Result<()> {
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("audio-contract.json"))?)?;
    ensure!(manifest["schema"] == 1 && manifest["algorithm"] == "tg-fast-v1" && manifest["generator_kind"] == "deiteris-onnx-v1" && manifest["tf32"] == false, "unsupported audio contract");
    ensure!(manifest["ports"]["contentvec"] == serde_json::json!(["audio:float16", "units9:float16", "unit12:float16", "unit12s:float16"])
        && manifest["ports"]["rmvpe"] == serde_json::json!(["mel:float16", "salience:float16"])
        && manifest["ports"]["post"] == serde_json::json!(POST_PORTS), "unsupported audio port contract");
    for name in ["contentvec.onnx", "rmvpe-salience.onnx", "post.onnx", "mel-basis.f32", "hann.f32"] {
        let mut file = std::fs::File::open(root.join(name))?;
        let mut digest = Sha256::new();
        let mut block = [0u8; 65536];
        loop { let n = file.read(&mut block)?; if n == 0 { break } digest.update(&block[..n]); }
        ensure!(manifest["files"][name].as_str() == Some(format!("{:x}", digest.finalize()).as_str()), "audio asset hash mismatch: {name}");
    }
    Ok(())
}

fn session(path: &Path, cuda: bool) -> Result<Session> {
    let provider = if cuda { ort::ep::CUDA::default().with_tf32(false).build() } else { ort::ep::CPU::default().build() };
    let session = Session::builder()?.with_intra_threads(1).map_err(|e| anyhow::anyhow!(e.to_string()))?
        .with_execution_providers([provider.error_on_failure()]).map_err(|e| anyhow::anyhow!(e.to_string()))?.commit_from_file(path)?;
    let expected: &[&str] = match path.file_name().and_then(|n|n.to_str()) {
        Some("contentvec.onnx") => &["audio:float16", "units9:float16", "unit12:float16", "unit12s:float16"],
        Some("rmvpe-salience.onnx") => &["mel:float16", "salience:float16"],
        Some("post.onnx") => POST_PORTS,
        _ => anyhow::bail!("unknown audio graph"),
    };
    use ort::value::TensorElementType as T;
    let actual = session.inputs().iter().chain(session.outputs()).map(|p| {
        let dtype = match p.dtype().tensor_type() { Some(T::Float32)=>"float32", Some(T::Float16)=>"float16", Some(T::Int64)=>"int64", Some(T::Bool)=>"bool", _=>"unsupported" };
        format!("{}:{dtype}", p.name())
    }).collect::<Vec<_>>();
    ensure!(actual == expected, "audio graph port/dtype mismatch: {}", path.display());
    Ok(session)
}

fn halves(shape: &[usize], data: Vec<f16>) -> Result<DynValue> { Ok(Tensor::from_array((shape.to_vec(), data))?.into_dyn()) }
fn append(cache: &mut [f16], values: &[f32]) {
    let count = values.len().min(cache.len());
    cache.copy_within(count.., 0);
    let start = cache.len()-count;
    for (out, value) in cache[start..].iter_mut().zip(&values[values.len()-count..]) { *out = f16::from_f32(*value); }
}

enum Analysis {
    Reference { mel: Mel, cv: Session, f0: Session },
    FastV2 { cv: crate::gpu_pitch::GpuFeatures, f0: crate::gpu_pitch::GpuPitch },
}

/// Isolated trial choices. The normal FastV2 constructor enables only the
/// proven CPU completion-wait optimization; audio-changing trials remain off.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Trial {
    pub full_pitch_context: bool,
    pub serial_analysis: bool,
    pub delayed_input: bool,
    pub precise_volume: bool,
    pub aligned_pitch_history: bool,
    pub blocking_analysis_wait: bool,
}
impl Trial {
    pub fn dimensions(&self, startup: &Startup, rate: usize) -> Result<Dims> {
        let mut dims = startup.tg_dims(rate)?;
        ensure!(!self.aligned_pitch_history || dims.block * 100 % startup.sample_rate == 0,
            "aligned pitch history PoC requires an exact 10ms input hop");
        // RVCr2.realloc: silenceFront=false. Do not change generator context,
        // output length, or the reference's whole-window pitch-cache writes.
        if self.full_pitch_context { dims.silence_front = 0; }
        Ok(dims)
    }
}

pub struct Engine {
    trial: Trial,
    startup: Startup,
    dims: Dims,
    input: InputTimeline,
    audio: Vec<f16>,
    context: Vec<f16>,
    pitch: PitchState,
    analysis: Analysis,
    generator: Generator,
    post: Session,
    post_values: Values,
    output: Vec<f32>,
    failed: bool,
}
impl Engine {
    /// CPU DSP / ordinary ORT reference adapter, retained for numeric comparisons.
    pub fn new(model: &Path, assets: &Path, startup: Startup, graph: bool) -> Result<Self> {
        Self::build(model, assets, startup, graph, None, Trial::default())
    }
    /// GPU normal RMVPE and ContentVec on independent streams. Input resampling,
    /// feature expansion, pitch cache and postprocessing remain CPU in this slice.
    pub fn new_fast_v2(model: &Path, assets: &Path, gpu_assets: &Path, runtime: &Path, startup: Startup, graph: bool) -> Result<Self> {
        // Fail before allocating sessions if the required bundle is absent.
        ensure!(gpu_assets.join("contract.json").is_file(), "missing Fast V2 GPU assets: {}", gpu_assets.display());
        Self::build(model, assets, startup, graph, Some((runtime, gpu_assets)),
            Trial { blocking_analysis_wait: true, ..Trial::default() })
    }
    #[cfg(feature = "evaluation")]
    pub fn new_trial(model: &Path, assets: &Path, gpu_assets: &Path, runtime: &Path,
        startup: Startup, graph: bool, trial: Trial) -> Result<Self> {
        ensure!(gpu_assets.join("contract.json").is_file(), "missing Fast V2 GPU assets: {}", gpu_assets.display());
        Self::build(model, assets, startup, graph, Some((runtime, gpu_assets)), trial)
    }
    fn build(model: &Path, assets: &Path, startup: Startup, graph: bool, gpu: Option<(&Path,&Path)>, trial: Trial) -> Result<Self> {
        validate_assets(assets)?;
        let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(model.join("model.json"))?)?;
        let rate = metadata["model_sr"].as_u64().context("missing model rate")? as usize;
        let dims = trial.dimensions(&startup, rate)?;
        let analysis = if let Some((runtime, seams)) = gpu {
            // Both constructors finish their capture before the next is created.
            let f0 = crate::gpu_pitch::GpuPitch::new(runtime, seams, dims.convert-dims.silence_front, graph)?;
            let cv = crate::gpu_pitch::GpuFeatures::new(runtime, &assets.join("contentvec.onnx"), dims.convert, graph)?;
            if trial.blocking_analysis_wait {
                f0.enable_blocking_wait()?;
                cv.enable_blocking_wait()?;
            }
            Analysis::FastV2 { cv, f0 }
        } else {
            Analysis::Reference { mel: Mel::new(assets)?, cv: session(&assets.join("contentvec.onnx"), true)?, f0: session(&assets.join("rmvpe-salience.onnx"), true)? }
        };
        let mut post_values = Values::new();
        let formant = ResampleInputs::new(dims.scaled_window, rate/100, dims.trimmed)?;
        let model_len = formant.output_length;
        put_resample(&mut post_values, "formant_", formant, "formant_output_length")?;
        put_resample(&mut post_values, "output_", ResampleInputs::new(rate, startup.sample_rate, model_len)?, "output_length")?;
        let (fade, inverse) = fade_windows(dims.crossfade)?;
        post_values.insert("fade_in".into(), floats(&[dims.crossfade], fade)?);
        post_values.insert("fade_out".into(), floats(&[dims.crossfade], inverse)?);
        post_values.insert("sola_buffer".into(), floats(&[dims.crossfade], vec![0.;dims.crossfade])?);
        post_values.insert("gather_range".into(), i64s(&[dims.block+dims.crossfade], (0..(dims.block+dims.crossfade) as i64).collect())?);
        post_values.insert("search_end".into(), i64s(&[1], vec![(dims.crossfade+dims.search) as i64])?);
        post_values.insert("block_length".into(), i64s(&[1], vec![dims.block as i64])?);
        let input = InputTimeline::new(startup.sample_rate, dims.block)?;
        #[cfg(feature = "evaluation")]
        let input = if trial.delayed_input { InputTimeline::new_delayed(startup.sample_rate, dims.block)? } else { input };
        Ok(Self { trial, input, audio: vec![f16::ZERO;dims.audio], context: vec![f16::ZERO;dims.convert],
            pitch: PitchState::new(dims.features)?, analysis, post: session(&assets.join("post.onnx"), false)?,
            generator: Generator::new_with_graph(&model.join("generator.onnx"), model, rate, graph)?,
            output: vec![0.;dims.block], dims, startup, post_values, failed: false })
    }
    pub fn block_frames(&self) -> usize { self.dims.block }
    pub fn dimensions(&self) -> &Dims { &self.dims }
    pub fn seed(&mut self, seed: u64) -> Result<()> { self.generator.seed(seed) }
    pub fn reset_stream_state(&mut self) -> Result<()> {
        ensure!(!self.failed, "failed engine must be recreated");
        self.input.reset(); self.audio.fill(f16::ZERO); self.context.fill(f16::ZERO); self.pitch.reset(); self.output.fill(0.);
        self.post_values.insert("sola_buffer".into(), floats(&[self.dims.crossfade], vec![0.;self.dims.crossfade])?);
        Ok(())
    }
    pub fn prepare(&mut self, seed: u64) -> Result<()> {
        let block: Vec<_> = (0..self.dims.block).map(|i| (std::f64::consts::TAU*180.*i as f64/self.startup.sample_rate as f64).sin() as f32*0.05).collect();
        for _ in 0..3 { self.process(&block, &Params { pitch: 0., threshold_db: -90. })?; }
        self.reset_stream_state()?;
        self.seed(seed)
    }
    pub fn process(&mut self, block: &[f32], params: &Params) -> Result<&[f32]> {
        self.process_impl(block, params, None, None)
    }
    pub fn process_observed(&mut self, block: &[f32], params: &Params, observe: &mut dyn FnMut(&str, &DynValue)->Result<()>) -> Result<&[f32]> {
        self.process_impl(block, params, Some(observe), None)
    }
    /// Diagnostic shared-noise boundary; leaves the product RNG untouched.
    pub fn process_with_noise(&mut self, block: &[f32], params: &Params, rnd: &[f16], sine: &[f16],
        observe: &mut dyn FnMut(&str, &DynValue)->Result<()>) -> Result<&[f32]> {
        self.process_impl(block, params, Some(observe), Some((rnd,sine)))
    }
    fn process_impl(&mut self, block: &[f32], params: &Params, observe: Option<&mut dyn FnMut(&str, &DynValue)->Result<()>>, noise: Option<(&[f16], &[f16])>) -> Result<&[f32]> {
        ensure!(!self.failed, "failed engine must be recreated");
        if let Err(error) = self.inner(block, params, observe, noise) { self.failed=true; return Err(error) }
        Ok(&self.output)
    }
    fn inner(&mut self, block: &[f32], params: &Params, mut observe: Option<&mut dyn FnMut(&str,&DynValue)->Result<()>>, noise: Option<(&[f16], &[f16])>) -> Result<()> {
        ensure!(params.threshold_db.is_finite() && (-120.0..=0.0).contains(&params.threshold_db) && params.pitch.is_finite() && (-48.0..=48.0).contains(&params.pitch), "invalid live parameters");
        let input = self.input.push(block)?;
        append(&mut self.audio, input); append(&mut self.context, input);
        if let Some(observe) = &mut observe {
            observe("audio16", &floats(&[input.len()], input.to_vec())?)?;
            observe("next_convert_buffer", &halves(&[self.context.len()], self.context.clone())?)?;
        }
        let volume = input_volume(&self.audio, self.trial.precise_volume);
        let active = volume.to_f32() >= 10f64.powf(params.threshold_db/20.) as f32;
        if let Some(observe) = &mut observe {
            observe("volume", &halves(&[], vec![volume])?)?;
            observe("active", &floats(&[], vec![if active { 1. } else { 0. }])?)?;
        }
        let (cv_data, raw) = match &mut self.analysis {
        Analysis::Reference { cv, f0, mel } => {
        let cv = run(cv, &BTreeMap::from([("audio".into(), halves(&[1,self.context.len()], self.context.clone())?)]))?;
        let cv = cv.get("unit12").context("missing ContentVec features")?;
        if let Some(observe) = &mut observe { observe("cv_features", cv)?; }
        let (_, cv_data) = cv.try_extract_tensor::<f16>()?;
        let cv_data = cv_data.to_vec();
        let (mel, real_frames, padded) = mel.extract(&self.context[self.dims.silence_front..])?;
        let mel = halves(&[1,128,padded], mel)?;
        if let Some(observe) = &mut observe { observe("mel", &mel)?; }
        let hidden = run(f0, &BTreeMap::from([("mel".into(), mel)]))?;
        let hidden = hidden.get("salience").context("missing RMVPE salience")?;
        if let Some(observe) = &mut observe { observe("salience", hidden)?; }
        let hidden = hidden.try_extract_tensor::<f16>()?.1;
        ensure!(hidden.len() == padded*360, "RMVPE frame mismatch");
        let raw = decode(&hidden[..real_frames*360], 0.05)?;
        (cv_data, raw)
        }
        Analysis::FastV2 { cv, f0 } => {
            // Upload both before any work: synchronous H2D must not serialize the
            // other stream. Each enqueue returns without draining its GPU stream.
            cv.upload(&self.context)?;
            f0.upload(&self.context[self.dims.silence_front..])?;
            let timing = observe.as_ref().map(|_| crate::gpu_pitch::ParallelProbe::new(f0, cv)).transpose()?;
            if let Some(timing) = &timing { timing.mark(0)?; }
            f0.enqueue()?;
            let serial_pitch = if self.trial.serial_analysis { Some(f0.finish()?.to_vec()) } else { None };
            if let Some(timing) = &timing { timing.mark(1)?; timing.mark(2)?; }
            cv.enqueue()?;
            if let Some(timing) = &timing { timing.mark(3)?; }
            let cv_data = cv.finish()?.to_vec();
            let raw = match serial_pitch { Some(raw) => raw, None => f0.finish()?.to_vec() };
            if let Some(observe) = &mut observe {
                observe("parallel_gpu_ms", &floats(&[3], timing.as_ref().unwrap().milliseconds()?)?)?;
                observe("cv_features", &halves(&[1,cv_data.len()/768,768],cv_data.clone())?)?;
                let values = f0.observe()?;
                observe("mel", &halves(&[1,128,values.mel.len()/128],values.mel)?)?;
                observe("salience", &halves(&[1,raw.len(),360],values.salience)?)?;
            }
            (cv_data, raw)
        }
        };
        let cv_frames = cv_data.len()/768;
        ensure!(cv_frames>0 && 2*(cv_frames+1)>=self.dims.features, "ContentVec frame mismatch");
        let mut features = Vec::with_capacity(self.dims.features*768);
        for frame in 0..self.dims.features { let from = (frame/2).min(cv_frames-1)*768; features.extend_from_slice(&cv_data[from..from+768]); }
        if let Some(observe) = &mut observe { observe("rawf0", &floats(&[raw.len()], raw.clone())?)?; }
        #[cfg(feature = "evaluation")]
        if self.trial.aligned_pitch_history {
            self.pitch.update_aligned(&raw, params.pitch, self.startup.formant, self.dims.block * 100 / self.startup.sample_rate)?;
        } else {
            self.pitch.update(&raw, params.pitch, self.startup.formant)?;
        }
        #[cfg(not(feature = "evaluation"))]
        self.pitch.update(&raw, params.pitch, self.startup.formant)?;
        let (pitch, pitchf) = self.pitch.generator_pitch(self.dims.features, (self.dims.formant_length as f64/self.dims.ret as f64) as f32)?;
        if let Some(observe) = &mut observe {
            observe("features", &halves(&[1,self.dims.features,768], features.clone())?)?;
            observe("pitch", &i64s(&[1,pitch.len()], pitch.to_vec())?)?;
            observe("pitchf", &halves(&[1,pitchf.len()], pitchf.clone())?)?;
        }
        let generated = if let Some((rnd,sine))=noise {
            self.generator.infer_with_noise(&features,pitch,&pitchf,self.dims.skip,self.dims.ret,self.dims.formant_length,rnd,sine)?
        } else { self.generator.infer(&features, pitch, &pitchf, self.dims.skip,self.dims.ret,self.dims.formant_length)? };
        if let Some(observe) = &mut observe { observe("generated", &floats(&[generated.len()],generated.clone())?)?; }
        self.post_values.insert("generated".into(), floats(&[generated.len()],generated)?);
        self.post_values.insert("volume".into(), halves(&[],vec![volume])?);
        self.post_values.insert("active".into(), Tensor::from_array((Vec::<usize>::new(),vec![active]))?.into_dyn());
        let mut post = run(&mut self.post, &self.post_values)?;
        if let Some(observe) = &mut observe { for (name,value) in &post { observe(name,value)?; } }
        let output = post["output"].try_extract_tensor::<f32>()?.1;
        ensure!(output.len()==self.dims.block && output.iter().all(|v|v.is_finite()), "invalid output block");
        self.output.copy_from_slice(output);
        let next = if active { post.remove("next_sola_buffer").context("missing SOLA state")? } else { floats(&[self.dims.crossfade],vec![0.;self.dims.crossfade])? };
        self.post_values.insert("sola_buffer".into(),next);
        Ok(())
    }
}

fn input_volume(audio: &[f16], precise: bool) -> f16 {
    #[cfg(feature = "evaluation")]
    if precise {
        // Keep only the final scalar in the existing fp16 postprocessing port.
        // Squaring quiet samples in fp16 loses energy before the reduction.
        let mean = audio.iter().map(|v| v.to_f32().powi(2)).sum::<f32>() / audio.len() as f32;
        return f16::from_f32(mean.sqrt());
    }
    #[cfg(not(feature = "evaluation"))]
    let _ = precise;
    let mean = audio.iter().map(|v| f16::from_f32(v.to_f32()*v.to_f32()).to_f32()).sum::<f32>() / audio.len() as f32;
    f16::from_f32(f16::from_f32(mean).to_f32().sqrt())
}

#[cfg(all(test, feature = "evaluation"))]
mod volume_tests {
    use super::*;

    #[test]
    fn quiet_signal_above_threshold_survives_energy_reduction() {
        let audio = vec![f16::from_f32(0.0001); 3040]; // -80 dBFS, above -83 dB threshold.
        assert_eq!(input_volume(&audio, false), f16::ZERO);
        let actual = input_volume(&audio, true).to_f32();
        assert!((actual - 0.0001).abs() < 0.0000001);
        assert!(actual >= 10f32.powf(-83. / 20.));
    }

    #[test]
    fn precision_change_preserves_silence_and_signal_below_threshold() {
        assert_eq!(input_volume(&[f16::ZERO; 3040], true), f16::ZERO);
        let quiet = vec![f16::from_f32(10f32.powf(-90. / 20.)); 3040];
        assert!(input_volume(&quiet, true).to_f32() < 10f32.powf(-83. / 20.));
    }

    #[test]
    fn measured_rms_tracks_independent_f64_oracle() {
        for db in [-100., -80., -76., -60., -30., -6.] {
            let audio: Vec<_> = (0..3040).map(|i| f16::from_f32(
                (std::f64::consts::TAU * 200. * i as f64 / 16000.).sin() as f32
                * 2f32.sqrt() * 10f32.powf(db / 20.))).collect();
            let expected = (audio.iter().map(|v| (v.to_f32() as f64).powi(2)).sum::<f64>() / audio.len() as f64).sqrt();
            assert!((input_volume(&audio, true).to_f32() as f64 / expected - 1.).abs() < 0.004);
        }
    }
}
