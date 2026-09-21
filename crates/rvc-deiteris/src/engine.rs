//! Stateful pinned Deiteris block conversion; shared with the PoC verifier.
use crate::startup::{fade_windows, Dims, ResampleInputs, Startup};
use anyhow::{ensure, Context, Result};
use crate::Generator;
use half::f16;
use ort::{
    ep,
    session::{Session, SessionInputValue},
    value::{DynValue, Tensor},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub struct Assets {
    pub pre: PathBuf,
    pub prepare: PathBuf,
    pub post: PathBuf,
    pub contentvec: PathBuf,
    pub rmvpe: PathBuf,
    pub template: PathBuf,
    pub model_rate: usize,
}
pub struct Params {
    pub pitch: f64,
    pub threshold_db: f64,
}
pub(crate) type Values = BTreeMap<String, DynValue>;
pub struct Engine {
    startup: Startup,
    dims: Dims,
    generator: Generator,
    pre: Session,
    cv: Session,
    f0: Session,
    prepare: Session,
    post: Session,
    pre_values: Values,
    prepare_values: Values,
    post_values: Values,
    output: Vec<f32>,
    failed: bool,
}
pub(crate) fn session(path: &Path, cuda: bool) -> Result<Session> {
    let provider = if cuda {
        ep::CUDA::default().build()
    } else {
        ep::CPU::default().build()
    };
    Ok(Session::builder()?
        // CUDA owns the heavy arithmetic. Five default CPU worker pools spin
        // concurrently between calls and contend with the native generator.
        .with_intra_threads(1)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .with_execution_providers([provider.error_on_failure()])
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .commit_from_file(path)?)
}
pub(crate) fn run(session: &mut Session, values: &Values) -> Result<Values> {
    let ports = session
        .inputs()
        .iter()
        .map(|i| i.name().to_owned())
        .collect::<Vec<_>>();
    let inputs = ports
        .iter()
        .map(|name| {
            Ok((
                name.as_str(),
                SessionInputValue::from(
                    values
                        .get(name)
                        .with_context(|| format!("missing {name}"))?,
                ),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(session
        .run(inputs)?
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect())
}
pub(crate) fn i64s(shape: &[usize], values: Vec<i64>) -> Result<DynValue> {
    Ok(Tensor::from_array((shape.to_vec(), values))?.into_dyn())
}
pub(crate) fn floats(shape: &[usize], values: Vec<f32>) -> Result<DynValue> {
    Ok(Tensor::from_array((shape.to_vec(), values))?.into_dyn())
}
fn half_zeros(count: usize) -> Result<DynValue> {
    Ok(Tensor::from_array(([count], vec![f16::ZERO; count]))?.into_dyn())
}
pub(crate) fn put_resample(
    values: &mut Values,
    prefix: &str,
    r: ResampleInputs,
    length_name: &str,
) -> Result<()> {
    values.insert(
        format!("{prefix}kernel"),
        floats(&[r.phases, r.taps], r.kernel)?,
    );
    values.insert(
        format!("{prefix}indices"),
        i64s(&[r.frames, r.taps], r.indices)?,
    );
    values.insert(
        length_name.into(),
        i64s(&[1], vec![r.output_length as i64])?,
    );
    Ok(())
}
fn take(values: &mut Values, name: &str) -> Result<DynValue> {
    values
        .remove(name)
        .with_context(|| format!("missing {name}"))
}

impl Engine {
    pub fn new(model: &Path, startup: Startup, assets: &Assets) -> Result<Self> {
        Self::create(model, startup, assets, None)
    }
    pub fn new_with_graph(model: &Path, startup: Startup, assets: &Assets, cuda_graph: bool) -> Result<Self> {
        Self::create(model, startup, assets, Some(cuda_graph))
    }
    fn create(model: &Path, startup: Startup, assets: &Assets, cuda_graph: Option<bool>) -> Result<Self> {
        let dims = startup.dims(assets.model_rate)?;
        let pre = session(&assets.pre, true)?;
        let cv = session(&assets.contentvec, true)?;
        let f0 = session(&assets.rmvpe, true)?;
        let prepare = session(&assets.prepare, true)?;
        // Same graph on CPU: 327+54+55 isolated blocks preserve all SOLA offsets.
        let post = session(&assets.post, false)?;
        let generator = match cuda_graph {
            Some(enabled) => Generator::new_with_graph(&assets.template, model, assets.model_rate, enabled)?,
            None => Generator::new(&assets.template, model, assets.model_rate)?,
        };
        let mut pre_values = Values::new();
        pre_values.insert("audio_buffer".into(), half_zeros(dims.audio)?);
        pre_values.insert("convert_buffer".into(), half_zeros(dims.convert)?);
        put_resample(
            &mut pre_values,
            "",
            ResampleInputs::new(startup.sample_rate, 16000, dims.block)?,
            "output_length",
        )?;
        pre_values.insert(
            "silence_front".into(),
            i64s(&[1], vec![dims.silence_front as i64])?,
        );
        let frames = (dims.convert - dims.silence_front) / 160 + 1;
        let indices = (0..frames)
            .flat_map(|i| (0..1024).map(move |j| (i * 160 + j) as i64))
            .collect();
        pre_values.insert("stft_indices".into(), i64s(&[frames, 1024], indices)?);
        let mut prepare_values = Values::new();
        prepare_values.insert(
            "pitch_buffer".into(),
            i64s(&[dims.features + 1], vec![0; dims.features + 1])?,
        );
        prepare_values.insert("pitchf_buffer".into(), half_zeros(dims.features + 1)?);
        prepare_values.insert(
            "feature_length".into(),
            i64s(&[1], vec![dims.features as i64])?,
        );
        prepare_values.insert(
            "formant_ratio".into(),
            floats(
                &[],
                vec![(dims.formant_length as f64 / dims.ret as f64) as f32],
            )?,
        );
        let mut post_values = Values::new();
        let formant =
            ResampleInputs::new(dims.scaled_window, assets.model_rate / 100, dims.trimmed)?;
        let model_len = formant.output_length;
        put_resample(
            &mut post_values,
            "formant_",
            formant,
            "formant_output_length",
        )?;
        put_resample(
            &mut post_values,
            "output_",
            ResampleInputs::new(assets.model_rate, startup.sample_rate, model_len)?,
            "output_length",
        )?;
        let (fade_in, fade_out) = fade_windows(dims.crossfade)?;
        post_values.insert("fade_in".into(), floats(&[dims.crossfade], fade_in)?);
        post_values.insert("fade_out".into(), floats(&[dims.crossfade], fade_out)?);
        post_values.insert(
            "sola_buffer".into(),
            floats(&[dims.crossfade], vec![0.0; dims.crossfade])?,
        );
        post_values.insert(
            "gather_range".into(),
            i64s(
                &[dims.block + dims.crossfade],
                (0..(dims.block + dims.crossfade) as i64).collect(),
            )?,
        );
        post_values.insert(
            "search_end".into(),
            i64s(&[1], vec![(dims.crossfade + dims.search) as i64])?,
        );
        post_values.insert("block_length".into(), i64s(&[1], vec![dims.block as i64])?);
        Ok(Self {
            startup,
            dims,
            generator,
            pre,
            cv,
            f0,
            prepare,
            post,
            pre_values,
            prepare_values,
            post_values,
            output: Vec::new(),
            failed: false,
        })
    }
    pub fn block_frames(&self) -> usize {
        self.dims.block
    }
    pub fn dimensions(&self) -> &Dims {
        &self.dims
    }
    #[cfg(feature = "native")]
    pub fn loaded_modules(&self) -> Result<Vec<String>> {
        self.generator.loaded_modules()
    }
    #[cfg(feature = "native")]
    pub fn restore_rng(&mut self, cpu: &[u8], cuda: &[u8]) -> Result<()> {
        self.generator.restore_rng(cpu, cuda)
    }
    pub fn seed(&mut self, seed: u64) -> Result<()> {
        self.generator.seed(seed)
    }
    pub fn process(&mut self, block: &[f32], params: &Params) -> Result<&[f32]> {
        self.process_observed(block, params, &mut |_, _| Ok(()))
    }
    /// Observation only: never replaces stage inputs or state.
    pub fn process_observed(
        &mut self,
        block: &[f32],
        params: &Params,
        observer: &mut dyn FnMut(&str, &DynValue) -> Result<()>,
    ) -> Result<&[f32]> {
        ensure!(
            !self.failed,
            "engine previously failed; recreate before continuing"
        );
        ensure!(
            block.len() == self.dims.block && block.iter().all(|v| v.is_finite()),
            "invalid audio block"
        );
        ensure!(
            params.pitch.is_finite()
                && (-48.0..=48.0).contains(&params.pitch)
                && params.threshold_db.is_finite()
                && (-120.0..=0.0).contains(&params.threshold_db),
            "invalid live parameters"
        );
        if let Err(error) = self.process_inner(block, params, observer) {
            self.failed = true;
            return Err(error);
        }
        Ok(&self.output)
    }
    fn process_inner(
        &mut self,
        block: &[f32],
        params: &Params,
        observer: &mut dyn FnMut(&str, &DynValue) -> Result<()>,
    ) -> Result<()> {
        self.pre_values
            .insert("block".into(), floats(&[block.len()], block.to_vec())?);
        self.pre_values.insert(
            "sensitivity".into(),
            floats(&[], vec![10.0f64.powf(params.threshold_db / 20.0) as f32])?,
        );
        let mut pre = run(&mut self.pre, &self.pre_values)?;
        for (name, value) in &pre {
            observer(name, value)?;
        }
        let cv_audio = take(&mut pre, "cv_audio")?;
        let mut cv = run(&mut self.cv, &BTreeMap::from([("audio".into(), cv_audio)]))?;
        observer("cv_features", cv.get("unit12").context("missing unit12")?)?;
        let mel = take(&mut pre, "mel")?;
        let threshold =
            Tensor::from_array((Vec::<usize>::new(), vec![f16::from_f32(0.05)]))?.into_dyn();
        let mut pitch = run(
            &mut self.f0,
            &BTreeMap::from([("mel".into(), mel), ("threshold".into(), threshold)]),
        )?;
        self.prepare_values
            .insert("cv_features".into(), take(&mut cv, "unit12")?);
        self.prepare_values
            .insert("rawf0".into(), take(&mut pitch, "pitchf")?);
        observer("rawf0", self.prepare_values.get("rawf0").unwrap())?;
        self.prepare_values.insert(
            "pitch_factor".into(),
            floats(
                &[],
                vec![2.0f64.powf((params.pitch - self.startup.formant) / 12.0) as f32],
            )?,
        );
        let mut ready = run(&mut self.prepare, &self.prepare_values)?;
        for (name, value) in &ready {
            observer(name, value)?;
        }
        let features = ready
            .get("features")
            .context("missing features")?
            .try_extract_tensor::<f16>()?
            .1;
        let pitch = ready
            .get("pitch")
            .context("missing pitch")?
            .try_extract_tensor::<i64>()?
            .1;
        let pitchf = ready
            .get("pitchf")
            .context("missing pitchf")?
            .try_extract_tensor::<f16>()?
            .1;
        let generated = self.generator.infer(
            features,
            pitch,
            pitchf,
            self.dims.skip,
            self.dims.ret,
            self.dims.formant_length,
        )?;
        self.post_values
            .insert("generated".into(), floats(&[generated.len()], generated)?);
        observer("generated", self.post_values.get("generated").unwrap())?;
        self.post_values
            .insert("volume".into(), take(&mut pre, "volume")?);
        self.post_values
            .insert("active".into(), take(&mut pre, "active")?);
        let mut post = run(&mut self.post, &self.post_values)?;
        for (name, value) in &post {
            observer(name, value)?;
        }
        let output = post
            .get("output")
            .context("missing output")?
            .try_extract_tensor::<f32>()?
            .1;
        ensure!(
            output.len() == self.dims.block && output.iter().all(|v| v.is_finite()),
            "invalid converted output"
        );
        self.output.clear();
        self.output.extend_from_slice(output);
        for name in ["audio_buffer", "convert_buffer"] {
            self.pre_values
                .insert(name.into(), take(&mut pre, &format!("next_{name}"))?);
        }
        for name in ["pitch_buffer", "pitchf_buffer"] {
            self.prepare_values
                .insert(name.into(), take(&mut ready, &format!("next_{name}"))?);
        }
        self.post_values
            .insert("sola_buffer".into(), take(&mut post, "next_sola_buffer")?);
        Ok(())
    }
}
