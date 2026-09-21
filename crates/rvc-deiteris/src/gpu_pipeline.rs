//! Fixed-shape, two-stream TG conversion pipeline with device-resident state.
//! The only normal host boundaries are the input block/live scalars, volume,
//! and the final output block.
use crate::{
    engine::{floats, i64s, Params},
    gpu_buffer::Buffer,
    gpu_noise::Noise,
    gpu_pitch::{GpuFeatures, GpuPitch},
    gpu_pre::GpuPre,
    gpu_runtime::{share, Cuda, Dependency, GraphKind, Stage},
    onnx_generator::validate_model,
    startup::{fade_windows, Dims, ResampleInputs, Startup},
};
use anyhow::{ensure, Context, Result};
use half::f16;
use ort::value::{DynTensor, DynValue, PrimitiveTensorElementType, Tensor};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fmt::Debug, path::Path, rc::Rc};

const ALGORITHM: &str = "tg-full-gpu-pipeline-v1";

const PRE_PORTS: &[&str] = &[
    "block:float32",
    "source_history:float32",
    "audio_buffer:float16",
    "context_buffer:float16",
    "resample_kernel:float32",
    "resample_sample_index:int64",
    "resample_tap_index:int64",
    "resample_kernel_index:int64",
    "phase:int64",
    "origin:int64",
    "valid:int64",
    "source_limit:int64",
    "source_history_commit_index:int64",
    "audio_append_index:int64",
    "context_append_index:int64",
    "pitch_input_index:int64",
    "sensitivity:float32",
    "audio16:float16",
    "next_source_history:float32",
    "next_audio_buffer:float16",
    "next_context_buffer:float16",
    "cv_audio:float16",
    "pitch_audio:float16",
    "volume:float16",
    "active:bool",
];
const PREPARE_PORTS: &[&str] = &[
    "cv_features:float16",
    "raw_f0:float32",
    "pitch_buffer:int64",
    "pitchf_buffer:float16",
    "pitch_factor:float32",
    "formant_ratio:float32",
    "feature_index:int64",
    "pitch_cache_index:int64",
    "generator_pitch_index:int64",
    "features:float16",
    "pitch:int64",
    "pitchf:float16",
    "next_pitch_buffer:int64",
    "next_pitchf_buffer:float16",
];
const POST_PORTS: &[&str] = &[
    "generated:float32",
    "volume:float16",
    "sola_buffer:float32",
    "formant_kernel:float32",
    "formant_indices:int64",
    "formant_output_index:int64",
    "output_kernel:float32",
    "output_indices:int64",
    "output_output_index:int64",
    "sola_search_index:int64",
    "sola_ones:float32",
    "alignment_index:int64",
    "sola_blend_index:int64",
    "fade_in:float32",
    "fade_out:float32",
    "output_index:int64",
    "state_index:int64",
    "audio_model:float32",
    "pre_sola:float32",
    "sola_offset:int64",
    "output:float32",
    "next_sola_buffer:float32",
];
const NOISE_PORTS: &[&str] = &[
    "rnd_f32:float32",
    "sine_noise_f32:float32",
    "rnd:float16",
    "sine_noise:float16",
];

fn sha256(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}

fn contract_ports(graph: &serde_json::Value) -> Result<Vec<String>> {
    graph["inputs"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(graph["outputs"].as_array().into_iter().flatten())
        .map(|port| {
            Ok(format!(
                "{}:{}",
                port["name"].as_str().context("missing graph port name")?,
                port["dtype"].as_str().context("missing graph port dtype")?
            ))
        })
        .collect()
}

fn validate_assets(root: &Path) -> Result<()> {
    let contract: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("contract.json"))?)?;
    ensure!(
        contract["schema"] == 1
            && contract["algorithm"] == ALGORITHM
            && contract["opset"] == 18
            && contract["fixed_shape"] == true
            && contract["runtime_torch_dependency"] == false,
        "unsupported full GPU pipeline contract"
    );
    for (name, ports) in [
        ("pre", PRE_PORTS),
        ("prepare", PREPARE_PORTS),
        ("post", POST_PORTS),
        ("noise", NOISE_PORTS),
    ] {
        let graph = &contract["graphs"][name];
        let file = graph["file"].as_str().context("missing graph file")?;
        let expected_hash = graph["sha256"].as_str().context("missing graph hash")?;
        ensure!(
            file == format!("{name}.onnx"),
            "unexpected graph file: {file}"
        );
        ensure!(
            expected_hash == sha256(&root.join(file))?,
            "full GPU graph hash mismatch: {file}"
        );
        ensure!(
            contract_ports(graph)? == ports.iter().map(|v| (*v).to_owned()).collect::<Vec<_>>(),
            "full GPU graph port mismatch: {file}"
        );
    }
    Ok(())
}

/// One fixed IoBinding plus the allocations which back its owned ports.
struct BoundStage {
    values: BTreeMap<&'static str, Buffer>,
    aliases: Vec<DynTensor>,
    // Last so owned/aliased tensors are released before their session.
    stage: Stage,
}
impl BoundStage {
    fn new(
        path: &Path,
        cuda: Rc<Cuda>,
        dimensions: &[(&str, usize)],
        graph: bool,
        kind: GraphKind,
    ) -> Result<Self> {
        Ok(Self {
            values: BTreeMap::new(),
            aliases: Vec::new(),
            // Captured sessions own their arena. Share tensor handles across
            // stages, never allocate later buffers from a captured CV arena.
            stage: Stage::fixed(path, cuda, dimensions, graph, None, kind)?,
        })
    }
    fn input<T: PrimitiveTensorElementType + Debug>(
        &mut self,
        name: &'static str,
        shape: Vec<usize>,
        data: &[T],
    ) -> Result<()> {
        let mut value = Buffer::new::<T>(
            Rc::clone(&self.stage.allocator),
            Rc::clone(&self.stage.cuda),
            shape,
        )?;
        value.upload(data)?;
        self.stage.binding.bind_input(name, &value.value)?;
        self.values.insert(name, value);
        Ok(())
    }
    fn output<T: PrimitiveTensorElementType + Debug>(
        &mut self,
        name: &'static str,
        shape: Vec<usize>,
    ) -> Result<()> {
        let value = Buffer::new::<T>(
            Rc::clone(&self.stage.allocator),
            Rc::clone(&self.stage.cuda),
            shape,
        )?;
        self.stage.binding.bind_output(name, share(&value.value)?)?;
        self.values.insert(name, value);
        Ok(())
    }
    fn alias_input(&mut self, name: &'static str, value: &DynTensor) -> Result<()> {
        let owned = share(value)?;
        ensure!(
            owned.data_ptr() == value.data_ptr(),
            "GPU alias unexpectedly changed allocation: {name}"
        );
        self.stage.binding.bind_input(name, &owned)?;
        self.aliases.push(owned);
        Ok(())
    }
    fn value(&self, name: &str) -> Result<&DynTensor> {
        Ok(&self
            .values
            .get(name)
            .with_context(|| format!("missing {name}"))?
            .value)
    }
    fn buffer_mut(&mut self, name: &str) -> Result<&mut Buffer> {
        self.values
            .get_mut(name)
            .with_context(|| format!("missing {name}"))
    }
    fn read<T>(&self, name: &str) -> Result<Vec<T>>
    where
        T: PrimitiveTensorElementType + Debug + Default + Clone,
    {
        self.values
            .get(name)
            .with_context(|| format!("missing {name}"))?
            .read()
    }
    fn copy(&mut self, destination: &str, source: &str) -> Result<()> {
        let source = share(self.value(source)?)?;
        self.buffer_mut(destination)?.copy_value(&source)
    }
    fn run(&mut self) -> Result<()> {
        self.stage.run()
    }
    fn capture(&mut self) -> Result<()> {
        self.run()?;
        self.stage.cuda.synchronize()
    }
    fn clear(&mut self) {
        self.stage.binding.clear();
        self.aliases.clear();
    }
}
impl Drop for BoundStage {
    fn drop(&mut self) {
        let _ = self.stage.cuda.synchronize();
        self.clear();
    }
}

fn drain(main: &Rc<Cuda>, pitch: &Rc<Cuda>) -> Result<()> {
    let first = main.synchronize();
    let second = pitch.synchronize();
    first.and(second)
}

fn halves(shape: &[usize], values: Vec<f16>) -> Result<DynValue> {
    Ok(Tensor::from_array((shape.to_vec(), values))?.into_dyn())
}

fn read_device<T>(value: &DynTensor, cuda: &Rc<Cuda>) -> Result<Vec<T>>
where
    T: PrimitiveTensorElementType + Debug + Default + Clone,
{
    ensure!(
        value.dtype().tensor_type() == Some(T::into_tensor_element_type()),
        "diagnostic device read type mismatch"
    );
    let count = value
        .dtype()
        .tensor_shape()
        .context("diagnostic value is not a tensor")?
        .iter()
        .try_fold(1usize, |count, &dimension| {
            usize::try_from(dimension)
                .ok()
                .and_then(|dimension| count.checked_mul(dimension))
        })
        .context("invalid diagnostic tensor shape")?;
    cuda.synchronize()?;
    let mut output = vec![T::default(); count];
    unsafe {
        cuda.copy(
            output.as_mut_ptr().cast(),
            value.data_ptr(),
            std::mem::size_of_val(output.as_slice()),
            2,
        )?;
    }
    Ok(output)
}

/// The current TG index/effects-off conversion path. This type is deliberately
/// not wired into the product until its stage and waveform gates pass.
pub struct GpuPipeline {
    // Consumer bindings precede their producers so normal field destruction
    // also releases every cross-session reference in dependency order.
    post: BoundStage,
    generator: BoundStage,
    noise_cast: BoundStage,
    prepare_stage: BoundStage,
    pre: GpuPre,
    features: GpuFeatures,
    pitch: GpuPitch,
    pre_done: Dependency,
    pitch_done: Dependency,
    noise: Noise,
    main: Rc<Cuda>,
    pitch_stream: Rc<Cuda>,
    startup: Startup,
    dims: Dims,
    flow_frames: usize,
    sine_samples: usize,
    output: Vec<f32>,
    processed: bool,
    post_active: bool,
    failed: bool,
}

impl GpuPipeline {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &Path,
        assets_deiteris_root: &Path,
        pipeline_assets: &Path,
        runtime: &Path,
        startup: Startup,
        graph: bool,
    ) -> Result<Self> {
        validate_assets(pipeline_assets)?;
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(model.join("model.json"))?)?;
        let rate = metadata["model_sr"]
            .as_u64()
            .context("missing model rate")? as usize;
        let dims = startup.tg_dims(rate)?;
        validate_model(&model.join("generator.onnx"), model, rate)?;
        let fast_assets = assets_deiteris_root.join("tg-fast-v1");
        crate::tg::validate_assets(&fast_assets)?;

        let main = Cuda::new(runtime, 0)?;
        let pitch = GpuPitch::new(
            runtime,
            &assets_deiteris_root.join("tg-gpu-pitch-v2"),
            dims.convert - dims.silence_front,
            graph,
        )?;
        let pitch_stream = pitch.stream();
        drain(&main, &pitch_stream)?;
        let features = GpuFeatures::on_stream(
            Rc::clone(&main),
            &fast_assets.join("contentvec.onnx"),
            dims.convert,
            graph,
        )?;
        drain(&main, &pitch_stream)?;
        let pre = GpuPre::on_stream(
            Rc::clone(&main),
            pipeline_assets,
            &startup,
            rate,
            graph,
            None,
            Some((features.device_input(), pitch.device_input())),
        )?;
        ensure!(
            pre.values
                .get("cv_audio")
                .context("missing shared CV input")?
                .value
                .data_ptr()
                == features.device_input().data_ptr()
                && pre
                    .values
                    .get("pitch_audio")
                    .context("missing shared pitch input")?
                    .value
                    .data_ptr()
                    == pitch.device_input().data_ptr(),
            "pre/CV/pitch device handoff was copied instead of shared"
        );
        drain(&main, &pitch_stream)?;

        let cv_frames = (dims.convert - 400) / 320 + 1;
        let raw_frames = (dims.convert - dims.silence_front) / 160 + 1;
        let cache_frames = dims.features + 1;
        ensure!(
            cv_frames > 0 && 2 * (cv_frames + 1) >= dims.features,
            "ContentVec frame mismatch"
        );
        let mut prepare_stage = BoundStage::new(
            &pipeline_assets.join("prepare.onnx"),
            Rc::clone(&main),
            &[
                ("cv_frames", cv_frames),
                ("raw_f0_frames", raw_frames),
                ("cache_frames", cache_frames),
                ("feature_frames", dims.features),
            ],
            graph,
            GraphKind::Dsp,
        )?;
        prepare_stage.alias_input("cv_features", features.device_output())?;
        prepare_stage.alias_input("raw_f0", pitch.device_output())?;
        prepare_stage.input::<i64>("pitch_buffer", vec![cache_frames], &vec![0; cache_frames])?;
        prepare_stage.input::<f16>(
            "pitchf_buffer",
            vec![cache_frames],
            &vec![f16::ZERO; cache_frames],
        )?;
        prepare_stage.input::<f32>("pitch_factor", vec![], &[1.0])?;
        prepare_stage.input::<f32>(
            "formant_ratio",
            vec![],
            &[(dims.formant_length as f64 / dims.ret as f64) as f32],
        )?;
        prepare_stage.input::<i64>(
            "feature_index",
            vec![dims.features],
            &(0..dims.features)
                .map(|i| (i / 2).min(cv_frames - 1) as i64)
                .collect::<Vec<_>>(),
        )?;
        prepare_stage.input::<i64>(
            "pitch_cache_index",
            vec![cache_frames],
            &(0..cache_frames)
                .map(|i| (i + raw_frames) as i64)
                .collect::<Vec<_>>(),
        )?;
        prepare_stage.input::<i64>(
            "generator_pitch_index",
            vec![dims.features],
            &(1..=dims.features as i64).collect::<Vec<_>>(),
        )?;
        prepare_stage.output::<f16>("features", vec![1, dims.features, 768])?;
        prepare_stage.output::<i64>("pitch", vec![1, dims.features])?;
        prepare_stage.output::<f16>("pitchf", vec![1, dims.features])?;
        prepare_stage.output::<i64>("next_pitch_buffer", vec![cache_frames])?;
        prepare_stage.output::<f16>("next_pitchf_buffer", vec![cache_frames])?;
        drain(&main, &pitch_stream)?;
        prepare_stage.capture()?;

        let flow_head = dims.skip.saturating_sub(24);
        let flow_frames = dims
            .features
            .checked_sub(flow_head)
            .context("invalid generator flow length")?;
        let sine_samples = dims.ret * (rate / 100);
        let mut noise_cast = BoundStage::new(
            &pipeline_assets.join("noise.onnx"),
            Rc::clone(&main),
            &[("flow_frames", flow_frames), ("sine_samples", sine_samples)],
            graph,
            GraphKind::Dsp,
        )?;
        noise_cast.input::<f32>(
            "rnd_f32",
            vec![1, 192, flow_frames],
            &vec![0.; 192 * flow_frames],
        )?;
        noise_cast.input::<f32>(
            "sine_noise_f32",
            vec![1, sine_samples, 1],
            &vec![0.; sine_samples],
        )?;
        noise_cast.output::<f16>("rnd", vec![1, 192, flow_frames])?;
        noise_cast.output::<f16>("sine_noise", vec![1, sine_samples, 1])?;
        drain(&main, &pitch_stream)?;
        noise_cast.capture()?;
        let noise = Noise::new(runtime, Rc::clone(&main))?;

        let dec_head = dims.skip - flow_head;
        let mut generator = BoundStage::new(
            &model.join("generator.onnx"),
            Rc::clone(&main),
            &[
                ("p_len", dims.features),
                ("flow_len", flow_frames),
                ("sine_len", sine_samples),
                ("ret2_len", dims.formant_length),
                ("dec_head", dec_head),
            ],
            graph,
            GraphKind::Generator,
        )?;
        generator.alias_input("features", prepare_stage.value("features")?)?;
        generator.alias_input("pitch", prepare_stage.value("pitch")?)?;
        generator.alias_input("pitchf", prepare_stage.value("pitchf")?)?;
        generator.alias_input("rnd", noise_cast.value("rnd")?)?;
        generator.alias_input("sine_noise", noise_cast.value("sine_noise")?)?;
        generator.input::<f32>(
            "n_res",
            vec![1, dims.formant_length],
            &vec![0.; dims.formant_length],
        )?;
        generator.input::<f32>("n_head", vec![1, dec_head], &vec![0.; dec_head])?;
        generator.output::<f32>("generated", vec![dims.generated])?;
        drain(&main, &pitch_stream)?;
        generator.capture()?;

        let formant = ResampleInputs::new(dims.scaled_window, rate / 100, dims.trimmed)?;
        let model_samples = formant.output_length;
        let output_resample = ResampleInputs::new(rate, startup.sample_rate, model_samples)?;
        let pre_sola_samples = output_resample.output_length;
        let alignment_samples = dims.block + dims.crossfade;
        let sola_search_samples = dims.crossfade + dims.search;
        ensure!(
            sola_search_samples <= pre_sola_samples
                && dims.search + alignment_samples <= pre_sola_samples,
            "post output is too short for SOLA"
        );
        let mut post = BoundStage::new(
            &pipeline_assets.join("post.onnx"),
            Rc::clone(&main),
            &[
                ("generated_samples", dims.generated),
                ("crossfade_samples", dims.crossfade),
                ("formant_phases", formant.phases),
                ("formant_taps", formant.taps),
                ("formant_frames", formant.frames),
                ("model_samples", model_samples),
                ("output_phases", output_resample.phases),
                ("output_taps", output_resample.taps),
                ("output_frames", output_resample.frames),
                ("pre_sola_samples", pre_sola_samples),
                ("sola_search_samples", sola_search_samples),
                ("alignment_samples", alignment_samples),
                ("block_samples", dims.block),
            ],
            graph,
            GraphKind::Dsp,
        )?;
        post.alias_input("generated", generator.value("generated")?)?;
        post.alias_input(
            "volume",
            &pre.values
                .get("volume")
                .context("missing pre volume")?
                .value,
        )?;
        post.input::<f32>(
            "sola_buffer",
            vec![dims.crossfade],
            &vec![0.; dims.crossfade],
        )?;
        post.input::<f32>(
            "formant_kernel",
            vec![formant.phases, formant.taps],
            &formant.kernel,
        )?;
        post.input::<i64>(
            "formant_indices",
            vec![formant.frames, formant.taps],
            &formant.indices,
        )?;
        post.input::<i64>(
            "formant_output_index",
            vec![model_samples],
            &(0..model_samples as i64).collect::<Vec<_>>(),
        )?;
        post.input::<f32>(
            "output_kernel",
            vec![output_resample.phases, output_resample.taps],
            &output_resample.kernel,
        )?;
        post.input::<i64>(
            "output_indices",
            vec![output_resample.frames, output_resample.taps],
            &output_resample.indices,
        )?;
        post.input::<i64>(
            "output_output_index",
            vec![pre_sola_samples],
            &(0..pre_sola_samples as i64).collect::<Vec<_>>(),
        )?;
        post.input::<i64>(
            "sola_search_index",
            vec![sola_search_samples],
            &(0..sola_search_samples as i64).collect::<Vec<_>>(),
        )?;
        post.input::<f32>("sola_ones", vec![dims.crossfade], &vec![1.; dims.crossfade])?;
        post.input::<i64>(
            "alignment_index",
            vec![alignment_samples],
            &(0..alignment_samples as i64).collect::<Vec<_>>(),
        )?;
        post.input::<i64>(
            "sola_blend_index",
            vec![alignment_samples],
            &(0..alignment_samples)
                .map(|i| if i < dims.crossfade { i as i64 } else { 0 })
                .collect::<Vec<_>>(),
        )?;
        let (fade, inverse) = fade_windows(dims.crossfade)?;
        let mut fade_in = vec![1.; alignment_samples];
        let mut fade_out = vec![0.; alignment_samples];
        fade_in[..dims.crossfade].copy_from_slice(&fade);
        fade_out[..dims.crossfade].copy_from_slice(&inverse);
        post.input::<f32>("fade_in", vec![alignment_samples], &fade_in)?;
        post.input::<f32>("fade_out", vec![alignment_samples], &fade_out)?;
        post.input::<i64>(
            "output_index",
            vec![dims.block],
            &(0..dims.block as i64).collect::<Vec<_>>(),
        )?;
        post.input::<i64>(
            "state_index",
            vec![dims.crossfade],
            &(dims.block as i64..(dims.block + dims.crossfade) as i64).collect::<Vec<_>>(),
        )?;
        post.output::<f32>("audio_model", vec![model_samples])?;
        post.output::<f32>("pre_sola", vec![pre_sola_samples])?;
        post.output::<i64>("sola_offset", vec![])?;
        post.output::<f32>("output", vec![dims.block])?;
        post.output::<f32>("next_sola_buffer", vec![dims.crossfade])?;
        drain(&main, &pitch_stream)?;
        post.capture()?;

        let pre_done = Dependency::new(Rc::clone(&main), Rc::clone(&pitch_stream))?;
        let pitch_done = Dependency::new(Rc::clone(&pitch_stream), Rc::clone(&main))?;
        Ok(Self {
            post,
            generator,
            noise_cast,
            prepare_stage,
            pre,
            features,
            pitch,
            pre_done,
            pitch_done,
            noise,
            main,
            pitch_stream,
            startup,
            output: vec![0.; dims.block],
            dims,
            flow_frames,
            sine_samples,
            processed: false,
            post_active: false,
            failed: false,
        })
    }

    pub fn block_frames(&self) -> usize {
        self.dims.block
    }

    pub fn seed(&mut self, seed: u64) -> Result<()> {
        ensure!(!self.failed, "failed GPU pipeline must be recreated");
        let result = self.noise.seed(seed);
        self.finish_gpu_mutation(result)
    }

    pub fn reset_stream_state(&mut self) -> Result<()> {
        ensure!(!self.failed, "failed GPU pipeline must be recreated");
        let result = (|| -> Result<()> {
            drain(&self.main, &self.pitch_stream)?;
            self.pre.reset()?;
            for name in ["pitch_buffer", "pitchf_buffer"] {
                self.prepare_stage.buffer_mut(name)?.zero()?;
            }
            self.post.buffer_mut("sola_buffer")?.zero()?;
            self.post.buffer_mut("output")?.zero()?;
            self.main.synchronize()?;
            self.output.fill(0.0);
            self.processed = false;
            self.post_active = false;
            Ok(())
        })();
        self.finish_gpu_mutation(result)
    }

    pub fn prepare(&mut self, seed: u64) -> Result<()> {
        ensure!(!self.failed, "failed GPU pipeline must be recreated");
        let block = (0..self.dims.block)
            .map(|i| {
                (std::f64::consts::TAU * 180.0 * i as f64 / self.startup.sample_rate as f64).sin()
                    as f32
                    * 0.05
            })
            .collect::<Vec<_>>();
        for _ in 0..3 {
            self.process(
                &block,
                &Params {
                    pitch: 0.0,
                    threshold_db: -90.0,
                },
            )?;
        }
        self.reset_stream_state()?;
        self.seed(seed)
    }

    pub fn process(&mut self, block: &[f32], params: &Params) -> Result<&[f32]> {
        self.process_impl(block, params, None)
    }

    /// Diagnostic shared-noise seam. It bypasses cuRAND and therefore does not
    /// consume or otherwise change the product generator's seed/offset sequence.
    pub fn process_with_noise(
        &mut self,
        block: &[f32],
        params: &Params,
        rnd: &[f32],
        sine_noise: &[f32],
    ) -> Result<&[f32]> {
        ensure!(
            rnd.len() == 192 * self.flow_frames
                && sine_noise.len() == self.sine_samples
                && rnd.iter().chain(sine_noise).all(|v| v.is_finite()),
            "invalid shared generator noise"
        );
        self.process_impl(block, params, Some((rnd, sine_noise)))
    }

    /// Diagnostic-only stage copies. Call after a successful `process`; these
    /// host values never feed a later block or alter the product noise stream.
    pub fn observe(&mut self) -> Result<BTreeMap<String, DynValue>> {
        ensure!(
            self.processed && !self.failed,
            "GPU observation requires a completed block"
        );
        let result = self.observe_inner();
        self.finish_gpu_mutation(result)
    }

    fn observe_inner(&mut self) -> Result<BTreeMap<String, DynValue>> {
        drain(&self.main, &self.pitch_stream)?;
        let mut values = BTreeMap::new();

        let audio16 = self.pre.audio16()?;
        values.insert("audio16".into(), halves(&[audio16.len()], audio16)?);
        let context = self.pre.context()?;
        values.insert(
            "next_convert_buffer".into(),
            halves(&[context.len()], context)?,
        );
        values.insert("volume".into(), halves(&[], vec![self.pre.volume()?])?);

        let cv = read_device::<f16>(self.features.device_output(), &self.main)?;
        values.insert("cv_features".into(), halves(&[1, cv.len() / 768, 768], cv)?);
        let raw = read_device::<f32>(self.pitch.device_output(), &self.pitch_stream)?;
        values.insert("rawf0".into(), floats(&[raw.len()], raw)?);
        let pitch_observation = self.pitch.observe()?;
        values.insert(
            "mel".into(),
            halves(
                &[1, 128, pitch_observation.mel.len() / 128],
                pitch_observation.mel,
            )?,
        );
        values.insert(
            "salience".into(),
            halves(
                &[1, pitch_observation.salience.len() / 360, 360],
                pitch_observation.salience,
            )?,
        );

        let features = self.prepare_stage.read::<f16>("features")?;
        values.insert(
            "features".into(),
            halves(&[1, self.dims.features, 768], features)?,
        );
        let pitch = self.prepare_stage.read::<i64>("pitch")?;
        values.insert("pitch".into(), i64s(&[1, self.dims.features], pitch)?);
        let pitchf = self.prepare_stage.read::<f16>("pitchf")?;
        values.insert("pitchf".into(), halves(&[1, self.dims.features], pitchf)?);
        let pitch_cache = self.prepare_stage.read::<i64>("pitch_buffer")?;
        values.insert(
            "pitch_buffer".into(),
            i64s(&[pitch_cache.len()], pitch_cache)?,
        );
        let pitchf_cache = self.prepare_stage.read::<f16>("pitchf_buffer")?;
        values.insert(
            "pitchf_buffer".into(),
            halves(&[pitchf_cache.len()], pitchf_cache)?,
        );

        let generated = self.generator.read::<f32>("generated")?;
        values.insert("generated".into(), floats(&[generated.len()], generated)?);
        for name in ["rnd", "sine_noise"] {
            let noise = self.noise_cast.read::<f16>(name)?;
            values.insert(name.into(), halves(&[noise.len()], noise)?);
        }
        for name in ["audio_model", "pre_sola", "output"] {
            // Muted blocks do not execute post. Its intermediate buffers still
            // contain older data and must not masquerade as this block's values.
            if name != "output" && !self.post_active {
                continue;
            }
            let value = self.post.read::<f32>(name)?;
            values.insert(name.into(), floats(&[value.len()], value)?);
        }
        if self.post_active {
            let offset = self.post.read::<i64>("sola_offset")?;
            values.insert("sola_offset".into(), i64s(&[], offset)?);
        }
        let sola = self.post.read::<f32>("sola_buffer")?;
        values.insert("next_sola_buffer".into(), floats(&[sola.len()], sola)?);
        Ok(values)
    }

    fn process_impl(
        &mut self,
        block: &[f32],
        params: &Params,
        shared_noise: Option<(&[f32], &[f32])>,
    ) -> Result<&[f32]> {
        ensure!(!self.failed, "failed GPU pipeline must be recreated");
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
        if let Err(error) = self.process_inner(block, params, shared_noise) {
            self.failed = true;
            let _ = drain(&self.main, &self.pitch_stream);
            return Err(error);
        }
        self.processed = true;
        Ok(&self.output)
    }

    fn finish_gpu_mutation<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err() {
            self.failed = true;
            let _ = drain(&self.main, &self.pitch_stream);
        }
        result
    }

    fn process_inner(
        &mut self,
        block: &[f32],
        params: &Params,
        shared_noise: Option<(&[f32], &[f32])>,
    ) -> Result<()> {
        let sensitivity = 10.0f64.powf(params.threshold_db / 20.0) as f32;
        let pitch_factor = 2.0f64.powf((params.pitch - self.startup.formant) / 12.0) as f32;
        // Both live values are uploaded before either stream starts this block.
        self.prepare_stage
            .buffer_mut("pitch_factor")?
            .upload(&[pitch_factor])?;
        if let Some((rnd, sine)) = shared_noise {
            // Keep diagnostic H2D beside the other live uploads. This remains
            // race-free if DSP stages later regain asynchronous RunOptions.
            self.noise_cast.buffer_mut("rnd_f32")?.upload(rnd)?;
            self.noise_cast.buffer_mut("sine_noise_f32")?.upload(sine)?;
        }
        self.pre.process(block, sensitivity)?;
        self.pre_done.enqueue()?;
        self.pitch.enqueue_device()?;
        self.features.enqueue_device()?;
        self.pitch_done.enqueue()?;
        self.prepare_stage.run()?;
        self.prepare_stage
            .copy("pitch_buffer", "next_pitch_buffer")?;
        self.prepare_stage
            .copy("pitchf_buffer", "next_pitchf_buffer")?;

        if shared_noise.is_none() {
            let (noise, cast) = (&mut self.noise, &mut self.noise_cast);
            noise.fill(cast.buffer_mut("rnd_f32")?)?;
            noise.fill(cast.buffer_mut("sine_noise_f32")?)?;
        }
        self.noise_cast.run()?;
        self.generator.run()?;

        // This is the single decision scalar. The read drains main after the
        // generator, so the Rust mute branch never skips analysis or inference.
        let volume = self
            .pre
            .values
            .get("volume")
            .context("missing pre volume")?
            .read::<f16>()?[0];
        let active = volume.to_f32() >= sensitivity;
        self.post_active = active;
        if active {
            self.post.run()?;
            self.post.copy("sola_buffer", "next_sola_buffer")?;
        } else {
            self.post.buffer_mut("sola_buffer")?.zero()?;
            self.post.buffer_mut("output")?.zero()?;
        }
        let output = self
            .post
            .values
            .get("output")
            .context("missing post output")?
            .read::<f32>()?;
        ensure!(
            output.len() == self.dims.block && output.iter().all(|v| v.is_finite()),
            "invalid converted output"
        );
        self.output.copy_from_slice(&output);
        Ok(())
    }
}

impl Drop for GpuPipeline {
    fn drop(&mut self) {
        let _ = drain(&self.main, &self.pitch_stream);
        // Release downstream aliases while every producer/session is alive.
        self.post.clear();
        self.generator.clear();
        self.noise_cast.clear();
        self.prepare_stage.clear();
    }
}
