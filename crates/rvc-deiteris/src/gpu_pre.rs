//! Continuous-input GPU preprocessing; CPU owns only absolute sample counters.
use crate::{
    gpu_buffer::Buffer,
    gpu_runtime::{share, Cuda, GraphKind, Stage},
    startup::{Dims, ResampleInputs, Startup},
};
use anyhow::{ensure, Result};
use half::f16;
use ort::{
    memory::Allocator,
    value::{DynTensor, PrimitiveTensorElementType},
};
use std::{collections::BTreeMap, fmt::Debug, path::Path, rc::Rc};

pub struct GpuPre {
    stage: Stage,
    pub(crate) values: BTreeMap<&'static str, Buffer>,
    cuda: Rc<Cuda>,
    dims: Dims,
    rate: usize,
    grid: usize,
    phases: usize,
    history: usize,
    input_count: u64,
    output_count: u64,
    valid: usize,
    failed: bool,
}

impl GpuPre {
    pub fn new(
        runtime: &Path,
        assets: &Path,
        startup: &Startup,
        model_rate: usize,
        graph: bool,
    ) -> Result<Self> {
        Self::on_stream(
            Cuda::new(runtime, 0)?,
            assets,
            startup,
            model_rate,
            graph,
            None,
            None,
        )
    }
    pub(crate) fn on_stream(
        cuda: Rc<Cuda>,
        assets: &Path,
        startup: &Startup,
        model_rate: usize,
        graph: bool,
        allocator: Option<Rc<Allocator>>,
        sinks: Option<(&DynTensor, &DynTensor)>,
    ) -> Result<Self> {
        let dims = startup.tg_dims(model_rate)?;
        let coeff = ResampleInputs::new(startup.sample_rate, 16000, 1)?;
        let phases = coeff.phases;
        let grid = startup.sample_rate * phases / 16000;
        let width = (coeff.taps - grid) / 2;
        let history = 1920.max(2 * width * grid);
        let n = (dims.block * 16000).div_ceil(startup.sample_rate);
        let q = dims.convert - dims.silence_front;
        let stage = Stage::fixed(
            &assets.join("pre.onnx"),
            Rc::clone(&cuda),
            &[
                ("block_samples", dims.block),
                ("source_history_samples", history),
                ("resample_phases", phases),
                ("resample_output_samples", n),
                ("resample_taps", coeff.taps),
                ("audio_buffer_samples", dims.audio),
                ("context_samples", dims.convert),
                ("pitch_input_samples", q),
            ],
            graph,
            allocator,
            GraphKind::Dsp,
        )?;
        let mut this = Self {
            stage,
            values: BTreeMap::new(),
            cuda,
            dims,
            rate: startup.sample_rate,
            grid,
            phases,
            history,
            input_count: 0,
            output_count: 0,
            valid: 0,
            failed: false,
        };
        macro_rules! input {
            ($ty:ty, $name:literal, $shape:expr, $data:expr) => {
                this.add::<$ty>($name, $shape, Some(&$data), false)?;
            };
        }
        input!(
            f32,
            "block",
            vec![this.dims.block],
            vec![0.; this.dims.block]
        );
        input!(f32, "source_history", vec![history], vec![0.; history]);
        input!(
            f16,
            "audio_buffer",
            vec![this.dims.audio],
            vec![f16::ZERO; this.dims.audio]
        );
        input!(
            f16,
            "context_buffer",
            vec![this.dims.convert],
            vec![f16::ZERO; this.dims.convert]
        );
        input!(
            f32,
            "resample_kernel",
            vec![phases, coeff.taps],
            coeff.kernel
        );
        input!(
            i64,
            "resample_sample_index",
            vec![phases, n],
            (0..phases)
                .flat_map(|p| (0..n).map(move |i| ((p + i) / phases * grid) as i64))
                .collect::<Vec<_>>()
        );
        input!(
            i64,
            "resample_kernel_index",
            vec![phases, n],
            (0..phases)
                .flat_map(|p| (0..n).map(move |i| ((p + i) % phases) as i64))
                .collect::<Vec<_>>()
        );
        input!(
            i64,
            "resample_tap_index",
            vec![coeff.taps],
            (0..coeff.taps)
                .map(|i| i as i64 - width as i64)
                .collect::<Vec<_>>()
        );
        for name in ["phase", "origin", "valid"] {
            this.add::<i64>(name, vec![], Some(&[0]), false)?;
        }
        input!(
            i64,
            "source_limit",
            vec![],
            vec![(history + this.dims.block) as i64]
        );
        input!(
            i64,
            "source_history_commit_index",
            vec![history],
            (this.dims.block..this.dims.block + history)
                .map(|i| i as i64)
                .collect::<Vec<_>>()
        );
        input!(
            i64,
            "audio_append_index",
            vec![this.dims.audio],
            (0..this.dims.audio as i64).collect::<Vec<_>>()
        );
        input!(
            i64,
            "context_append_index",
            vec![this.dims.convert],
            (0..this.dims.convert as i64).collect::<Vec<_>>()
        );
        input!(
            i64,
            "pitch_input_index",
            vec![q],
            (this.dims.silence_front..this.dims.convert)
                .map(|i| i as i64)
                .collect::<Vec<_>>()
        );
        input!(f32, "sensitivity", vec![], vec![0.]);
        this.add::<f16>("audio16", vec![n], None, true)?;
        this.add::<f32>("next_source_history", vec![history], None, true)?;
        this.add::<f16>("next_audio_buffer", vec![this.dims.audio], None, true)?;
        this.add::<f16>("next_context_buffer", vec![this.dims.convert], None, true)?;
        if let Some((cv, pitch)) = sinks {
            for (name, value) in [("cv_audio", cv), ("pitch_audio", pitch)] {
                this.stage.binding.bind_output(name, share(value)?)?;
                this.values.insert(
                    name,
                    Buffer::shared(
                        value,
                        Rc::clone(&this.stage.allocator),
                        Rc::clone(&this.cuda),
                    )?,
                );
            }
        } else {
            this.add::<f16>("cv_audio", vec![1, this.dims.convert], None, true)?;
            this.add::<f16>("pitch_audio", vec![q], None, true)?;
        }
        this.add::<f16>("volume", vec![], None, true)?;
        this.add::<bool>("active", vec![], None, true)?;
        this.process(&vec![0.; this.dims.block], 0.)?;
        this.reset()?;
        Ok(this)
    }
    fn add<T: PrimitiveTensorElementType + Debug>(
        &mut self,
        name: &'static str,
        shape: Vec<usize>,
        data: Option<&[T]>,
        output: bool,
    ) -> Result<()> {
        if std::env::var_os("RVC_GPU_PITCH_DIAGNOSTICS").is_some() {
            eprintln!("pre bind {name} {shape:?}");
        }
        let mut buffer = Buffer::new::<T>(
            Rc::clone(&self.stage.allocator),
            Rc::clone(&self.cuda),
            shape,
        )?;
        if let Some(data) = data {
            buffer.upload(data)?;
        }
        if output {
            self.stage
                .binding
                .bind_output(name, share(&buffer.value)?)?;
        } else {
            self.stage.binding.bind_input(name, &buffer.value)?;
        }
        self.values.insert(name, buffer);
        Ok(())
    }
    pub fn process(&mut self, block: &[f32], sensitivity: f32) -> Result<usize> {
        ensure!(
            !self.failed
                && block.len() == self.dims.block
                && block.iter().all(|v| v.is_finite())
                && sensitivity.is_finite(),
            "invalid GPU pre input or failed state"
        );
        let result = self.enqueue(block, sensitivity);
        if result.is_err() {
            self.failed = true;
            let _ = self.cuda.synchronize();
        }
        result
    }
    fn enqueue(&mut self, block: &[f32], sensitivity: f32) -> Result<usize> {
        let end = self
            .input_count
            .checked_add(block.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("input counter overflow"))?;
        let end_out = end
            .checked_mul(16000)
            .ok_or_else(|| anyhow::anyhow!("output counter overflow"))?
            .div_ceil(self.rate as u64);
        let valid = (end_out - self.output_count) as usize;
        let phase = (self.output_count % self.phases as u64) as i64;
        let origin = (self.output_count / self.phases as u64 * self.grid as u64) as i64
            - self.input_count as i64
            + self.history as i64;
        self.values.get_mut("block").unwrap().upload(block)?;
        for (name, value) in [
            ("phase", phase),
            ("origin", origin),
            ("valid", valid as i64),
        ] {
            self.values.get_mut(name).unwrap().upload(&[value])?;
        }
        self.values
            .get_mut("sensitivity")
            .unwrap()
            .upload(&[sensitivity])?;
        self.stage.run()?;
        for (dst, src) in [
            ("source_history", "next_source_history"),
            ("audio_buffer", "next_audio_buffer"),
            ("context_buffer", "next_context_buffer"),
        ] {
            let source = share(&self.values[src].value)?;
            self.values.get_mut(dst).unwrap().copy_value(&source)?;
        }
        self.input_count = end;
        self.output_count = end_out;
        self.valid = valid;
        Ok(valid)
    }
    pub fn audio16(&self) -> Result<Vec<f16>> {
        Ok(self.values["audio16"].read::<f16>()?[..self.valid].to_vec())
    }
    pub fn context(&self) -> Result<Vec<f16>> {
        self.values["next_context_buffer"].read()
    }
    pub fn volume(&self) -> Result<f16> {
        Ok(self.values["volume"].read::<f16>()?[0])
    }
    pub fn total_output(&self) -> u64 {
        self.output_count
    }
    pub fn reset(&mut self) -> Result<()> {
        ensure!(!self.failed, "failed GPU pre must be recreated");
        self.cuda.synchronize()?;
        for name in ["source_history", "audio_buffer", "context_buffer"] {
            self.values.get_mut(name).unwrap().zero()?;
        }
        self.cuda.synchronize()?;
        self.input_count = 0;
        self.output_count = 0;
        self.valid = 0;
        Ok(())
    }
}
impl Drop for GpuPre {
    fn drop(&mut self) {
        let _ = self.cuda.synchronize();
        self.stage.binding.clear();
        self.values.clear();
    }
}
