//! Deiteris arithmetic on ORT, with explicit noise and the existing weight format.
use anyhow::{ensure, Context, Result};
use half::f16;
use ort::{
    ep,
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, Session},
    value::{DynTensor, Tensor},
};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use std::{
    ffi::c_void,
    path::{Path, PathBuf},
};

pub const KIND: &str = "deiteris-onnx-v1";

fn valid_lengths(frames: usize, skip: usize, ret: usize, formant: usize) -> bool {
    frames <= 100000
        && skip > 0
        && frames > skip
        && ret == frames - skip
        && (1..=100000).contains(&formant)
}

struct Cuda {
    memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32,
    _library: libloading::Library,
}
impl Cuda {
    fn load() -> Result<Self> {
        unsafe {
            let library = libloading::Library::new("cudart64_13.dll")?;
            let memcpy = *library.get(b"cudaMemcpy")?;
            Ok(Self {
                memcpy,
                _library: library,
            })
        }
    }
    fn copy(&self, dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> Result<()> {
        let status = unsafe { (self.memcpy)(dst, src, bytes, kind) };
        ensure!(status == 0, "generator cudaMemcpy failed: {status}");
        Ok(())
    }
}

struct Runner {
    binding: IoBinding,
    inputs: Vec<DynTensor>,
    _allocator: Allocator,
    session: Session,
    cuda: Cuda,
    dimensions: [usize; 4],
    output_len: usize,
}

impl Runner {
    fn new(path: &Path, rate: usize, dimensions: [usize; 4], graph: bool) -> Result<Self> {
        let [frames, skip, ret, formant] = dimensions;
        let flow_head = skip.saturating_sub(24);
        let dec_head = skip - flow_head;
        ensure!(
            frames > skip && ret == frames - skip && dec_head > 0 && formant > 0,
            "invalid generator dimensions"
        );
        let upp = rate / 100;
        let mut builder = Session::builder()?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        for (key, value) in [
            ("p_len", frames),
            ("flow_len", frames - flow_head),
            ("sine_len", ret * upp),
            ("ret2_len", formant),
            ("dec_head", dec_head),
        ] {
            builder = builder
                .with_dimension_override(key, value as i64)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        }
        let session = builder
            .with_execution_providers([ep::CUDA::default()
                .with_cuda_graph(graph)
                .build()
                .error_on_failure()])
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(path)?;
        let allocator = Allocator::new(
            &session,
            MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?,
        )?;
        let mut binding = session.create_binding()?;
        let mut inputs = Vec::new();
        for (name, shape, dtype) in [
            ("features", vec![1, frames, 768], 0),
            ("pitch", vec![1, frames], 1),
            ("pitchf", vec![1, frames], 0),
            ("rnd", vec![1, 192, frames - flow_head], 0),
            ("sine_noise", vec![1, ret * upp, 1], 0),
            ("n_res", vec![1, formant], 2),
            ("n_head", vec![1, dec_head], 2),
        ] {
            let value = match dtype {
                0 => Tensor::<f16>::new(&allocator, shape)?.upcast(),
                1 => Tensor::<i64>::new(&allocator, shape)?.upcast(),
                _ => Tensor::<f32>::new(&allocator, shape)?.upcast(),
            };
            binding.bind_input(name, &value)?;
            inputs.push(value);
        }
        binding.bind_output(
            "generated",
            Tensor::<f32>::new(&allocator, vec![formant * upp])?,
        )?;
        Ok(Self {
            binding,
            inputs,
            _allocator: allocator,
            session,
            cuda: Cuda::load()?,
            dimensions,
            output_len: formant * upp,
        })
    }
    fn run(&mut self, data: &[(*const c_void, usize)]) -> Result<Vec<f32>> {
        ensure!(
            data.len() == self.inputs.len(),
            "wrong generator input count"
        );
        for (tensor, &(ptr, bytes)) in self.inputs.iter_mut().zip(data) {
            self.cuda.copy(tensor.data_ptr_mut(), ptr, bytes, 1)?;
        }
        let output = self.session.run_binding(&self.binding)?;
        self.binding.synchronize_outputs()?;
        let mut host = vec![0.0f32; self.output_len];
        self.cuda.copy(
            host.as_mut_ptr().cast(),
            output["generated"].data_ptr(),
            host.len() * 4,
            2,
        )?;
        ensure!(
            host.iter().all(|v| v.is_finite()),
            "nonfinite generator output"
        );
        Ok(host)
    }
}

pub struct Generator {
    path: PathBuf,
    rate: usize,
    graph: bool,
    runner: Option<Runner>,
    rng: StdRng,
    failed: bool,
}
impl Generator {
    /// `model` is a converted model directory, never a Torch checkpoint.
    pub fn new(template: &Path, model: &Path, rate: usize) -> Result<Self> {
        Self::new_with_graph(template, model, rate, true)
    }
    pub fn new_with_graph(template: &Path, model: &Path, rate: usize, graph: bool) -> Result<Self> {
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(model.join("model.json"))?)?;
        ensure!(
            metadata["generator_kind"] == KIND,
            "requires converted {KIND} model"
        );
        ensure!(
            matches!(rate, 32000 | 40000 | 48000)
                && metadata["model_sr"].as_u64() == Some(rate as u64)
                && metadata["inter_channels"] == 192
                && metadata["half"] == true,
            "incompatible generator model"
        );
        ensure!(template.is_file(), "missing generator graph");
        Ok(Self {
            path: template.into(),
            rate,
            graph,
            runner: None,
            rng: StdRng::seed_from_u64(1),
            failed: false,
        })
    }
    pub fn seed(&mut self, seed: u64) -> Result<()> {
        self.rng = StdRng::seed_from_u64(seed);
        Ok(())
    }
    pub fn infer(
        &mut self,
        features: &[f16],
        pitch: &[i64],
        pitchf: &[f16],
        skip: usize,
        ret: usize,
        formant: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            valid_lengths(pitch.len(), skip, ret, formant),
            "invalid generator lengths"
        );
        let latent = 192 * (pitch.len() - skip.saturating_sub(24));
        let sine = ret * (self.rate / 100);
        let mut noise = |count| {
            (0..count)
                .map(|_| {
                    let value: f32 = StandardNormal.sample(&mut self.rng);
                    f16::from_f32(value)
                })
                .collect::<Vec<_>>()
        };
        let rnd = noise(latent);
        let sine_noise = noise(sine);
        self.infer_with_noise(
            features,
            pitch,
            pitchf,
            skip,
            ret,
            formant,
            &rnd,
            &sine_noise,
        )
    }
    /// Shared-noise validation seam; does not alter the product RNG sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn infer_with_noise(
        &mut self,
        features: &[f16],
        pitch: &[i64],
        pitchf: &[f16],
        skip: usize,
        ret: usize,
        formant: usize,
        rnd: &[f16],
        sine_noise: &[f16],
    ) -> Result<Vec<f32>> {
        ensure!(!self.failed, "generator previously failed; recreate it");
        let frames = pitch.len();
        ensure!(valid_lengths(frames, skip, ret, formant), "invalid lengths");
        ensure!(
            features.len() == frames * 768
                && pitchf.len() == frames
                && rnd.len() == 192 * (frames - skip.saturating_sub(24))
                && sine_noise.len() == ret * (self.rate / 100),
            "invalid generator inputs"
        );
        ensure!(
            features
                .iter()
                .chain(pitchf)
                .chain(rnd)
                .chain(sine_noise)
                .all(|v| v.is_finite())
                && pitch.iter().all(|&v| (0..=255).contains(&v)),
            "invalid generator values"
        );
        let dimensions = [frames, skip, ret, formant];
        if self.runner.is_none() {
            self.runner = Some(
                Runner::new(&self.path, self.rate, dimensions, self.graph)
                    .context("initialize Deiteris ONNX generator")?,
            );
        }
        let runner = self.runner.as_mut().unwrap();
        ensure!(
            runner.dimensions == dimensions,
            "startup dimensions changed; recreate generator"
        );
        let n_res = vec![0f32; formant];
        let n_head = vec![0f32; skip.min(24)];
        let data = [
            (features.as_ptr().cast(), std::mem::size_of_val(features)),
            (pitch.as_ptr().cast(), std::mem::size_of_val(pitch)),
            (pitchf.as_ptr().cast(), std::mem::size_of_val(pitchf)),
            (rnd.as_ptr().cast(), std::mem::size_of_val(rnd)),
            (
                sine_noise.as_ptr().cast(),
                std::mem::size_of_val(sine_noise),
            ),
            (n_res.as_ptr().cast(), n_res.len() * 4),
            (n_head.as_ptr().cast(), n_head.len() * 4),
        ];
        let output = runner.run(&data);
        self.failed = output.is_err();
        output
    }
}

#[cfg(test)]
mod tests {
    use super::valid_lengths;
    #[test]
    fn short_context_and_invalid_lengths() {
        assert!(valid_lengths(7, 5, 2, 3));
        assert!(valid_lengths(67, 50, 17, 17));
        assert!(!valid_lengths(7, 8, 1, 1));
        assert!(!valid_lengths(7, 0, 7, 1));
        assert!(!valid_lengths(67, 50, 18, 17));
        assert!(!valid_lengths(67, 50, 17, usize::MAX));
    }
}
