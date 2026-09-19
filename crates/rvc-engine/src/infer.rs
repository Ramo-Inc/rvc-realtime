//! ONNX Runtime sessions for the models: CUDA with `enable_cuda_graph`, each session run through an IoBinding on
//! fixed CUDA tensors (required by ORT's CUDA graphs).

use std::path::Path;
use std::sync::Once;

use half::f16;
use ort::{
    ep::{self, ArenaExtendStrategy},
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{builder::GraphOptimizationLevel, IoBinding, Session},
    value::{DynTensor, Tensor},
};

use crate::config::{F0Method, Setup};
use crate::error::{Error, Result};

fn rt<E: std::fmt::Display>(e: E) -> Error {
    Error::Runtime(e.to_string())
}

fn inference<E: std::fmt::Display>(e: E) -> Error {
    Error::Inference(e.to_string())
}

/// Puts the CUDA / cuDNN runtime directory first on PATH, once per process, before any session loads.
pub(crate) fn prepend_path(dir: &Path) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let old = std::env::var_os("PATH").unwrap_or_default();
        let mut paths = vec![std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf())];
        paths.extend(std::env::split_paths(&old));
        if let Ok(joined) = std::env::join_paths(paths) {
            std::env::set_var("PATH", joined);
        }
    });
}

#[derive(Clone, Copy)]
enum Elem {
    F16,
    F32,
    I64,
}

enum Data<'a> {
    F16(&'a [f16]),
    F32(&'a [f32]),
    I64(&'a [i64]),
}

struct Graph {
    binding: IoBinding,
    inputs: Vec<(String, DynTensor)>,
    output: String,
    output_len: usize,
    output_f16: bool,
    // declared last: device tensors above must be released before their allocator
    _alloc: Allocator,
}

/// `cudaMemcpy` from the CUDA runtime DLL (host <-> device copies for CUDA-graph IoBinding tensors).
struct Cudart {
    _lib: libloading::Library,
    memcpy: unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_void, usize, i32) -> i32,
}

impl Cudart {
    fn load() -> Result<Self> {
        unsafe {
            let lib = libloading::Library::new("cudart64_13.dll").map_err(rt)?;
            let memcpy = *lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_void, usize, i32) -> i32>(b"cudaMemcpy").map_err(rt)?;
            Ok(Self { _lib: lib, memcpy })
        }
    }

    fn copy(&self, dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, bytes: usize, kind: i32) -> Result<()> {
        let rc = unsafe { (self.memcpy)(dst, src, bytes, kind) };
        if rc != 0 {
            return Err(Error::Inference(format!("cudaMemcpy failed: {rc}")));
        }
        Ok(())
    }
}

const H2D: i32 = 1;
const D2H: i32 = 2;

struct Runner {
    graph: Graph,
    session: Session,
    cudart: Cudart,
}

/// `dims`: values for the model's named dimensions (ORT free dimension overrides), fixing every shape at load.
fn session(path: &Path, dims: &[(&str, usize)]) -> Result<Session> {
    let mut b = Session::builder().map_err(rt)?;
    for (name, size) in dims {
        b = b.with_dimension_override(*name, *size as i64).map_err(rt)?;
    }
    b = b.with_optimization_level(GraphOptimizationLevel::Level3).map_err(rt)?;
    let cuda = ep::CUDA::default()
        .with_device_id(0)
        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
        .with_cuda_graph(true);
    b = b.with_execution_providers([cuda.build().error_on_failure()]).map_err(rt)?;
    b.commit_from_file(path).map_err(|e| Error::Runtime(format!("load {}: {e}", path.display())))
}

fn device_tensor(alloc: &Allocator, elem: Elem, shape: &[usize]) -> Result<DynTensor> {
    Ok(match elem {
        Elem::F16 => Tensor::<f16>::new(alloc, shape.to_vec()).map_err(rt)?.upcast(),
        Elem::F32 => Tensor::<f32>::new(alloc, shape.to_vec()).map_err(rt)?.upcast(),
        Elem::I64 => Tensor::<i64>::new(alloc, shape.to_vec()).map_err(rt)?.upcast(),
    })
}

impl Runner {
    fn new(path: &Path, dims: &[(&str, usize)], inputs: &[(&str, Elem, Vec<usize>)], output: (&str, Elem, Vec<usize>)) -> Result<Self> {
        let session = session(path, dims)?;
        let graph = {
            let alloc = Allocator::new(&session, MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default).map_err(rt)?).map_err(rt)?;
            let mut binding = session.create_binding().map_err(rt)?;
            let mut dev_inputs = Vec::new();
            for (name, elem, shape) in inputs {
                let t = device_tensor(&alloc, *elem, shape)?;
                binding.bind_input(*name, &t).map_err(rt)?;
                dev_inputs.push((name.to_string(), t));
            }
            binding.bind_output(output.0, device_tensor(&alloc, output.1, &output.2)?).map_err(rt)?;
            let output_len = output.2.iter().product();
            let output_f16 = matches!(output.1, Elem::F16);
            Graph { binding, inputs: dev_inputs, output: output.0.to_string(), output_len, output_f16, _alloc: alloc }
        };
        Ok(Self { graph, session, cudart: Cudart::load()? })
    }

    /// Runs with host data; returns the single output copied to host as f32.
    fn run(&mut self, inputs: &[(&str, Vec<usize>, Data)]) -> Result<Vec<f32>> {
        let (g, cudart) = (&mut self.graph, &self.cudart);
        for ((name, _shape, data), (bound, dev)) in inputs.iter().zip(g.inputs.iter_mut()) {
            debug_assert_eq!(name, bound);
            let (ptr, bytes) = match data {
                Data::F16(v) => (v.as_ptr() as *const std::ffi::c_void, v.len() * 2),
                Data::F32(v) => (v.as_ptr() as *const std::ffi::c_void, v.len() * 4),
                Data::I64(v) => (v.as_ptr() as *const std::ffi::c_void, v.len() * 8),
            };
            cudart.copy(dev.data_ptr_mut() as *mut std::ffi::c_void, ptr, bytes, H2D)?;
        }
        let outputs = self.session.run_binding(&g.binding).map_err(inference)?;
        g.binding.synchronize_outputs().map_err(inference)?;
        let out = &outputs[g.output.as_str()];
        let src = out.data_ptr() as *const std::ffi::c_void;
        if g.output_f16 {
            let mut host = vec![f16::ZERO; g.output_len];
            cudart.copy(host.as_mut_ptr() as *mut std::ffi::c_void, src, host.len() * 2, D2H)?;
            return Ok(host.iter().map(|x| x.to_f32()).collect());
        }
        let mut host = vec![0f32; g.output_len];
        cudart.copy(host.as_mut_ptr() as *mut std::ffi::c_void, src, host.len() * 4, D2H)?;
        Ok(host)
    }
}

pub(crate) struct Models {
    contentvec: Runner,
    f0: Runner,
    generator: Runner,
    half: bool,
    n16: usize,
    n_res: Vec<f32>,
}

impl Models {
    pub fn load(m: &Setup) -> Result<Self> {
        let d = &m.dims;
        let fe = if m.model.half { Elem::F16 } else { Elem::F32 };
        let (f0_file, f0_out) = match m.startup.f0 {
            F0Method::Rmvpe => (&m.model.files.rmvpe, ("hidden", Elem::F32, vec![1, d.f0_frames, 360])),
            F0Method::Fcpe => (&m.model.files.fcpe, ("f0", Elem::F32, vec![1, d.f0_frames, 1])),
        };
        let contentvec = Runner::new(
            &m.model_path(&m.model.files.contentvec),
            &[("n16", d.n16)],
            &[("audio", fe, vec![1, d.n16])],
            ("feats", fe, vec![1, d.feats_frames, 768]),
        )?;
        let f0 = Runner::new(
            &m.model_path(f0_file),
            &[("f0_frames", d.f0_frames)],
            &[("mag", Elem::F32, vec![1, crate::dsp::BINS, d.f0_frames])],
            f0_out,
        )?;
        let upp = m.model.upp;
        let generator = Runner::new(
            &m.model_path(&m.model.files.generator),
            &[("p_len", d.p_len), ("flow_len", d.p_len - d.flow_head), ("sine_len", d.return_length * upp), ("ret2_len", d.return_length2)],
            &[
                ("feats", fe, vec![1, d.p_len, 768]),
                ("pitch", Elem::I64, vec![1, d.p_len]),
                ("pitchf", Elem::F32, vec![1, d.p_len]),
                ("rnd", fe, vec![1, m.model.inter_channels, d.p_len - d.flow_head]),
                ("sine_noise", Elem::F32, vec![1, d.return_length * upp, 1]),
                ("n_res", Elem::F32, vec![1, d.return_length2]),
            ],
            ("audio", Elem::F32, vec![1, 1, d.return_length2 * upp]),
        )?;
        Ok(Self { contentvec, f0, generator, half: m.model.half, n16: d.n16, n_res: vec![0.0; d.return_length2] })
    }

    /// `last_hidden_state` for 16 kHz audio [1, n16] -> flat [frames x 768] (f32).
    pub fn contentvec(&mut self, audio16: &[f32]) -> Result<Vec<f32>> {
        let shape = vec![1, self.n16];
        if self.half {
            let data: Vec<f16> = audio16.iter().map(|&v| f16::from_f32(v)).collect();
            self.contentvec.run(&[("audio", shape, Data::F16(&data))])
        } else {
            self.contentvec.run(&[("audio", shape, Data::F32(audio16))])
        }
    }

    /// RMVPE: salience [frames x 360]; FCPE: f0 Hz [frames].
    pub fn f0(&mut self, mag: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.f0.run(&[("mag", vec![1, crate::dsp::BINS, frames], Data::F32(mag))])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generator(
        &mut self,
        feats: &[f32],
        p_len: usize,
        pitch: &[i64],
        pitchf: &[f32],
        rnd: &[f32],
        rnd_shape: [usize; 3],
        sine_noise: &[f32],
    ) -> Result<Vec<f32>> {
        let fshape = vec![1, p_len, 768];
        let pshape = vec![1, p_len];
        let sshape = vec![1, sine_noise.len(), 1];
        if self.half {
            let feats16: Vec<f16> = feats.iter().map(|&v| f16::from_f32(v)).collect();
            let rnd16: Vec<f16> = rnd.iter().map(|&v| f16::from_f32(v)).collect();
            self.generator.run(&[
                ("feats", fshape, Data::F16(&feats16)),
                ("pitch", pshape.clone(), Data::I64(pitch)),
                ("pitchf", pshape, Data::F32(pitchf)),
                ("rnd", rnd_shape.to_vec(), Data::F16(&rnd16)),
                ("sine_noise", sshape.clone(), Data::F32(sine_noise)),
                ("n_res", vec![1, self.n_res.len()], Data::F32(&self.n_res)),
            ])
        } else {
            self.generator.run(&[
                ("feats", fshape, Data::F32(feats)),
                ("pitch", pshape.clone(), Data::I64(pitch)),
                ("pitchf", pshape, Data::F32(pitchf)),
                ("rnd", rnd_shape.to_vec(), Data::F32(rnd)),
                ("sine_noise", sshape, Data::F32(sine_noise)),
                ("n_res", vec![1, self.n_res.len()], Data::F32(&self.n_res)),
            ])
        }
    }
}
