//! Normal TG RMVPE on one Rust-owned CUDA stream, with cuFFT between ORT graphs.
//! No audio devices, Torch runtime, intermediate host copies, or CPU fallback.
//! The engine's Fast V2 path uses this part. Strict upstream numerical acceptance
//! remains separate; see docs/plans/tg-fast-reference-rebuild.
use anyhow::{ensure, Context, Result};
use crate::gpu_runtime::{check, share, Cuda, GraphKind, Stage, Stream};
use half::f16;
use ort::value::{DynTensor, Tensor};
use sha2::{Digest, Sha256};
use std::{ffi::c_void, path::Path, rc::Rc};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Uploaded,
    Queued,
}

pub struct Observation {
    pub mel: Vec<f16>,
    pub salience: Vec<f16>,
}

/// A fixed-startup normal RMVPE engine. Thread-affine (Rc), fail-stop on GPU errors.
/// GPU intermediate observation is opt-in and never used to make processing work.
pub struct GpuPitch {
    audio: DynTensor,
    spectrum: DynTensor,
    raw: DynTensor,
    stages: Vec<Stage>,
    windowed: *mut c_void,
    mel: *const c_void,
    salience: *const c_void,
    f0: *const c_void,
    length: usize,
    frames: usize,
    padded: usize,
    output: Vec<f32>,
    failed: bool,
    phase: Phase,
    cuda: Rc<Cuda>,
}
impl GpuPitch {
    pub(crate) fn enable_blocking_wait(&self) -> Result<()> {
        ensure!(self.phase == Phase::Idle && !self.failed, "pitch block pending or failed");
        self.cuda.enable_blocking_wait()
    }
    pub fn new(runtime: &Path, seams: &Path, length: usize, graph: bool) -> Result<Self> {
        ensure!(
            (3040..=160000).contains(&length),
            "invalid RMVPE window length"
        );
        let contract: serde_json::Value =
            serde_json::from_slice(&std::fs::read(seams.join("contract.json"))?)?;
        ensure!(
            contract["schema"] == 2
                && contract["algorithm"] == "tg-cufft-rmvpe-v2"
                && contract["model_export"] == "unfused-conv-bn",
            "requires unfused GPU pitch bundle; folded assets are not supported"
        );
        for name in [
            "frame.onnx",
            "mel.onnx",
            "decode.onnx",
            "rmvpe-salience.onnx",
        ] {
            let path = seams.join(name);
            let hash = format!("{:x}", Sha256::digest(std::fs::read(path)?));
            ensure!(
                contract["files"][name].as_str() == Some(&hash),
                "GPU pitch asset mismatch: {name}"
            );
        }
        let frames = length / 160 + 1;
        let padded = frames.div_ceil(32) * 32;
        let cuda = Cuda::new(runtime, frames)?;
        let mut stages: Vec<Stage> = Vec::new();
        for path in [
            seams.join("frame.onnx"),
            seams.join("mel.onnx"),
            seams.join("rmvpe-salience.onnx"),
            seams.join("decode.onnx"),
        ] {
            let allocator = stages.first().map(|s| Rc::clone(&s.allocator));
            stages.push(Stage::new(
                &path,
                Rc::clone(&cuda),
                length,
                graph,
                allocator,
                GraphKind::Pitch,
            )?);
        }
        let audio = Tensor::<f16>::new(&stages[0].allocator, vec![length])?.upcast();
        stages[0].binding.bind_input("audio", &audio)?;
        let index: Vec<i64> = (0..frames)
            .flat_map(|f| {
                (0..1024).map(move |n| {
                    let i = f as i64 * 160 + n - 512;
                    if i < 0 {
                        -i
                    } else if i >= length as i64 {
                        2 * length as i64 - 2 - i
                    } else {
                        i
                    }
                })
            })
            .collect();
        stages[0].indices("index", vec![frames, 1024], &index)?;
        let mut window = Tensor::<f32>::new(&stages[0].allocator, vec![frames, 1024])?;
        let windowed = window.data_ptr_mut();
        stages[0].binding.bind_output("windowed", window)?;
        let spectrum = Tensor::<f32>::new(&stages[1].allocator, vec![frames, 513, 2])?.upcast();
        stages[1].binding.bind_input("spectrum", &spectrum)?;
        let pad_index: Vec<_> = (0..padded)
            .map(|i| {
                if i < frames {
                    i as i64
                } else {
                    (2 * frames - 2 - i) as i64
                }
            })
            .collect();
        stages[1].indices("pad_index", vec![padded], &pad_index)?;
        let mel_tensor = Tensor::<f16>::new(&stages[1].allocator, vec![1, 128, padded])?;
        let mel = mel_tensor.data_ptr();
        stages[2].binding.bind_input("mel", &mel_tensor)?;
        stages[1].binding.bind_output("mel", mel_tensor)?;
        let hidden = Tensor::<f16>::new(&stages[2].allocator, vec![1, padded, 360])?;
        let salience = hidden.data_ptr();
        stages[3].binding.bind_input("salience", &hidden)?;
        stages[2].binding.bind_output("salience", hidden)?;
        stages[3].indices(
            "trim_index",
            vec![frames],
            &(0..frames as i64).collect::<Vec<_>>(),
        )?;
        let raw = Tensor::<f32>::new(&stages[3].allocator, vec![1, frames])?.upcast();
        let f0 = raw.data_ptr();
        stages[3].binding.bind_output("f0", share(&raw)?)?;
        let mut this = Self {
            stages,
            audio,
            spectrum,
            raw,
            windowed,
            mel,
            salience,
            f0,
            length,
            frames,
            padded,
            output: vec![0.; frames],
            failed: false,
            phase: Phase::Idle,
            cuda,
        };
        // Capture each session in sequence on the owning thread before use.
        this.process(&vec![f16::ZERO; length])?;
        Ok(this)
    }
    pub fn process(&mut self, audio: &[f16]) -> Result<&[f32]> {
        self.upload(audio)?;
        self.enqueue()?;
        self.finish()
    }
    pub(crate) fn stream(&self) -> Rc<Cuda> { Rc::clone(&self.cuda) }
    pub(crate) fn device_input(&self) -> &DynTensor { &self.audio }
    pub(crate) fn device_output(&self) -> &DynTensor { &self.raw }
    /// Producer has already recorded a dependency on this stream. The consumer
    /// must record/wait for our completion before reading output or reusing input.
    pub(crate) fn enqueue_device(&mut self) -> Result<()> {
        ensure!(self.phase == Phase::Idle && !self.failed, "GPU pitch block pending or failed");
        self.phase = Phase::Uploaded;
        self.enqueue()?;
        self.phase = Phase::Idle;
        Ok(())
    }
    // Internal split permits both CV/F0 uploads before either stream is launched.
    pub(crate) fn upload(&mut self, audio: &[f16]) -> Result<()> {
        ensure!(!self.failed, "GPU pitch failed; recreate it");
        ensure!(self.phase == Phase::Idle, "GPU pitch block already pending");
        ensure!(
            audio.len() == self.length && audio.iter().all(|v| v.is_finite()),
            "invalid pitch input"
        );
        let result = unsafe {
            self.cuda.copy(
                self.audio.data_ptr_mut(),
                audio.as_ptr().cast(),
                std::mem::size_of_val(audio),
                1,
            )
        };
        if let Err(error) = result {
            self.failed = true;
            return Err(error);
        }
        self.phase = Phase::Uploaded;
        Ok(())
    }
    pub(crate) fn enqueue(&mut self) -> Result<()> {
        ensure!(!self.failed, "GPU pitch failed; recreate it");
        ensure!(
            self.phase == Phase::Uploaded,
            "GPU pitch enqueue requires upload"
        );
        let queued = (|| -> Result<()> {
            // Same stream: no inter-stage host synchronization, no intermediate copies.
            self.stages[0].run()?;
            check(
                unsafe {
                    (self.cuda.fft)(self.cuda.plan, self.windowed, self.spectrum.data_ptr_mut())
                },
                "cufftExecR2C",
            )?;
            for stage in &mut self.stages[1..] {
                stage.run()?;
            }
            Ok(())
        })();
        if let Err(error) = queued {
            let _ = self.cuda.synchronize();
            self.failed = true;
            return Err(error);
        }
        self.phase = Phase::Queued;
        Ok(())
    }
    pub(crate) fn finish(&mut self) -> Result<&[f32]> {
        ensure!(!self.failed, "GPU pitch failed; recreate it");
        ensure!(
            self.phase == Phase::Queued,
            "GPU pitch finish requires enqueue"
        );
        let result = (|| -> Result<()> {
            self.cuda.synchronize()?;
            unsafe {
                self.cuda
                    .copy(self.output.as_mut_ptr().cast(), self.f0, self.frames * 4, 2)?;
            }
            ensure!(
                self.output.iter().all(|v| v.is_finite()),
                "nonfinite RMVPE output"
            );
            Ok(())
        })();
        if let Err(error) = result {
            self.failed = true;
            return Err(error);
        }
        self.phase = Phase::Idle;
        Ok(&self.output)
    }
    /// Call only after process. Diagnostic copies do not participate in scheduling.
    pub fn observe(&mut self) -> Result<Observation> {
        ensure!(!self.failed, "GPU pitch failed; recreate it");
        ensure!(
            self.phase == Phase::Idle,
            "GPU pitch observation requires completed block"
        );
        let mut mel = vec![f16::ZERO; 128 * self.padded];
        let mut salience = vec![f16::ZERO; 360 * self.padded];
        let queued = (|| -> Result<()> {
            unsafe {
                self.cuda
                    .copy(mel.as_mut_ptr().cast(), self.mel, mel.len() * 2, 2)?;
                self.cuda.copy(
                    salience.as_mut_ptr().cast(),
                    self.salience,
                    salience.len() * 2,
                    2,
                )?;
            }
            Ok(())
        })();
        let synchronized = self.cuda.synchronize();
        if let Err(error) = queued.and(synchronized) {
            self.failed = true;
            return Err(error);
        }
        salience.truncate(self.frames * 360);
        Ok(Observation { mel, salience })
    }
}

/// ContentVec's fixed buffers and private nonblocking stream. All three original
/// outputs are bound, as in TG OnnxEmbedder; only unit12 is read by the v2 voice.
pub(crate) struct GpuFeatures {
    audio: DynTensor,
    units: DynTensor,
    stage: Stage,
    features: *const c_void,
    output: Vec<f16>,
    length: usize,
    failed: bool,
    phase: Phase,
}
impl GpuFeatures {
    pub(crate) fn enable_blocking_wait(&self) -> Result<()> {
        ensure!(self.phase == Phase::Idle && !self.failed, "ContentVec block pending or failed");
        self.stage.cuda.enable_blocking_wait()
    }
    pub(crate) fn new(runtime: &Path, path: &Path, length: usize, graph: bool) -> Result<Self> {
        let cuda = Cuda::new(runtime, 0)?;
        Self::on_stream(cuda, path, length, graph)
    }
    pub(crate) fn on_stream(cuda: Rc<Cuda>, path: &Path, length: usize, graph: bool) -> Result<Self> {
        ensure!(length >= 400, "ContentVec input too short");
        let mut stage = Stage::new(path, cuda, length, graph, None, GraphKind::ContentVec)?;
        let audio = Tensor::<f16>::new(&stage.allocator, vec![1, length])?.upcast();
        stage.binding.bind_input("audio", &audio)?;
        // HuBERT seven valid convolutions: kernels 10,3,3,3,3,2,2,
        // strides 5,2,2,2,2,2,2; receptive field 400, total stride 320.
        let frames = (length - 400) / 320 + 1;
        let mut features = std::ptr::null();
        let mut units = None;
        for (name, channels) in [("units9", 256), ("unit12", 768), ("unit12s", 768)] {
            let tensor = Tensor::<f16>::new(&stage.allocator, vec![1, frames, channels])?.upcast();
            if name == "unit12" {
                features = tensor.data_ptr();
                units = Some(share(&tensor)?);
            }
            stage.binding.bind_output(name, tensor)?;
        }
        let mut this = Self {
            audio,
            stage,
            features,
            units: units.context("missing ContentVec unit12")?,
            output: vec![f16::ZERO; frames * 768],
            length,
            failed: false,
            phase: Phase::Idle,
        };
        // Capture before another stream is allowed to enqueue work.
        this.upload(&vec![f16::ZERO; length])?;
        this.enqueue()?;
        this.finish()?;
        Ok(this)
    }
    pub(crate) fn device_input(&self) -> &DynTensor { &self.audio }
    pub(crate) fn device_output(&self) -> &DynTensor { &self.units }
    pub(crate) fn enqueue_device(&mut self) -> Result<()> {
        ensure!(self.phase == Phase::Idle && !self.failed, "ContentVec block pending or failed");
        self.phase = Phase::Uploaded;
        self.enqueue()?;
        self.phase = Phase::Idle;
        Ok(())
    }
    pub(crate) fn upload(&mut self, audio: &[f16]) -> Result<()> {
        ensure!(
            !self.failed && audio.len() == self.length && audio.iter().all(|v| v.is_finite()),
            "invalid ContentVec input or failed engine"
        );
        ensure!(
            self.phase == Phase::Idle,
            "ContentVec block already pending"
        );
        let result = unsafe {
            self.stage.cuda.copy(
                self.audio.data_ptr_mut(),
                audio.as_ptr().cast(),
                std::mem::size_of_val(audio),
                1,
            )
        };
        if result.is_err() {
            self.failed = true;
        } else {
            self.phase = Phase::Uploaded;
        }
        result
    }
    pub(crate) fn enqueue(&mut self) -> Result<()> {
        ensure!(!self.failed, "ContentVec failed; recreate it");
        ensure!(
            self.phase == Phase::Uploaded,
            "ContentVec enqueue requires upload"
        );
        if let Err(error) = self.stage.run() {
            let _ = self.stage.cuda.synchronize();
            self.failed = true;
            return Err(error);
        }
        self.phase = Phase::Queued;
        Ok(())
    }
    pub(crate) fn finish(&mut self) -> Result<&[f16]> {
        ensure!(!self.failed, "ContentVec failed; recreate it");
        ensure!(
            self.phase == Phase::Queued,
            "ContentVec finish requires enqueue"
        );
        let result = (|| -> Result<()> {
            self.stage.cuda.synchronize()?;
            unsafe {
                self.stage.cuda.copy(
                    self.output.as_mut_ptr().cast(),
                    self.features,
                    self.output.len() * 2,
                    2,
                )?;
            }
            ensure!(
                self.output.iter().all(|v| v.is_finite()),
                "nonfinite ContentVec output"
            );
            Ok(())
        })();
        if let Err(error) = result {
            self.failed = true;
            return Err(error);
        }
        self.phase = Phase::Idle;
        Ok(&self.output)
    }
}
impl Drop for GpuFeatures {
    fn drop(&mut self) {
        let _ = self.stage.cuda.synchronize();
        self.stage.binding.clear();
    }
}

// Observation only: event timestamps measure real device overlap, not host call
// duration. No timing probe is allocated on the normal processing path.
pub(crate) struct ParallelProbe {
    events: [*mut c_void; 4],
    pitch: Rc<Cuda>,
    cv: Rc<Cuda>,
    record: unsafe extern "C" fn(*mut c_void, Stream) -> i32,
    elapsed: unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void) -> i32,
}
impl ParallelProbe {
    pub(crate) fn new(pitch: &GpuPitch, cv: &GpuFeatures) -> Result<Self> {
        unsafe {
            let lib = &pitch.cuda._runtime;
            let create: unsafe extern "C" fn(*mut *mut c_void, u32) -> i32 =
                *lib.get(b"cudaEventCreateWithFlags")?;
            let mut probe = Self {
                events: [std::ptr::null_mut(); 4],
                pitch: Rc::clone(&pitch.cuda),
                cv: Rc::clone(&cv.stage.cuda),
                record: *lib.get(b"cudaEventRecord")?,
                elapsed: *lib.get(b"cudaEventElapsedTime")?,
                destroy: *lib.get(b"cudaEventDestroy")?,
            };
            for event in &mut probe.events {
                check(create(event, 0), "cudaEventCreateWithFlags")?;
            }
            Ok(probe)
        }
    }
    pub(crate) fn mark(&self, index: usize) -> Result<()> {
        let stream = if index < 2 {
            self.pitch.stream
        } else {
            self.cv.stream
        };
        check(
            unsafe { (self.record)(self.events[index], stream) },
            "cudaEventRecord",
        )
    }
    /// pitch duration, CV duration, overlap. Call after both streams finish.
    pub(crate) fn milliseconds(&self) -> Result<Vec<f32>> {
        let time = |a, b| -> Result<f32> {
            let mut value = 0.;
            check(
                unsafe { (self.elapsed)(&mut value, self.events[a], self.events[b]) },
                "cudaEventElapsedTime",
            )?;
            Ok(value)
        };
        let pitch_end = time(0, 1)?;
        let cv_start = time(0, 2)?;
        let cv_end = time(0, 3)?;
        Ok(vec![
            pitch_end,
            time(2, 3)?,
            (pitch_end.min(cv_end) - 0f32.max(cv_start)).max(0.),
        ])
    }
}
impl Drop for ParallelProbe {
    fn drop(&mut self) {
        let _ = self.pitch.synchronize();
        let _ = self.cv.synchronize();
        for &event in &self.events {
            if !event.is_null() {
                unsafe {
                    (self.destroy)(event);
                }
            }
        }
    }
}
impl Drop for GpuPitch {
    fn drop(&mut self) {
        let _ = self.cuda.synchronize();
        // Release cross-session references while every allocator/session is alive.
        for stage in &mut self.stages {
            stage.binding.clear();
        }
    }
}
