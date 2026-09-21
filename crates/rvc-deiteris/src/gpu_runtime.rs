//! Shared thread-affine CUDA stream and fixed-binding ORT stage ownership.
use anyhow::{ensure, Context, Result};
use ort::{
    ep,
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{builder::GraphOptimizationLevel, IoBinding, RunOptions, Session},
    value::{DynTensor, Tensor},
};
use std::{ffi::c_void, path::Path, rc::Rc};
pub(crate) type Stream = *mut c_void;
#[derive(Clone, Copy)]
pub(crate) enum GraphKind {
    Pitch,
    ContentVec,
    Dsp,
    Generator,
}
pub(crate) struct Cuda {
    pub(crate) stream: Stream,
    blocking_wait: std::cell::RefCell<Option<BlockingWait>>,
    pub(crate) plan: i32,
    sync: unsafe extern "C" fn(Stream) -> i32,
    stream_destroy: unsafe extern "C" fn(Stream) -> i32,
    plan_destroy: unsafe extern "C" fn(i32) -> i32,
    pub(crate) fft: unsafe extern "C" fn(i32, *mut c_void, *mut c_void) -> i32,
    copy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32,
    copy_async: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, Stream) -> i32,
    memset_async: unsafe extern "C" fn(*mut c_void, i32, usize, Stream) -> i32,
    pub(crate) _runtime: libloading::Library,
    _fft: libloading::Library,
}
pub(crate) fn check(code: i32, operation: &str) -> Result<()> {
    ensure!(code == 0, "{operation} failed: {code}");
    Ok(())
}
impl Cuda {
    pub(crate) fn new(runtime: &Path, frames: usize) -> Result<Rc<Self>> {
        ensure!(runtime.is_absolute(), "runtime directory must be absolute");
        unsafe {
            let lib = libloading::Library::new(runtime.join("cudart64_13.dll"))?;
            let fft = libloading::Library::new(runtime.join("cufft64_12.dll"))?;
            let create: unsafe extern "C" fn(*mut Stream, u32) -> i32 =
                *lib.get(b"cudaStreamCreateWithFlags")?;
            let plan_create: unsafe extern "C" fn(*mut i32, i32, i32, i32) -> i32 =
                *fft.get(b"cufftPlan1d")?;
            let set_stream: unsafe extern "C" fn(i32, Stream) -> i32 =
                *fft.get(b"cufftSetStream")?;
            let mut owner = Self {
                stream: std::ptr::null_mut(),
                blocking_wait: std::cell::RefCell::new(None),
                plan: 0,
                sync: *lib.get(b"cudaStreamSynchronize")?,
                stream_destroy: *lib.get(b"cudaStreamDestroy")?,
                plan_destroy: *fft.get(b"cufftDestroy")?,
                fft: *fft.get(b"cufftExecR2C")?,
                copy: *lib.get(b"cudaMemcpy")?,
                copy_async: *lib.get(b"cudaMemcpyAsync")?,
                memset_async: *lib.get(b"cudaMemsetAsync")?,
                _runtime: lib,
                _fft: fft,
            };
            check(create(&mut owner.stream, 1), "cudaStreamCreateWithFlags")?;
            if frames > 0 {
                check(
                    plan_create(&mut owner.plan, 1024, 0x2a, frames as i32),
                    "cufftPlan1d",
                )?;
                check(set_stream(owner.plan, owner.stream), "cufftSetStream")?;
            }
            Ok(Rc::new(owner))
        }
    }
    pub(crate) fn synchronize(&self) -> Result<()> {
        if let Some(wait) = self.blocking_wait.borrow().as_ref() {
            return wait.wait_or_drain(self.stream, self.sync);
        }
        check(unsafe { (self.sync)(self.stream) }, "cudaStreamSynchronize")
    }
    pub(crate) fn enable_blocking_wait(&self) -> Result<()> {
        let mut slot = self.blocking_wait.borrow_mut();
        if slot.is_none() {
            *slot = Some(BlockingWait::new(&self._runtime)?);
        }
        Ok(())
    }
    /// Both device allocations must remain alive until this stream is drained.
    pub(crate) unsafe fn device_copy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        check(
            (self.copy_async)(dst, src, bytes, 3, self.stream),
            "cudaMemcpyAsync D2D",
        )
    }
    pub(crate) unsafe fn zero(&self, dst: *mut c_void, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        check(
            (self.memset_async)(dst, 0, bytes, self.stream),
            "cudaMemsetAsync",
        )
    }
    // Host boundaries are synchronous. No caller-owned pointer survives a return,
    // including errors. H2D must finish on the default stream before the nonblocking
    // inference stream reads it; D2H callers first drain the inference stream.
    pub(crate) unsafe fn copy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: i32,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        check((self.copy)(dst, src, bytes, kind), "cudaMemcpy")?;
        if kind == 1 {
            check((self.sync)(std::ptr::null_mut()), "upload completion")?;
        }
        Ok(())
    }
}

/// Host completion wait. The event belongs to exactly one stream/owner,
/// and is destroyed before Cuda's runtime library is unloaded.
struct BlockingWait {
    event: *mut c_void,
    record: unsafe extern "C" fn(*mut c_void, Stream) -> i32,
    synchronize: unsafe extern "C" fn(*mut c_void) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void) -> i32,
}
impl BlockingWait {
    fn new(runtime: &libloading::Library) -> Result<Self> {
        unsafe {
            let create: unsafe extern "C" fn(*mut *mut c_void, u32) -> i32 =
                *runtime.get(b"cudaEventCreateWithFlags")?;
            let mut wait = Self {
                event: std::ptr::null_mut(),
                record: *runtime.get(b"cudaEventRecord")?,
                synchronize: *runtime.get(b"cudaEventSynchronize")?,
                destroy: *runtime.get(b"cudaEventDestroy")?,
            };
            // cudaEventBlockingSync | cudaEventDisableTiming.
            check(create(&mut wait.event, 1 | 2), "create blocking completion event")?;
            Ok(wait)
        }
    }
    fn wait(&self, stream: Stream) -> Result<()> {
        unsafe {
            check((self.record)(self.event, stream), "record blocking completion event")?;
            check((self.synchronize)(self.event), "wait for blocking completion event")
        }
    }
    fn wait_or_drain(&self, stream: Stream, drain: unsafe extern "C" fn(Stream) -> i32) -> Result<()> {
        let result = self.wait(stream);
        if result.is_err() {
            // Retain the stream drain on event failure before returning to callers
            // that own host buffers. Report the original failure even if drain fails.
            let _ = unsafe { drain(stream) };
        }
        result
    }
}
impl Drop for BlockingWait {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe { (self.destroy)(self.event); }
        }
    }
}

#[cfg(test)]
mod blocking_wait_tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Default)]
    struct Trace {
        calls: Vec<&'static str>,
        record_error: i32,
        wait_error: i32,
        stream: usize,
    }
    thread_local! { static TRACE: RefCell<Trace> = RefCell::new(Trace::default()); }
    unsafe extern "C" fn record(_: *mut c_void, stream: Stream) -> i32 {
        TRACE.with(|t| {
            let mut t = t.borrow_mut();
            t.calls.push("record");
            t.stream = stream as usize;
            t.record_error
        })
    }
    unsafe extern "C" fn wait(_: *mut c_void) -> i32 {
        TRACE.with(|t| {
            let mut t = t.borrow_mut();
            t.calls.push("wait");
            t.wait_error
        })
    }
    unsafe extern "C" fn drain(stream: Stream) -> i32 {
        TRACE.with(|t| {
            let mut t = t.borrow_mut();
            t.calls.push("drain");
            t.stream = stream as usize;
        });
        99 // A drain failure must not replace the original event error.
    }
    unsafe extern "C" fn destroy(_: *mut c_void) -> i32 {
        TRACE.with(|t| t.borrow_mut().calls.push("destroy"));
        0
    }
    fn event(record_error: i32, wait_error: i32) -> BlockingWait {
        TRACE.with(|t| *t.borrow_mut() = Trace { record_error, wait_error, ..Trace::default() });
        BlockingWait { event: 1usize as *mut c_void, record, synchronize: wait, destroy }
    }
    #[test]
    fn reusing_completed_event_records_again_before_each_wait() {
        let event = event(0, 0);
        event.wait_or_drain(2usize as Stream, drain).unwrap();
        event.wait_or_drain(2usize as Stream, drain).unwrap();
        drop(event);
        TRACE.with(|t| {
            let t = t.borrow();
            assert_eq!(t.calls, ["record", "wait", "record", "wait", "destroy"]);
            assert_eq!(t.stream, 2);
        });
    }
    #[test]
    fn failed_record_drains_without_waiting_on_stale_event() {
        let event = event(7, 0);
        let error = event.wait_or_drain(2usize as Stream, drain).unwrap_err();
        assert_eq!(error.to_string(), "record blocking completion event failed: 7");
        drop(event);
        TRACE.with(|t| assert_eq!(t.borrow().calls, ["record", "drain", "destroy"]));
    }
    #[test]
    fn failed_wait_drains_and_preserves_original_error() {
        let event = event(0, 8);
        let error = event.wait_or_drain(2usize as Stream, drain).unwrap_err();
        assert_eq!(error.to_string(), "wait for blocking completion event failed: 8");
        drop(event);
        TRACE.with(|t| assert_eq!(t.borrow().calls, ["record", "wait", "drain", "destroy"]));
    }
}
impl Drop for Cuda {
    fn drop(&mut self) {
        unsafe {
            if !self.stream.is_null() {
                (self.sync)(self.stream);
            }
            // Drop the completed event before its stream and runtime library.
            self.blocking_wait.get_mut().take();
            if self.plan != 0 {
                (self.plan_destroy)(self.plan);
            }
            if !self.stream.is_null() {
                (self.stream_destroy)(self.stream);
            }
        }
    }
}

// Field order and shared CUDA owner keep streams alive on constructor errors too.
pub(crate) struct Stage {
    pub(crate) binding: IoBinding,
    pub(crate) allocator: Rc<Allocator>,
    session: Session,
    options: RunOptions,
    pub(crate) cuda: Rc<Cuda>,
}
impl Stage {
    pub(crate) fn new(
        path: &Path,
        cuda: Rc<Cuda>,
        length: usize,
        graph: bool,
        shared_allocator: Option<Rc<Allocator>>,
        kind: GraphKind,
    ) -> Result<Self> {
        Self::fixed(
            path,
            cuda,
            &[
                ("audio_len", length),
                ("real_frames", length / 160 + 1),
                ("mel_frames", (length / 160 + 1).div_ceil(32) * 32),
                ("audio_dynamic_axes_1", length),
            ],
            graph,
            shared_allocator,
            kind,
        )
    }

    pub(crate) fn fixed(
        path: &Path,
        cuda: Rc<Cuda>,
        dimensions: &[(&str, usize)],
        graph: bool,
        shared_allocator: Option<Rc<Allocator>>,
        kind: GraphKind,
    ) -> Result<Self> {
        if std::env::var_os("RVC_GPU_PITCH_DIAGNOSTICS").is_some() {
            eprintln!("creating CUDA stage {}", path.display());
        }
        let mut builder = Session::builder()?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        if std::env::var_os("RVC_GPU_PITCH_DIAGNOSTICS").is_some() {
            builder = builder
                .with_log_level(ort::logging::LogLevel::Verbose)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        }
        for &(name, count) in dimensions {
            builder = builder
                .with_dimension_override(name, count as i64)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        }
        // Preserve each model's numerical policy. RMVPE needs explicit half BN
        // boundaries; ContentVec retains the original ORT default optimizations.
        let (optimization, search) = match kind {
            GraphKind::Pitch | GraphKind::Dsp => (
                GraphOptimizationLevel::Level1,
                ep::cuda::ConvAlgorithmSearch::Heuristic,
            ),
            GraphKind::ContentVec | GraphKind::Generator => (
                GraphOptimizationLevel::Level3,
                ep::cuda::ConvAlgorithmSearch::Exhaustive,
            ),
        };
        if matches!(kind, GraphKind::Pitch) {
            builder = builder
                .with_disabled_optimizers("ConvBNFusion")
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        }
        if matches!(kind, GraphKind::ContentVec | GraphKind::Generator) {
            // Without this, the identical input changes between replays in one
            // captured session. The full-GPU generator shows the same failure
            // with identical features/pitch/noise. Both need deterministic ops.
            builder = builder
                .with_deterministic_compute(true)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        }
        let provider = unsafe { ep::CUDA::default().with_compute_stream(cuda.stream.cast()) }
            .with_tf32(false)
            .with_conv_algorithm_search(search)
            .with_cuda_graph(graph)
            .build()
            .error_on_failure();
        let session = builder
            .with_optimization_level(optimization)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_disable_cpu_fallback()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_execution_providers([provider])
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(path)
            .with_context(|| format!("CUDA-only graph {}", path.display()))?;
        if std::env::var_os("RVC_GPU_PITCH_DIAGNOSTICS").is_some() {
            eprintln!("created CUDA stage {}", path.display());
        }
        let allocator = match shared_allocator {
            Some(allocator) => allocator,
            None => Rc::new(Allocator::new(
                &session,
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    0,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?,
            )?),
        };
        let binding = session.create_binding()?;
        let mut options = RunOptions::new()?;
        // ORT's small DSP graphs require its normal per-run synchronization:
        // disabling it reproduces a native failure in ReleaseSession even after
        // explicit stream/binding drains. CV/F0 retain asynchronous fan-out.
        if !matches!(kind, GraphKind::Dsp) {
            options.disable_device_sync()?;
        }
        Ok(Self {
            binding,
            allocator,
            session,
            options,
            cuda,
        })
    }
    pub(crate) fn run(&mut self) -> Result<()> {
        if let Err(error) = self
            .session
            .run_binding_with_options(&self.binding, &self.options)
        {
            eprintln!("CUDA stage execution failed: {error}");
            return Err(error.into());
        }
        Ok(())
    }
    pub(crate) fn indices(&mut self, name: &str, shape: Vec<usize>, values: &[i64]) -> Result<()> {
        let mut tensor = Tensor::<i64>::new(&self.allocator, shape)?;
        let copied = unsafe {
            self.cuda.copy(
                tensor.data_ptr_mut(),
                values.as_ptr().cast(),
                std::mem::size_of_val(values),
                1,
            )
        };
        let synchronized = self.cuda.synchronize(); // Even on copy failure, keep host values alive until drained.
        copied?;
        synchronized?;
        self.binding.bind_input(name, &tensor)?;
        Ok(()) // Binding retains this allocation, without Tensor::clone.
    }
}

/// Share the allocation, not Tensor::clone (which performs a device copy).
pub(crate) fn share(value: &DynTensor) -> Result<DynTensor> {
    value
        .view()
        .try_upgrade()
        .map_err(|_| anyhow::anyhow!("GPU tensor is not owned"))
}
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = self.cuda.synchronize();
    }
}

/// A dependency event owns both streams until all waits have finished.
pub(crate) struct Dependency {
    event: *mut c_void,
    source: Rc<Cuda>,
    target: Rc<Cuda>,
    record: unsafe extern "C" fn(*mut c_void, Stream) -> i32,
    wait: unsafe extern "C" fn(Stream, *mut c_void, u32) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void) -> i32,
}
impl Dependency {
    pub(crate) fn new(source: Rc<Cuda>, target: Rc<Cuda>) -> Result<Self> {
        unsafe {
            let lib = &source._runtime;
            let create: unsafe extern "C" fn(*mut *mut c_void, u32) -> i32 =
                *lib.get(b"cudaEventCreateWithFlags")?;
            let mut this = Self {
                event: std::ptr::null_mut(),
                record: *lib.get(b"cudaEventRecord")?,
                wait: *lib.get(b"cudaStreamWaitEvent")?,
                destroy: *lib.get(b"cudaEventDestroy")?,
                source,
                target,
            };
            check(create(&mut this.event, 2), "cudaEventCreateWithFlags")?;
            Ok(this)
        }
    }
    pub(crate) fn enqueue(&self) -> Result<()> {
        unsafe {
            check(
                (self.record)(self.event, self.source.stream),
                "cudaEventRecord",
            )?;
            check(
                (self.wait)(self.target.stream, self.event, 0),
                "cudaStreamWaitEvent",
            )
        }
    }
}
impl Drop for Dependency {
    fn drop(&mut self) {
        let _ = self.source.synchronize();
        let _ = self.target.synchronize();
        if !self.event.is_null() {
            unsafe {
                (self.destroy)(self.event);
            }
        }
    }
}
