//! Safe single-thread-owned interface to the already-validated native generator.
use anyhow::{ensure, Context, Result};
use half::f16;
use rvc_model::{Dtype, Source};
use std::{
    ffi::{c_char, c_void, CStr, CString},
    marker::PhantomData,
    path::Path,
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
};

unsafe extern "C" {
    fn rvc_error() -> *const c_char;
    fn rvc_modules() -> *const c_char;
    fn rvc_create(path: *const c_char, out: *mut *mut c_void) -> i32;
    fn rvc_set_graph(ptr: *mut c_void, enabled: bool) -> i32;
    fn rvc_destroy(ptr: *mut c_void);
    fn rvc_weight(
        ptr: *mut c_void,
        name: *const c_char,
        data: *const c_void,
        dims: *const i64,
        rank: usize,
        dtype: i32,
    ) -> i32;
    fn rvc_finalize(ptr: *mut c_void) -> i32;
    fn rvc_rng(cpu: *const u8, cn: usize, cuda: *const u8, gn: usize) -> i32;
    fn rvc_seed(seed: u64) -> i32;
    fn rvc_rng_matches(cpu: *const u8, cn: usize, cuda: *const u8, gn: usize) -> i32;
    fn rvc_infer(
        ptr: *mut c_void,
        features: *const c_void,
        frames: i64,
        pitch: *const c_void,
        pitchf: *const c_void,
        skip: i64,
        ret: i64,
        formant: i64,
        output: *mut f32,
        len: usize,
    ) -> i32;
}
fn check(code: i32) -> Result<()> {
    if code != 0 {
        anyhow::bail!(
            "{}",
            unsafe { CStr::from_ptr(rvc_error()) }.to_string_lossy()
        );
    }
    Ok(())
}

static LIVE_GENERATOR: AtomicBool = AtomicBool::new(false);
struct ExclusiveGenerator;
impl Drop for ExclusiveGenerator {
    fn drop(&mut self) {
        LIVE_GENERATOR.store(false, Ordering::Release);
    }
}

pub struct Generator {
    pointer: *mut c_void,
    rate: usize,
    // LibTorch's RNG is process global and this instance is callback-thread owned.
    // Do not promise Sync/Send through the raw pointer.
    _thread: PhantomData<Rc<()>>,
    _exclusive: ExclusiveGenerator,
}
impl Drop for Generator {
    fn drop(&mut self) {
        unsafe { rvc_destroy(self.pointer) }
    }
}
impl Generator {
    pub fn check_rng(&self, cpu: &[u8], cuda: &[u8]) -> Result<()> {
        check(unsafe { rvc_rng_matches(cpu.as_ptr(), cpu.len(), cuda.as_ptr(), cuda.len()) })
    }
    pub fn new(template: &Path, model_path: &Path, template_rate: usize) -> Result<Self> {
        Self::create(template, model_path, template_rate, None)
    }
    /// Product entry point: explicit option, independent of diagnostic environment variables.
    pub fn new_with_graph(template: &Path, model_path: &Path, template_rate: usize, cuda_graph: bool) -> Result<Self> {
        Self::create(template, model_path, template_rate, Some(cuda_graph))
    }
    fn create(template: &Path, model_path: &Path, template_rate: usize, cuda_graph: Option<bool>) -> Result<Self> {
        ensure!(
            LIVE_GENERATOR
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "only one native generator may be alive; stop/drop it before restarting"
        );
        let exclusive = ExclusiveGenerator;
        let model = Source::read(model_path).map_err(anyhow::Error::msg)?;
        ensure!(
            model.version.as_deref() == Some("v2") && model.f0,
            "requires RVC v2 F0"
        );
        ensure!(
            model.embedder.as_deref().is_none_or(|v| v == "contentvec")
                && model.vocoder.as_deref().is_none_or(|v| v == "HiFi-GAN"),
            "unsupported embedder/vocoder"
        );
        let rate = usize::try_from(
            model
                .config
                .last()
                .and_then(|v| v.as_u64())
                .context("missing model rate")?,
        )?;
        ensure!(rate == template_rate, "model/template rate mismatch");
        let mut expected: Vec<serde_json::Value> = serde_json::from_str(
            r#"[1025,32,192,192,768,2,6,3,0,"1",[3,7,11],[[1,3,5],[1,3,5],[1,3,5]],[10,10,2,2],512,[16,16,4,4],1,256,40000]"#,
        )?;
        match rate {
            32000 => {
                expected[0] = 513.into();
                expected[12] = serde_json::json!([10, 8, 2, 2]);
                expected[14] = serde_json::json!([20, 16, 4, 4]);
            }
            40000 => {}
            48000 => {
                expected[12] = serde_json::json!([12, 10, 2, 2]);
                expected[14] = serde_json::json!([24, 20, 4, 4]);
            }
            _ => anyhow::bail!("unsupported model rate"),
        }
        expected[17] = rate.into();
        ensure!(
            model.config.len() == expected.len()
                && model
                    .config
                    .iter()
                    .zip(&expected)
                    .enumerate()
                    .all(|(i, (a, b))| i == 15
                        || a == b
                        || a.as_f64().zip(b.as_f64()).is_some_and(|(a, b)| a == b)),
            "unsupported generator architecture"
        );
        let path = template.canonicalize()?;
        let path = CString::new(
            path.to_str()
                .context("non-UTF8 template path")?
                .trim_start_matches(r"\\?\"),
        )?;
        let mut pointer = std::ptr::null_mut();
        check(unsafe { rvc_create(path.as_ptr(), &mut pointer) })?;
        let generator = Self {
            pointer,
            rate,
            _thread: PhantomData,
            _exclusive: exclusive,
        };
        if let Some(enabled) = cuda_graph {
            check(unsafe { rvc_set_graph(pointer, enabled) })?;
        }
        for (name, tensor) in &model.tensors {
            let (kind, width) = match tensor.dtype {
                Dtype::F16 => (0, 2usize),
                Dtype::F32 => (1, 4usize),
            };
            let bytes = tensor.shape.iter().try_fold(width, |n, &d| {
                ensure!(d > 0, "zero dimension");
                n.checked_mul(d).context("weight size overflow")
            })?;
            ensure!(bytes == tensor.data.len(), "invalid weight byte count");
            let dims = tensor
                .shape
                .iter()
                .map(|&d| i64::try_from(d))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let name = CString::new(name.as_str())?;
            check(unsafe {
                rvc_weight(
                    pointer,
                    name.as_ptr(),
                    tensor.data.as_ptr().cast(),
                    dims.as_ptr(),
                    dims.len(),
                    kind,
                )
            })?;
        }
        check(unsafe { rvc_finalize(pointer) })?;
        Ok(generator)
    }

    /// Diagnostic start-state control. Restore once after warmup for streaming
    /// comparison; normal calls advance the original generators naturally.
    pub fn restore_rng(&mut self, cpu: &[u8], cuda: &[u8]) -> Result<()> {
        ensure!(!cpu.is_empty() && !cuda.is_empty(), "empty RNG state");
        check(unsafe { rvc_rng(cpu.as_ptr(), cpu.len(), cuda.as_ptr(), cuda.len()) })
    }
    /// Sets the stream's generator seed. Call before the first block.
    pub fn seed(&mut self, seed: u64) -> Result<()> {
        check(unsafe { rvc_seed(seed) })
    }

    pub fn loaded_modules(&self) -> Result<Vec<String>> {
        let ptr = unsafe { rvc_modules() };
        ensure!(!ptr.is_null(), "module enumeration failed");
        Ok(unsafe { CStr::from_ptr(ptr) }
            .to_str()?
            .lines()
            .map(str::to_owned)
            .collect())
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
        let frames = pitch.len();
        ensure!(
            frames > 0
                && frames <= 100000
                && features.len() == frames * 768
                && pitchf.len() == frames
                && skip < frames
                && ret > 0
                && formant > 0
                && formant <= 100000,
            "invalid generator inputs"
        );
        ensure!(
            features.iter().all(|v| v.is_finite())
                && pitchf.iter().all(|v| v.is_finite())
                && pitch.iter().all(|&v| (0..=255).contains(&v)),
            "invalid generator values"
        );
        let mut output = vec![0.0; formant * (self.rate / 100)];
        check(unsafe {
            rvc_infer(
                self.pointer,
                features.as_ptr().cast(),
                frames as i64,
                pitch.as_ptr().cast(),
                pitchf.as_ptr().cast(),
                skip as i64,
                ret as i64,
                formant as i64,
                output.as_mut_ptr(),
                output.len(),
            )
        })?;
        Ok(output)
    }
}
