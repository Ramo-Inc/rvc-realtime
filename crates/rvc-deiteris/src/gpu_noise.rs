//! cuRAND uses the existing runtime DLL; no Torch or CUDA compiler at runtime.
use crate::{
    gpu_buffer::Buffer,
    gpu_runtime::{check, Cuda, Stream},
};
use anyhow::{ensure, Result};
use std::{ffi::c_void, path::Path, rc::Rc};

pub(crate) struct Noise {
    generator: *mut c_void,
    generate: unsafe extern "C" fn(*mut c_void, *mut f32, usize, f32, f32) -> i32,
    seed_fn: unsafe extern "C" fn(*mut c_void, u64) -> i32,
    offset_fn: unsafe extern "C" fn(*mut c_void, u64) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    cuda: Rc<Cuda>,
    _library: libloading::Library,
}
impl Noise {
    pub(crate) fn new(runtime: &Path, cuda: Rc<Cuda>) -> Result<Self> {
        unsafe {
            let library = libloading::Library::new(runtime.join("curand64_10.dll"))?;
            let create: unsafe extern "C" fn(*mut *mut c_void, i32) -> i32 =
                *library.get(b"curandCreateGenerator")?;
            let set_stream: unsafe extern "C" fn(*mut c_void, Stream) -> i32 =
                *library.get(b"curandSetStream")?;
            let mut this = Self {
                generator: std::ptr::null_mut(),
                generate: *library.get(b"curandGenerateNormal")?,
                seed_fn: *library.get(b"curandSetPseudoRandomGeneratorSeed")?,
                offset_fn: *library.get(b"curandSetGeneratorOffset")?,
                destroy: *library.get(b"curandDestroyGenerator")?,
                cuda,
                _library: library,
            };
            check(
                create(&mut this.generator, 161),
                "curandCreateGenerator PHILOX4_32_10",
            )?;
            check(
                set_stream(this.generator, this.cuda.stream),
                "curandSetStream",
            )?;
            this.seed(1)?;
            Ok(this)
        }
    }
    pub(crate) fn seed(&mut self, seed: u64) -> Result<()> {
        self.cuda.synchronize()?;
        unsafe {
            check(
                (self.seed_fn)(self.generator, seed),
                "curandSetPseudoRandomGeneratorSeed",
            )?;
            check(
                (self.offset_fn)(self.generator, 0),
                "curandSetGeneratorOffset",
            )
        }
    }
    pub(crate) fn fill(&mut self, buffer: &mut Buffer) -> Result<()> {
        ensure!(
            buffer.value.dtype().tensor_type() == Some(ort::value::TensorElementType::Float32),
            "noise buffer must be f32"
        );
        let count = buffer
            .value
            .dtype()
            .tensor_shape()
            .unwrap()
            .iter()
            .map(|&v| v as usize)
            .product::<usize>();
        ensure!(
            count > 0 && count % 2 == 0,
            "cuRAND normal count must be even"
        );
        unsafe {
            check(
                (self.generate)(
                    self.generator,
                    buffer.value.data_ptr_mut().cast(),
                    count,
                    0.,
                    1.,
                ),
                "curandGenerateNormal",
            )
        }
    }
}
impl Drop for Noise {
    fn drop(&mut self) {
        let _ = self.cuda.synchronize();
        if !self.generator.is_null() {
            unsafe {
                (self.destroy)(self.generator);
            }
        }
    }
}
