//! Fixed device storage. Host transfers are explicit diagnostic/I/O boundaries.
use crate::gpu_runtime::Cuda;
use anyhow::{ensure, Result};
use ort::{
    memory::Allocator,
    value::{DynTensor, PrimitiveTensorElementType, Tensor},
};
use std::{fmt::Debug, rc::Rc};

pub(crate) struct Buffer {
    pub(crate) value: DynTensor,
    bytes: usize,
    _allocator: Rc<Allocator>,
    cuda: Rc<Cuda>,
}
impl Buffer {
    pub(crate) fn shared(
        value: &DynTensor,
        allocator: Rc<Allocator>,
        cuda: Rc<Cuda>,
    ) -> Result<Self> {
        let count = value
            .dtype()
            .tensor_shape()
            .unwrap()
            .iter()
            .try_fold(1usize, |n, &d| {
                usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
            })
            .ok_or_else(|| anyhow::anyhow!("invalid fixed device shape"))?;
        let bytes = value
            .dtype()
            .tensor_type()
            .unwrap()
            .byte_size(count)
            .ok_or_else(|| anyhow::anyhow!("unsupported device type"))?;
        Ok(Self {
            value: crate::gpu_runtime::share(value)?,
            bytes,
            _allocator: allocator,
            cuda,
        })
    }
    pub(crate) fn new<T: PrimitiveTensorElementType + Debug>(
        allocator: Rc<Allocator>,
        cuda: Rc<Cuda>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        let bytes = shape
            .iter()
            .try_fold(std::mem::size_of::<T>(), |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| anyhow::anyhow!("device buffer size overflow"))?;
        let value = Tensor::<T>::new(&allocator, shape)?.upcast();
        let mut this = Self {
            value,
            bytes,
            _allocator: allocator,
            cuda,
        };
        this.zero()?;
        this.cuda.synchronize()?;
        Ok(this)
    }
    pub(crate) fn upload<T: PrimitiveTensorElementType + Debug>(
        &mut self,
        values: &[T],
    ) -> Result<()> {
        ensure!(
            self.value.dtype().tensor_type() == Some(T::into_tensor_element_type())
                && std::mem::size_of_val(values) == self.bytes,
            "device upload shape/type mismatch"
        );
        unsafe {
            self.cuda.copy(
                self.value.data_ptr_mut(),
                values.as_ptr().cast(),
                self.bytes,
                1,
            )
        }
    }
    pub(crate) fn read<T: PrimitiveTensorElementType + Debug + Default + Clone>(
        &self,
    ) -> Result<Vec<T>> {
        ensure!(
            self.value.dtype().tensor_type() == Some(T::into_tensor_element_type()),
            "device read type mismatch"
        );
        self.cuda.synchronize()?;
        let mut values = vec![T::default(); self.bytes / std::mem::size_of::<T>()];
        unsafe {
            self.cuda.copy(
                values.as_mut_ptr().cast(),
                self.value.data_ptr(),
                self.bytes,
                2,
            )?;
        }
        Ok(values)
    }
    pub(crate) fn copy_value(&mut self, source: &DynTensor) -> Result<()> {
        ensure!(
            self.value.dtype() == source.dtype(),
            "device state copy shape/type mismatch"
        );
        unsafe {
            self.cuda
                .device_copy(self.value.data_ptr_mut(), source.data_ptr(), self.bytes)
        }
    }
    pub(crate) fn zero(&mut self) -> Result<()> {
        unsafe { self.cuda.zero(self.value.data_ptr_mut(), self.bytes) }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = self.cuda.synchronize();
    }
}
