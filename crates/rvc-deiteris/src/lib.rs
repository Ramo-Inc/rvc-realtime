//! Pinned Deiteris conversion path, shared by the application and reference checks.
pub mod startup;
pub mod input;
#[cfg(feature = "onnx")]
pub mod pitch;
#[cfg(feature = "onnx")]
pub mod tg;
#[cfg(feature = "onnx")]
pub mod mel;
#[cfg(feature = "onnx")]
pub mod gpu_pitch;
#[cfg(feature = "onnx")]
mod gpu_runtime;
#[cfg(feature = "onnx")]
mod gpu_buffer;
#[cfg(feature = "onnx")]
pub mod gpu_pre;
#[cfg(feature = "onnx")]
mod gpu_noise;
#[cfg(feature = "onnx")]
pub mod gpu_pipeline;
#[cfg(all(feature = "native", feature = "onnx"))]
compile_error!("native is a reference-only backend; do not link it into the ONNX product");
#[cfg(feature = "native")]
mod generator;
#[cfg(feature = "native")]
pub use generator::Generator;
#[cfg(feature = "onnx")]
mod onnx_generator;
#[cfg(feature = "onnx")]
pub use onnx_generator::Generator;
#[cfg(any(feature = "native", feature = "onnx"))]
pub mod engine;
