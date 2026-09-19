//! Pinned Deiteris conversion path, shared by the application and reference checks.
pub mod startup;
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
