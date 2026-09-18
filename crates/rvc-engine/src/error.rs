use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum Error {
    /// model.json or an ONNX file is missing or unreadable
    ModelFiles { path: PathBuf, reason: String },
    /// DLL, GPU or session initialisation failed
    Runtime(String),
    /// a startup parameter is outside the official range
    Startup(String),
    /// audio device not found or the stream could not be opened
    Audio(String),
    BlockSize { expected: usize, got: usize },
    /// ONNX Runtime failed while running a block
    Inference(String),
    /// the voice model cannot be converted
    Unsupported(crate::voice_model::Unsupported),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ModelFiles { path, reason } => write!(f, "{}: {reason}", path.display()),
            Error::Runtime(s) => write!(f, "runtime: {s}"),
            Error::Startup(s) => write!(f, "startup parameter: {s}"),
            Error::Audio(s) => write!(f, "audio: {s}"),
            Error::BlockSize { expected, got } => write!(f, "block of {got} samples, engine expects {expected}"),
            Error::Inference(s) => write!(f, "inference: {s}"),
            Error::Unsupported(u) => write!(f, "unsupported voice model: {u}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
