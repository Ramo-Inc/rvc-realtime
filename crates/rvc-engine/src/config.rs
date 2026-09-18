//! Startup configuration (trial 02): the voice model's `model.json` from `tools/export_onnx_dynamic.py`
//! plus the realtime parameters the official VST exposes. Buffer dimensions are computed here with the
//! official expressions (`RVCStreamEngine.__init__`, `rtrvc.RVC.infer`, `SynthesizerTrnMs256NSFsid.infer`,
//! HuBERT conv lengths); `tests/dims_fixture.rs` checks them against the official engine.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};

/// Largest block the official VST sends to its worker (`WorkerClient.cpp` kMaxFrames).
pub const MAX_BLOCK_FRAMES: usize = 131072;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelFiles {
    pub contentvec: String,
    pub rmvpe: String,
    pub fcpe: String,
    pub generator: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub model_sr: usize,
    pub upp: usize,
    pub half: bool,
    pub f0_min: f32,
    pub f0_max: f32,
    pub pitch_cache_len: usize,
    pub inter_channels: usize,
    pub files: ModelFiles,
    #[serde(skip)]
    pub dir: PathBuf,
}

impl Model {
    /// Reads model.json and checks that the four ONNX files exist.
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("model.json");
        let text = std::fs::read_to_string(&path).map_err(|e| Error::ModelFiles { path: path.clone(), reason: e.to_string() })?;
        let mut m: Model = serde_json::from_str(&text).map_err(|e| Error::ModelFiles { path: path.clone(), reason: e.to_string() })?;
        m.dir = dir.to_path_buf();
        for file in [&m.files.contentvec, &m.files.rmvpe, &m.files.fcpe, &m.files.generator] {
            let onnx = dir.join(file);
            if !onnx.is_file() {
                return Err(Error::ModelFiles { path: onnx, reason: "not found".into() });
            }
        }
        Ok(m)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F0Method {
    Rmvpe,
    Fcpe,
}

/// Parameters fixed at engine start. Ranges are the official ones (`RVCRealtime.cpp` InitInt / InitDouble).
#[derive(Debug, Clone, Copy)]
pub struct Startup {
    pub sample_rate: u32,
    pub block_ms: f64,
    pub crossfade_ms: f64,
    pub extra_ms: f64,
    pub formant: f64,
    pub f0: F0Method,
}

#[derive(Debug, Clone)]
pub struct Dims {
    pub device_sr: usize,
    pub model_sr: usize,
    pub zc: usize,
    pub block_frame: usize,
    pub block_frame_16k: usize,
    pub crossfade_frame: usize,
    pub sola_buffer_frame: usize,
    pub sola_search_frame: usize,
    pub extra_frame: usize,
    pub input_wav_len: usize,
    pub n16: usize,
    pub p_len: usize,
    pub skip_head: usize,
    pub return_length: usize,
    pub return_length2: usize,
    pub upp_res: usize,
    pub flow_head: usize,
    pub f0_extractor_frame: usize,
    pub f0_frames: usize,
    pub f0_min: f32,
    pub f0_max: f32,
    pub pitch_cache_len: usize,
    pub feats_frames: usize,
}

/// Python `float // float`: CPython float_floor_div (fmod based), not `(a / b).floor()`.
fn py_floordiv(vx: f64, wx: f64) -> f64 {
    let mut m = vx % wx;
    let mut div = (vx - m) / wx;
    if m != 0.0 {
        if (wx < 0.0) != (m < 0.0) {
            m += wx;
            div -= 1.0;
        }
    }
    let _ = m;
    if div != 0.0 {
        let mut floordiv = div.floor();
        if div - floordiv > 0.5 {
            floordiv += 1.0;
        }
        floordiv
    } else {
        0.0f64.copysign(vx / wx)
    }
}

/// `int(round(ms / 1000 * sr / zc) * zc)` — Python round() is half-to-even.
fn ms_frames(ms: f64, sr: usize, zc: usize) -> usize {
    ((ms / 1000.0 * sr as f64 / zc as f64).round_ties_even() as usize) * zc
}

/// `2 ** (formant / 12)`
pub fn formant_factor(formant: f64) -> f64 {
    2f64.powf(formant / 12.0)
}

/// `int(np.ceil(return_length * factor))` (rtrvc.infer)
pub fn return_length2(return_length: usize, formant: f64) -> usize {
    (return_length as f64 * formant_factor(formant)).ceil() as usize
}

/// `int(np.floor(factor * tgt_sr // 100))` (rtrvc.infer)
pub fn upp_res(model_sr: usize, formant: f64) -> usize {
    py_floordiv(formant_factor(formant) * model_sr as f64, 100.0).floor() as usize
}

/// HuBERT feature-extractor output length (conv kernels 10,3,3,3,3,2,2 / strides 5,2,2,2,2,2,2).
pub fn feats_frames(n16: usize) -> usize {
    [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)].iter().fold(n16, |l, (k, s)| (l - k) / s + 1)
}

impl Startup {
    pub fn validate(&self) -> Result<()> {
        let ranges = [("block_ms", self.block_ms, 20.0, 1000.0), ("crossfade_ms", self.crossfade_ms, 10.0, 100.0),
                      ("extra_ms", self.extra_ms, 500.0, 3000.0), ("formant", self.formant, -12.0, 12.0)];
        for (name, v, lo, hi) in ranges {
            if !(lo..=hi).contains(&v) {
                return Err(Error::Startup(format!("{name} {v} is outside the official range {lo}..={hi}")));
            }
        }
        if self.sample_rate < 100 {
            return Err(Error::Startup(format!("sample rate {} too low", self.sample_rate)));
        }
        Ok(())
    }
}

impl Dims {
    pub fn compute(model: &Model, s: &Startup) -> Result<Self> {
        s.validate()?;
        let sr = s.sample_rate as usize;
        // RVCStreamEngine.__init__
        let zc = (sr / 100).max(1);
        let block_frame = ms_frames(s.block_ms, sr, zc);
        if block_frame > MAX_BLOCK_FRAMES {
            return Err(Error::Startup(format!("block of {block_frame} samples exceeds the official limit {MAX_BLOCK_FRAMES}")));
        }
        let block_frame_16k = 160 * block_frame / zc;
        let crossfade_frame = ms_frames(s.crossfade_ms, sr, zc);
        let sola_buffer_frame = crossfade_frame.min(4 * zc);
        let sola_search_frame = zc;
        let extra_frame = ms_frames(s.extra_ms, sr, zc);
        let input_wav_len = extra_frame + crossfade_frame + sola_search_frame + block_frame;
        let n16 = 160 * input_wav_len / zc;
        let skip_head = extra_frame / zc;
        let return_length = (block_frame + sola_buffer_frame + sola_search_frame) / zc;
        // rtrvc.RVC.infer
        let p_len = n16 / 160;
        let mut f0_extractor_frame = block_frame_16k + 800;
        if s.f0 == F0Method::Rmvpe {
            f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) / 5120 + 1) - 160;
        }
        Ok(Self {
            device_sr: sr,
            model_sr: model.model_sr,
            zc,
            block_frame,
            block_frame_16k,
            crossfade_frame,
            sola_buffer_frame,
            sola_search_frame,
            extra_frame,
            input_wav_len,
            n16,
            p_len,
            skip_head,
            return_length,
            return_length2: return_length2(return_length, s.formant),
            upp_res: upp_res(model.model_sr, s.formant),
            // SynthesizerTrnMs256NSFsid.infer
            flow_head: skip_head.saturating_sub(24),
            f0_extractor_frame,
            f0_frames: f0_extractor_frame / 160 + 1,
            f0_min: model.f0_min,
            f0_max: model.f0_max,
            pitch_cache_len: model.pitch_cache_len,
            feats_frames: feats_frames(n16),
        })
    }
}

/// Everything the engine needs: the model files and the dimensions for this startup configuration.
pub(crate) struct Setup<'a> {
    pub model: &'a Model,
    pub startup: &'a Startup,
    pub dims: Dims,
}

impl Setup<'_> {
    pub fn model_path(&self, file: &str) -> PathBuf {
        self.model.dir.join(file)
    }
}
