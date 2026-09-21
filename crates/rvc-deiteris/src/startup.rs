//! Original RVCr2 lengths and torchaudio coefficient construction, at startup only.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Startup {
    pub sample_rate: usize,
    pub chunk: usize,
    /// Resolved device frames for unit-tagged settings; None preserves old chunk*128.
    #[serde(default)]
    pub block_frames: Option<usize>,
    pub extra_ms: f64,
    pub crossfade_ms: f64,
    pub formant: f64,
}

#[derive(Debug, Serialize)]
pub struct Dims {
    pub block: usize,
    pub crossfade: usize,
    pub search: usize,
    pub audio: usize,
    pub convert: usize,
    pub features: usize,
    pub silence_front: usize,
    pub skip: usize,
    pub ret: usize,
    pub formant_length: usize,
    pub scaled_window: usize,
    pub generated: usize,
    pub trimmed: usize,
}

impl Startup {
    pub fn dims(&self, model_rate: usize) -> Result<Dims> {
        self.compute(model_rate, false)
    }
    pub fn tg_dims(&self, model_rate: usize) -> Result<Dims> {
        self.compute(model_rate, true)
    }
    fn compute(&self, model_rate: usize, tg: bool) -> Result<Dims> {
        ensure!(
            matches!(self.sample_rate, 44100 | 48000),
            "unsupported device rate"
        );
        ensure!(
            matches!(model_rate, 32000 | 40000 | 48000),
            "unsupported model rate"
        );
        ensure!(
            (1..=256).contains(&self.chunk),
            "chunk out of supported range"
        );
        ensure!(
            self.extra_ms.is_finite() && ((if tg { 0.0 } else { 50.0 })..=5000.0).contains(&self.extra_ms),
            "invalid extra context"
        );
        ensure!(
            self.crossfade_ms.is_finite() && (1.0..=1000.0).contains(&self.crossfade_ms),
            "invalid crossfade"
        );
        ensure!(
            self.formant.is_finite() && (-12.0..=12.0).contains(&self.formant),
            "invalid formant"
        );
        let block = self.block_frames.unwrap_or(self.chunk * 128);
        ensure!((128..=32768).contains(&block), "invalid resolved block frames");
        let crossfade = (self.crossfade_ms / 1000.0 * self.sample_rate as f64) as usize;
        let extra = (self.extra_ms / 1000.0 * self.sample_rate as f64) as usize;
        let search = self.sample_rate / 100;
        let to16 = |n: usize| (n as f64 / self.sample_rate as f64 * 16000.0) as usize;
        let mut convert =
            (to16(block) + to16(crossfade) + to16(extra) + to16(search)).div_ceil(160) * 160;
        let mut skip = to16(extra) / 160;
        let ret = convert / 160 - skip;
        let mut silence_front = to16(extra).saturating_sub(800);
        if tg {
            if convert < 3040 {
                skip += (3040 - convert) / 160;
                convert = 3040;
            }
            silence_front = silence_front.min(convert - 3040);
        }
        let features = convert / 160;
        let window = model_rate / 100;
        let factor = 2.0f64.powf(self.formant / 12.0);
        let formant_length = (ret as f64 * factor).ceil() as usize;
        let scaled_window = (factor * window as f64).floor() as usize;
        let generated = formant_length * window;
        let trimmed = if scaled_window == window {
            generated
        } else {
            ret * scaled_window
        };
        let dims = Dims {
            block,
            crossfade,
            search,
            audio: to16(block) + to16(crossfade),
            convert,
            features,
            silence_front,
            skip,
            ret,
            formant_length,
            scaled_window,
            generated,
            trimmed,
        };
        ensure!(
            dims.trimmed <= dims.generated && dims.convert - dims.silence_front > 512,
            "invalid processing window"
        );
        Ok(dims)
    }
}

pub struct ResampleInputs {
    pub phases: usize,
    pub taps: usize,
    pub frames: usize,
    pub kernel: Vec<f32>,
    pub indices: Vec<i64>,
    pub output_length: usize,
}

/// VoiceChangerV2._generate_strength (float32 linspace, sin squared).
pub fn fade_windows(samples: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    ensure!(samples > 1 && samples <= 192000, "invalid crossfade length");
    let step = 1.0f32 / (samples - 1) as f32;
    let fade: Vec<f32> = (0..samples)
        .map(|i| {
            let position = if i < samples / 2 {
                step * i as f32
            } else {
                1.0 - step * (samples - i - 1) as f32
            };
            (position * (std::f64::consts::PI * 0.5) as f32)
                .sin()
                .powi(2)
        })
        .collect();
    let inverse = fade.iter().map(|&x| 1.0 - x).collect();
    Ok((fade, inverse))
}

impl ResampleInputs {
    pub fn new(source: usize, target: usize, samples: usize) -> Result<Self> {
        ensure!(
            source > 0
                && target > 0
                && source <= 192000
                && target <= 192000
                && samples > 0
                && samples <= 1000000,
            "invalid resample dimensions"
        );
        if source == target {
            return Ok(Self {
                phases: 1,
                taps: 1,
                frames: samples,
                kernel: vec![1.0],
                indices: (1..=samples as i64).collect(),
                output_length: samples,
            });
        }
        let (mut a, mut b) = (source, target);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let (original, phases) = (source / a, target / a);
        let base = original.min(phases) as f64 * 0.99;
        let width = (6.0 * original as f64 / base).ceil() as usize;
        let taps = 2 * width + original;
        let frames = samples / original + 1;
        ensure!(
            phases * taps <= 16000000 && frames * taps <= 16000000,
            "resampling allocation exceeds bound"
        );
        let mut kernel = Vec::with_capacity(phases * taps);
        // Same operation boundaries as dsp.rs::Resampler. Upstream creates the
        // float32 kernel on CPU before moving it to CUDA.
        for phase in 0..phases {
            for tap in 0..taps {
                let idx = (tap as f32 - width as f32) / original as f32;
                let t = ((-(phase as f32) / phases as f32 + idx) * base as f32).clamp(-6.0, 6.0);
                let window = (t * std::f32::consts::PI / 6.0 / 2.0).cos().powi(2);
                let angle = t * std::f32::consts::PI;
                let sinc = if angle == 0.0 {
                    1.0
                } else {
                    angle.sin() / angle
                };
                kernel.push(sinc * (window * (base / original as f64) as f32));
            }
        }
        let mut indices = Vec::with_capacity(frames * taps);
        for frame in 0..frames {
            for tap in 0..taps {
                let position = (frame * original + tap) as i64 - width as i64;
                indices.push(if position >= 0 && position < samples as i64 {
                    position + 1
                } else {
                    0
                });
            }
        }
        Ok(Self {
            phases,
            taps,
            frames,
            kernel,
            indices,
            output_length: ((samples as f64 * phases as f64 / original as f64) as f32).ceil()
                as usize,
        })
    }
}
