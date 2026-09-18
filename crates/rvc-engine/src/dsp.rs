//! Numeric building blocks ported from the official realtime path (torchaudio / librosa / torch ops).
//! Each function names the original it reproduces; `tests/dsp_fixture.rs` checks them against values
//! produced by those originals (`export_onnx.py --dsp-fixture`).

use std::f64::consts::PI;
use std::sync::Arc;

use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `torchaudio.transforms.Resample(orig, new, dtype=torch.float32)` (sinc_interp_hann,
/// lowpass_filter_width=6, rolloff=0.99): kernel from `_get_sinc_resample_kernel`, applied as in
/// `_apply_sinc_resample_kernel` (zero pad `width` / `width + orig`, stride-`orig` conv, ceil length).
pub struct Resampler {
    orig: usize,
    new: usize,
    width: usize,
    taps: usize,
    kernel: Vec<f32>, // new x taps
}

impl Resampler {
    pub fn new(orig_freq: usize, new_freq: usize) -> Self {
        let g = gcd(orig_freq, new_freq);
        let (orig, new) = (orig_freq / g, new_freq / g);
        let lowpass = 6.0f64;
        let base_freq = orig.min(new) as f64 * 0.99;
        let width = (lowpass * orig as f64 / base_freq).ceil() as usize;
        let taps = 2 * width + orig;
        let mut kernel = vec![0f32; new * taps];
        // dtype=float32 in the official call: every tensor op below is float32.
        for j in 0..new {
            for (k, slot) in kernel[j * taps..(j + 1) * taps].iter_mut().enumerate() {
                let idx = (k as f32 - width as f32) / orig as f32;
                let mut t = (-(j as f32)) / new as f32 + idx;
                t *= base_freq as f32;
                t = t.clamp(-lowpass as f32, lowpass as f32);
                let window = (t * (PI as f32) / lowpass as f32 / 2.0).cos().powi(2);
                let tp = t * PI as f32;
                let sinc = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
                *slot = sinc * (window * (base_freq / orig as f64) as f32);
            }
        }
        Self { orig, new, width, taps, kernel }
    }

    pub fn output_len(&self, input_len: usize) -> usize {
        (self.new as f64 * input_len as f64 / self.orig as f64).ceil() as usize
    }

    pub fn process(&self, x: &[f32]) -> Vec<f32> {
        let padded_len = x.len() + 2 * self.width + self.orig;
        let get = |i: usize| -> f32 {
            if i < self.width || i >= self.width + x.len() { 0.0 } else { x[i - self.width] }
        };
        let positions = (padded_len - self.taps) / self.orig + 1;
        let target = self.output_len(x.len());
        let mut out = Vec::with_capacity(target);
        'outer: for p in 0..positions {
            let start = p * self.orig;
            for j in 0..self.new {
                if out.len() == target {
                    break 'outer;
                }
                let kern = &self.kernel[j * self.taps..(j + 1) * self.taps];
                let mut acc = 0f32;
                for (k, w) in kern.iter().enumerate() {
                    acc += w * get(start + k);
                }
                out.push(acc);
            }
        }
        out
    }
}

/// `librosa.feature.rms(y, frame_length, hop_length)` with center=True, pad_mode="constant".
pub fn rms(y: &[f32], frame: usize, hop: usize) -> Vec<f32> {
    let pad = frame / 2;
    let padded = y.len() + 2 * pad;
    let n = 1 + (padded - frame) / hop;
    (0..n)
        .map(|i| {
            let mut acc = 0f64;
            for k in 0..frame {
                let idx = i * hop + k;
                let v = if idx < pad || idx >= pad + y.len() { 0.0 } else { y[idx - pad] as f64 };
                acc += v * v;
            }
            (acc / frame as f64).sqrt() as f32
        })
        .collect()
}

/// `F.interpolate(x[None, None], size=size, mode="linear", align_corners=True)[0, 0]`.
pub fn interp_linear_align_corners(x: &[f32], size: usize) -> Vec<f32> {
    if x.len() == 1 || size == 1 {
        return vec![x[0]; size];
    }
    let scale = (x.len() - 1) as f32 / (size - 1) as f32;
    (0..size)
        .map(|i| {
            let pos = scale * i as f32;
            let i0 = (pos.floor() as usize).min(x.len() - 1);
            let i1 = (i0 + 1).min(x.len() - 1);
            let l = pos - i0 as f32;
            x[i0] * (1.0 - l) + x[i1] * l
        })
        .collect()
}

/// Magnitude STFT, n_fft = win = 1024, hop = 160, periodic Hann (`torch.hann_window`).
pub struct Stft {
    fft: Arc<dyn RealToComplex<f64>>,
    window: Vec<f64>,
}

pub const N_FFT: usize = 1024;
pub const HOP: usize = 160;
pub const BINS: usize = N_FFT / 2 + 1;

impl Stft {
    pub fn new() -> Self {
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(N_FFT);
        let window = (0..N_FFT).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / N_FFT as f64).cos()).collect();
        Self { fft, window }
    }

    /// Frames of `padded` (center=False), magnitude `sqrt(re^2 + im^2 + eps)`, layout [bin][frame].
    fn mags(&self, padded: &[f64], eps: f64) -> (Vec<f32>, usize) {
        let frames = 1 + (padded.len() - N_FFT) / HOP;
        let mut out = vec![0f32; BINS * frames];
        let mut input = vec![0f64; N_FFT];
        let mut spec = vec![Complex::new(0f64, 0f64); BINS];
        for f in 0..frames {
            for n in 0..N_FFT {
                input[n] = padded[f * HOP + n] * self.window[n];
            }
            self.fft.process(&mut input, &mut spec).expect("fft");
            for (b, c) in spec.iter().enumerate() {
                out[b * frames + f] = (c.re * c.re + c.im * c.im + eps).sqrt() as f32;
            }
        }
        (out, frames)
    }

    /// `infer/rmvpe.py MelSpectrogram.forward` STFT part: `torch.stft(center=True)` (reflect pad n_fft/2).
    pub fn rmvpe(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let pad = N_FFT / 2;
        let padded = reflect_pad(audio, pad, pad);
        self.mags(&padded, 0.0)
    }

    /// torchfcpe `MelModule.__call__` STFT part plus `Wav2MelModule` frame-count fix.
    pub fn fcpe(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let pad_left = (N_FFT - HOP) / 2;
        let pad_right = ((N_FFT - HOP + 1) / 2).max((N_FFT as isize - audio.len() as isize - pad_left as isize).max(0) as usize);
        let padded = if pad_right < audio.len() {
            reflect_pad(audio, pad_left, pad_right)
        } else {
            let mut v = vec![0f64; pad_left];
            v.extend(audio.iter().map(|&x| x as f64));
            v.extend(std::iter::repeat(0.0).take(pad_right));
            v
        };
        let (mags, frames) = self.mags(&padded, 1e-9);
        let n_frames = audio.len() / HOP + 1;
        if n_frames == frames {
            return (mags, frames);
        }
        let mut out = vec![0f32; BINS * n_frames];
        for b in 0..BINS {
            for f in 0..n_frames {
                out[b * n_frames + f] = mags[b * frames + f.min(frames - 1)];
            }
        }
        (out, n_frames)
    }
}

impl Default for Stft {
    fn default() -> Self {
        Self::new()
    }
}

fn reflect_pad(x: &[f32], left: usize, right: usize) -> Vec<f64> {
    let n = x.len();
    let mut v = Vec::with_capacity(n + left + right);
    for i in (1..=left).rev() {
        v.push(x[i] as f64);
    }
    v.extend(x.iter().map(|&s| s as f64));
    for i in 0..right {
        v.push(x[n - 2 - i] as f64);
    }
    v
}

/// `infer/rmvpe.py RMVPE.to_local_average_cents` + `decode` (f0 in Hz, 0 = unvoiced).
pub fn rmvpe_decode(hidden: &[f32], frames: usize, thred: f32) -> Vec<f32> {
    const N: usize = 360;
    let cents_pad = |k: usize| -> f64 {
        if (4..4 + N).contains(&k) { 20.0 * (k - 4) as f64 + 1997.3794084376191 } else { 0.0 }
    };
    (0..frames)
        .map(|f| {
            let row = &hidden[f * N..(f + 1) * N];
            let (mut center, mut maxv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in row.iter().enumerate() {
                if v > maxv {
                    maxv = v;
                    center = i;
                }
            }
            let (mut prod, mut weight) = (0f64, 0f64);
            for k in center..center + 9 {
                let s = if (4..4 + N).contains(&k) { row[k - 4] as f64 } else { 0.0 };
                prod += s * cents_pad(k);
                weight += s;
            }
            let cents = if maxv <= thred { 0.0 } else { prod / weight };
            let f0 = 10.0 * 2f64.powf(cents / 1200.0);
            if f0 == 10.0 { 0.0 } else { f0 as f32 }
        })
        .collect()
}

/// `rtrvc.get_f0_*` post-processing: fill unvoiced frames by `np.interp` over voiced ones, then
/// shift by `key` semitones; returns (coarse pitch, f0) as in `rtrvc.get_f0_post`.
pub fn f0_post(mut f0: Vec<f32>, key: f32, f0_min: f32, f0_max: f32) -> (Vec<i64>, Vec<f32>) {
    let voiced: Vec<usize> = (0..f0.len()).filter(|&i| f0[i] != 0.0).collect();
    if !voiced.is_empty() {
        let xs: Vec<f64> = voiced.iter().map(|&i| i as f64).collect();
        let ys: Vec<f64> = voiced.iter().map(|&i| f0[i] as f64).collect();
        for i in 0..f0.len() {
            if f0[i] == 0.0 {
                f0[i] = np_interp(i as f64, &xs, &ys) as f32;
            }
        }
    }
    let factor = 2f32.powf(key / 12.0);
    for v in f0.iter_mut() {
        *v *= factor;
    }
    let mel_min = 1127.0 * (1.0 + f0_min / 700.0).ln();
    let mel_max = 1127.0 * (1.0 + f0_max / 700.0).ln();
    let coarse = f0
        .iter()
        .map(|&v| {
            let mut mel = 1127.0 * (1.0 + v / 700.0).ln();
            if mel > 0.0 {
                mel = (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0;
            }
            if mel <= 1.0 {
                mel = 1.0;
            }
            if mel > 255.0 {
                mel = 255.0;
            }
            round_half_even(mel) as i64
        })
        .collect();
    (coarse, f0)
}

fn np_interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let j = xs.partition_point(|&v| v <= x);
    let (x0, x1, y0, y1) = (xs[j - 1], xs[j], ys[j - 1], ys[j]);
    y0 + (x - x0) * (y1 - y0) / (x1 - x0)
}

fn round_half_even(v: f32) -> f32 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - v.signum() } else { r }
}
