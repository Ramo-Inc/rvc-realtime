//! Normal RMVPE: reflect-centered FFT1024/hop160, HTK mel, half before log.
use anyhow::{ensure, Result};
use half::f16;
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex};
use std::{path::Path, sync::Arc};

fn coefficients(path: &Path, count: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() == count * 4, "invalid coefficient length: {}", path.display());
    let data: Vec<_> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    ensure!(data.iter().all(|v| v.is_finite()), "nonfinite coefficients");
    Ok(data)
}

pub struct Mel {
    fft: Arc<dyn RealToComplex<f32>>,
    basis: Vec<f32>,
    window: Vec<f32>,
    input: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}
impl Mel {
    pub fn new(root: &Path) -> Result<Self> {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(1024);
        let scratch = fft.make_scratch_vec();
        Ok(Self { fft, basis: coefficients(&root.join("mel-basis.f32"), 128*513)?,
            window: coefficients(&root.join("hann.f32"), 1024)?, input: vec![0.; 1024],
            spectrum: vec![Complex::new(0.,0.); 513], scratch })
    }
    pub fn extract(&mut self, audio: &[f16]) -> Result<(Vec<f16>, usize, usize)> {
        ensure!(audio.len() >= 3040, "RMVPE needs at least 190ms");
        let frames = audio.len()/160 + 1;
        let padded = frames.div_ceil(32)*32;
        let mut output = vec![f16::ZERO; 128*padded];
        let mut magnitude = [0f32; 513];
        for frame in 0..frames {
            for n in 0..1024 {
                let index = frame as isize*160 + n as isize - 512;
                let index = if index < 0 { -index } else if index >= audio.len() as isize { 2*audio.len() as isize-2-index } else { index } as usize;
                self.input[n] = audio[index].to_f32()*self.window[n];
            }
            self.fft.process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch)?;
            for (i, value) in self.spectrum.iter().enumerate() { magnitude[i] = (value.re*value.re + value.im*value.im).sqrt(); }
            for channel in 0..128 {
                let value: f32 = self.basis[channel*513..(channel+1)*513].iter().zip(&magnitude).map(|(a,b)| a*b).sum();
                let half = f16::from_f32(value).to_f32().max(f16::from_f32(1e-5).to_f32());
                output[channel*padded+frame] = f16::from_f32(half.ln());
            }
        }
        for channel in 0..128 {
            for frame in frames..padded { output[channel*padded+frame] = output[channel*padded+2*frames-2-frame]; }
        }
        Ok((output, frames, padded))
    }
}
