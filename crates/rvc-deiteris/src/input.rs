//! Past FIR context on a continuous 16 kHz sample grid; no future-input waiting.
use crate::startup::ResampleInputs;
use anyhow::{ensure, Result};

pub struct InputTimeline {
    source: usize,
    block: usize,
    grid: usize,
    phases: usize,
    width: usize,
    delay_samples: usize,
    taps: usize,
    kernel: Vec<f32>,
    history_frames: usize,
    history: Vec<f32>,
    start: u64,
    total_input: u64,
    total_output: u64,
    output: Vec<f32>,
}

impl InputTimeline {
    pub fn new(source: usize, block: usize) -> Result<Self> {
        ensure!(
            matches!(source, 44100 | 48000) && (1..=32768).contains(&block),
            "unsupported input dimensions"
        );
        let coefficients = ResampleInputs::new(source, 16000, 1)?;
        let mut gcd = source;
        let mut other = 16000;
        while other != 0 {
            (gcd, other) = (other, gcd % other);
        }
        let grid = source / gcd;
        let width = (coefficients.taps - grid) / 2;
        let history_frames = 1920.max(2 * width * grid);
        Ok(Self {
            source,
            block,
            grid,
            phases: coefficients.phases,
            width,
            delay_samples: 0,
            taps: coefficients.taps,
            kernel: coefficients.kernel,
            history_frames,
            history: Vec::with_capacity(history_frames + grid + block),
            start: 0,
            total_input: 0,
            total_output: 0,
            output: Vec::with_capacity((block * 16000).div_ceil(source) + 1),
        })
    }

    /// PoC: shift the FIR by its support radius instead of treating the end of
    /// every input block as silence. Adds width/source seconds of signal delay;
    /// does not wait for another block or change the sample-count timeline.
    #[cfg(feature = "evaluation")]
    pub fn new_delayed(source: usize, block: usize) -> Result<Self> {
        let mut input = Self::new(source, block)?;
        input.delay_samples = input.width;
        Ok(input)
    }

    pub fn total_output(&self) -> u64 {
        self.total_output
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.output.clear();
        self.start = 0;
        self.total_input = 0;
        self.total_output = 0;
    }

    pub fn push(&mut self, block: &[f32]) -> Result<&[f32]> {
        ensure!(
            block.len() == self.block && block.iter().all(|v| v.is_finite()),
            "invalid input block"
        );
        let end = self
            .total_input
            .checked_add(block.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("input timeline overflow"))?;
        let end_out = end
            .checked_mul(16000)
            .ok_or_else(|| anyhow::anyhow!("output timeline overflow"))?
            .div_ceil(self.source as u64);
        self.history.extend_from_slice(block);
        self.output.clear();
        let start_out = self.start * 16000 / self.source as u64;
        let offset = self.width + self.delay_samples;
        for absolute in self.total_output..end_out {
            let position = (absolute - start_out) as usize;
            let frame = position / self.phases;
            let phase = position % self.phases;
            let origin = frame * self.grid;
            let coefficients = &self.kernel[phase * self.taps..(phase + 1) * self.taps];
            let mut value = 0.0;
            for (tap, coefficient) in coefficients.iter().enumerate() {
                let index = origin + tap;
                if index >= offset {
                    if let Some(sample) = self.history.get(index - offset) {
                        value += sample * coefficient;
                    }
                }
            }
            self.output.push(value);
        }
        let keep_start = end.saturating_sub(self.history_frames as u64);
        let keep_start = keep_start - keep_start % self.grid as u64;
        let discard = (keep_start - self.start) as usize;
        self.history.copy_within(discard.., 0);
        self.history.truncate(self.history.len() - discard);
        self.start = keep_start;
        self.total_input = end;
        self.total_output = end_out;
        Ok(&self.output)
    }
}
