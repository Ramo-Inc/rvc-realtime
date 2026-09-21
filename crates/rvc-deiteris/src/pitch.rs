//! TG Fast normal-RMVPE contract. Separate from the historical half Prepare graph.
use anyhow::{ensure, Result};
use half::f16;

/// Decode [frames, 360] salience with the reference's mixed-precision arithmetic.
pub fn decode(hidden: &[f16], threshold: f32) -> Result<Vec<f32>> {
    ensure!(
        hidden.len() % 360 == 0 && threshold.is_finite() && (0.0..=1.0).contains(&threshold),
        "invalid salience shape or threshold"
    );
    ensure!(
        hidden.iter().all(|x| x.is_finite() && x.to_f32() >= 0.0),
        "non-finite or negative salience"
    );
    let threshold = f16::from_f32(threshold).to_f32();
    let mut output = Vec::with_capacity(hidden.len() / 360);
    for frame in hidden.chunks_exact(360) {
        // torch.argmax returns the FIRST maximum, not the last equal value.
        let mut center = 0;
        for i in 1..360 {
            if frame[i] > frame[center] {
                center = i;
            }
        }
        if frame[center].to_f32() < threshold {
            output.push(0.0);
            continue;
        }
        let mut sum = 0.0f32;
        let mut product = 0.0f32;
        for (i, value) in frame
            .iter()
            .enumerate()
            .take((center + 5).min(360))
            .skip(center.saturating_sub(4))
        {
            let weight = value.to_f32();
            sum += weight;
            product += weight * (i as f32 * 20.0 + 1997.3794084376191f32);
        }
        // Reduction result and denominator are half in the normal reference.
        let sum = f16::from_f32(sum).to_f32();
        let denominator = f16::from_f32(sum + if sum == 0.0 { 1.0 } else { 0.0 }).to_f32();
        let cents = product / denominator;
        output.push(10.0 * 2.0f32.powf(cents / 1200.0));
    }
    Ok(output)
}

/// Cache semantics are whole re-estimated-window append, not block-hop overwrite.
pub struct PitchState {
    coarse: Vec<i64>,
    continuous: Vec<f16>,
}

impl PitchState {
    pub fn new(features: usize) -> Result<Self> {
        ensure!(
            (1..100_000).contains(&features),
            "invalid pitch feature length"
        );
        Self::from_cache(vec![0; features + 1], vec![f16::ZERO; features + 1])
    }

    pub fn from_cache(coarse: Vec<i64>, continuous: Vec<f16>) -> Result<Self> {
        ensure!(
            !coarse.is_empty() && coarse.len() == continuous.len(),
            "invalid pitch cache dimensions"
        );
        ensure!(
            coarse.iter().all(|v| (0..=255).contains(v))
                && continuous
                    .iter()
                    .all(|v| v.is_finite() && v.to_f32() >= 0.0),
            "invalid pitch cache values"
        );
        Ok(Self { coarse, continuous })
    }

    pub fn coarse(&self) -> &[i64] {
        &self.coarse
    }
    pub fn continuous(&self) -> &[f16] {
        &self.continuous
    }

    pub fn reset(&mut self) {
        self.coarse.fill(0);
        self.continuous.fill(f16::ZERO);
    }

    pub fn update(&mut self, raw_f0: &[f32], tune: f64, formant: f64) -> Result<()> {
        self.update_inner(raw_f0, tune, formant, None)
    }

    /// PoC for exact 10ms input hops: advance old history by elapsed audio,
    /// then overwrite the entire re-estimated tail using the existing endpoint convention.
    #[cfg(feature = "evaluation")]
    pub fn update_aligned(
        &mut self,
        raw_f0: &[f32],
        tune: f64,
        formant: f64,
        hop: usize,
    ) -> Result<()> {
        ensure!(
            hop > 0 && hop <= raw_f0.len().min(self.coarse.len()),
            "invalid pitch history hop"
        );
        self.update_inner(raw_f0, tune, formant, Some(hop))
    }

    fn update_inner(
        &mut self,
        raw_f0: &[f32],
        tune: f64,
        formant: f64,
        hop: Option<usize>,
    ) -> Result<()> {
        ensure!(
            tune.is_finite()
                && (-48.0..=48.0).contains(&tune)
                && formant.is_finite()
                && (-12.0..=12.0).contains(&formant),
            "invalid pitch shift"
        );
        let factor = 2.0f64.powf((tune - formant) / 12.0) as f32;
        ensure!(
            raw_f0
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0 && f16::from_f32(*v * factor).is_finite()),
            "invalid raw F0"
        );
        let capacity = self.coarse.len();
        let count = raw_f0.len().min(capacity);
        if count == 0 {
            return Ok(());
        }
        let shift = hop.unwrap_or(count);
        self.coarse.copy_within(shift.., 0);
        self.continuous.copy_within(shift.., 0);
        // Constants originate in const.py (NumPy float64, cast by tensor ops).
        let mel_min = (1127.0f64 * (1.0 + 50.0 / 700.0f64).ln()) as f32;
        let mel_range = (1127.0f64 * (1.0 + 1100.0 / 700.0f64).ln()
            - 1127.0f64 * (1.0 + 50.0 / 700.0f64).ln()) as f32;
        for (i, raw) in raw_f0[raw_f0.len() - count..].iter().enumerate() {
            let f0 = *raw * factor;
            let mel = 1127.0 * (1.0 + f0 / 700.0).ln();
            let coarse = ((mel - mel_min) * 254.0 / mel_range + 1.0)
                .clamp(1.0, 255.0)
                .round_ties_even() as i64;
            self.coarse[capacity - count + i] = coarse;
            self.continuous[capacity - count + i] = f16::from_f32(f0);
        }
        Ok(())
    }

    /// Select last F elements first, then apply the formant ratio to stored half F0.
    pub fn generator_pitch(
        &self,
        features: usize,
        formant_ratio: f32,
    ) -> Result<(&[i64], Vec<f16>)> {
        ensure!(
            features > 0
                && features < self.coarse.len()
                && formant_ratio.is_finite()
                && formant_ratio > 0.0,
            "invalid generator pitch slice"
        );
        let start = self.coarse.len() - features;
        let pitchf: Vec<_> = self.continuous[start..]
            .iter()
            .map(|v| f16::from_f32(v.to_f32() * formant_ratio))
            .collect();
        ensure!(
            pitchf.iter().all(|v| v.is_finite()),
            "formant pitch overflow"
        );
        Ok((&self.coarse[start..], pitchf))
    }
}
