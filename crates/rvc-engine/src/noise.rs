//! Loudness of one 10 ms slot, which is how the engine decides whether the input is silent.

/// Loudness of one slot in dB (`20 log10(rms)`), on its own samples only. The gate's
/// `amplitude_to_db` clips 80 dB below the loudest frame of its window, so its numbers depend on the
/// window; a decision about a single slot needs this absolute one instead.
pub fn slot_db(slot: &[f32]) -> f32 {
    let mean = slot.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / slot.len() as f64;
    (10.0 * mean.max(1e-20).log10()) as f32
}
