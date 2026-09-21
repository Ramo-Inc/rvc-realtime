//! CPU-only probe of per-block right-padding versus the same FIR with future input.
use anyhow::{ensure, Result};
use rvc_deiteris::{input::InputTimeline, startup::ResampleInputs};

fn main() -> Result<()> {
    let out = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: input_continuity REPORT"))?;
    ensure!(!std::path::Path::new(&out).exists(), "report must be new");
    let mut rows = Vec::new();
    for rate in [48000usize, 44100] {
        for block in [2432usize, 7680] {
            for delayed in [false, true] {
            let samples = block * 30;
            let input: Vec<f32> = (0..samples).map(|i| {
                let t = i as f64 / rate as f64;
                (0.15 * (std::f64::consts::TAU * 173. * t).sin()
                    + 0.05 * (std::f64::consts::TAU * 1729. * t).sin()) as f32
            }).collect();
            let coefficients = ResampleInputs::new(rate, 16000, samples)?;
            let grid = rate * coefficients.phases / 16000;
            let width = (coefficients.taps - grid) / 2;
            let mut streaming = if delayed { InputTimeline::new_delayed(rate, block)? } else { InputTimeline::new(rate, block)? };
            let offset = width + if delayed { width } else { 0 };
            let mut actual = Vec::new();
            let mut boundaries = Vec::new();
            for chunk in input.chunks(block) {
                actual.extend_from_slice(streaming.push(chunk)?);
                boundaries.push(actual.len());
            }
            let mut reference = Vec::new();
            for n in 0..actual.len() {
                let origin = n / coefficients.phases * grid;
                let phase = n % coefficients.phases;
                let mut value = 0.0;
                for tap in 0..coefficients.taps {
                    if origin + tap >= offset {
                        if let Some(sample) = input.get(origin + tap - offset) {
                            value += sample * coefficients.kernel[phase * coefficients.taps + tap];
                        }
                    }
                }
                reference.push(value);
            }
            let mut edge_error = 0.0f64;
            let mut signal = 0.0f64;
            let mut max_error = 0.0f32;
            let mut outside_max = 0.0f32;
            // Exclude startup/end and identify the last 1ms before each boundary.
            for i in boundaries[0]..boundaries[boundaries.len()-2] {
                let error = actual[i] - reference[i];
                if boundaries.iter().any(|&b| i < b && b-i <= 16) {
                    edge_error += (error as f64).powi(2);
                    signal += (reference[i] as f64).powi(2);
                    max_error = max_error.max(error.abs());
                } else { outside_max = outside_max.max(error.abs()); }
            }
            rows.push(serde_json::json!({"rate":rate,"block":block,"fir_width":width,
                "delayed":delayed,"added_signal_delay_ms":if delayed { width as f64 * 1000. / rate as f64 } else { 0. },
                "boundary_max_error":max_error,"outside_boundary_max_error":outside_max,
                "boundary_snr_db":10.*(signal/edge_error).log10(),
                "scope":"173+1729Hz synthetic input, same coefficients and operation order; quality is not inferred"}));
            if delayed { ensure!(max_error < 1e-6 && outside_max < 1e-6, "delayed FIR remains block-dependent"); }
            }
        }
    }
    std::fs::write(out, serde_json::to_vec_pretty(&rows)?)?;
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}
