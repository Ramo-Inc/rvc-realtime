//! Headless, actual CUDA preprocessor vs the independently tested input timeline.
#[cfg(feature = "onnx")]
fn main() -> anyhow::Result<()> {
    use anyhow::ensure;
    use half::f16;
    use rvc_deiteris::{gpu_pre::GpuPre, input::InputTimeline, startup::Startup};
    use std::path::PathBuf;
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4 || args.len() == 5,
        "usage: check_gpu_pipeline ASSETS RUNTIME REPORT [ACTUAL_TG_FIXTURE]"
    );
    let assets = PathBuf::from(&args[1]);
    let runtime = std::path::absolute(&args[2])?;
    let output = PathBuf::from(&args[3]);
    ensure!(!output.exists(), "report must be new");
    if std::env::var_os("RVC_GPU_PITCH_DIAGNOSTICS").is_some() {
        ort::init()
            .with_logger(std::sync::Arc::new(|level, _, _, location, message| {
                eprintln!("ORT {level:?} {location}: {message}")
            }))
            .commit();
    }
    let old = std::env::var_os("PATH").unwrap_or_default();
    std::env::set_var(
        "PATH",
        std::env::join_paths(std::iter::once(runtime.clone()).chain(std::env::split_paths(&old)))?,
    );
    if args.len() == 5 {
        return check_actual(&assets, &runtime, &output, &PathBuf::from(&args[4]));
    }
    let mut rows = Vec::new();
    let mut passed = true;
    for (rate, block) in [(48000, 2432), (48000, 2944), (44100, 2432)] {
        let startup = Startup {
            sample_rate: rate,
            chunk: 19,
            block_frames: Some(block),
            extra_ms: 500.,
            crossfade_ms: 100.,
            formant: -0.2,
        };
        for graph in [false, true] {
            let mut reference = InputTimeline::new(rate, block)?;
            let mut gpu = GpuPre::new(&runtime, &assets, &startup, 40000, graph)?;
            let dims = startup.tg_dims(40000)?;
            let mut context = vec![f16::ZERO; dims.convert];
            let mut max_audio = 0f32;
            let mut max_context = 0f32;
            for b in 0..90 {
                let input: Vec<_> = (0..block)
                    .map(|i| {
                        if (30..60).contains(&b) {
                            0.
                        } else {
                            ((b * block + i) as f32 * std::f32::consts::TAU * 180. / rate as f32)
                                .sin()
                                * 0.05
                        }
                    })
                    .collect();
                let expected = reference.push(&input)?.to_vec();
                let count = gpu.process(&input, 0.00001)?;
                ensure!(
                    count == expected.len() && gpu.total_output() == reference.total_output(),
                    "continuous output grid changed"
                );
                let observed = gpu.audio16()?;
                for (a, b) in observed.iter().zip(&expected) {
                    max_audio = max_audio.max((a.to_f32() - f16::from_f32(*b).to_f32()).abs());
                }
                let count = count.min(context.len());
                let tail = context.len() - count;
                context.copy_within(count.., 0);
                for (dst, src) in context[tail..]
                    .iter_mut()
                    .zip(&expected[expected.len() - count..])
                {
                    *dst = f16::from_f32(*src);
                }
                let observed = gpu.context()?;
                for (a, b) in observed.iter().zip(&context) {
                    max_context = max_context.max((a.to_f32() - b.to_f32()).abs());
                }
            }
            gpu.reset()?;
            gpu.process(&vec![0.; block], 0.00001)?;
            let reset_zero = gpu.context()?.iter().all(|v| *v == f16::ZERO);
            let pass = max_audio <= 1e-4 && max_context <= 1e-4 && reset_zero;
            passed &= pass;
            rows.push(serde_json::json!({"rate":rate,"block":block,"graph":graph,"blocks":90,"max_audio_abs":max_audio,"max_context_abs":max_context,"reset_zero":reset_zero,"pass":pass}));
            std::fs::write(
                &output,
                serde_json::to_vec_pretty(
                    &serde_json::json!({"complete":false,"pass":false,"rows":rows}),
                )?,
            )?;
        }
    }
    std::fs::write(
        output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"scope":"GPU pre only; not full pipeline or waveform acceptance","complete":true,"pass":passed,"rows":rows}),
        )?,
    )?;
    ensure!(passed, "GPU continuous-input gate failed");
    Ok(())
}

#[cfg(feature = "onnx")]
fn check_actual(assets: &std::path::Path, runtime: &std::path::Path, output: &std::path::Path, fixture: &std::path::Path) -> anyhow::Result<()> {
    use anyhow::ensure;
    use rvc_deiteris::{gpu_pre::GpuPre, startup::{Startup, ResampleInputs}};
    #[derive(serde::Deserialize)]
    struct Case { rate: usize, block: usize, input: Vec<Vec<f32>>, output: Vec<Vec<f32>>, kernel: Vec<f32>, repeat_exact: bool }
    #[derive(serde::Deserialize)]
    struct Fixture { schema: u32, rows: Vec<Case> }
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(fixture)?)?;
    ensure!(fixture.schema == 1 && fixture.rows.len() == 3, "invalid actual TG fixture");
    let mut rows = Vec::new();
    let mut pass = true;
    for case in fixture.rows {
        ensure!(case.repeat_exact && !case.input.is_empty() && case.input.len() == case.output.len(), "invalid reference replay");
        let coeff = ResampleInputs::new(case.rate, 16000, 1)?;
        ensure!(coeff.kernel.len() == case.kernel.len(), "kernel shape mismatch");
        let kernel_diff = coeff.kernel.iter().zip(&case.kernel).filter(|(a,b)| a != b).count();
        let kernel_abs = coeff.kernel.iter().zip(&case.kernel).map(|(a,b)| (a-b).abs()).fold(0f32, f32::max);
        for graph in [false, true] {
            let startup = Startup { sample_rate: case.rate, chunk: 19, block_frames: Some(case.block), extra_ms: 500., crossfade_ms: 100., formant: -0.2 };
            let mut gpu = GpuPre::new(runtime, assets, &startup, 40000, graph)?;
            let mut different = 0;
            let mut max_abs = 0f32;
            for (input, expected) in case.input.iter().zip(&case.output) {
                ensure!(expected.iter().all(|v| v.is_finite()), "nonfinite reference");
                ensure!(gpu.process(input, 0.00001)? == expected.len(), "actual TG sample count mismatch");
                for (a,b) in gpu.audio16()?.iter().zip(expected) {
                    different += usize::from(a.to_f32() != *b);
                    max_abs = max_abs.max((a.to_f32() - b).abs());
                }
            }
            drop(gpu);
            pass &= different == 0;
            rows.push(serde_json::json!({"rate":case.rate,"block":case.block,"graph":graph,"different":different,"max_abs":max_abs,"kernel_different":kernel_diff,"kernel_max_abs":kernel_abs}));
        }
    }
    std::fs::write(output, serde_json::to_vec_pretty(&serde_json::json!({"scope":"actual TG streaming CUDA resampler, half output exact","complete":true,"pass":pass,"rows":rows}))?)?;
    ensure!(pass, "actual TG resampler exact gate failed");
    Ok(())
}
#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("requires --features onnx");
    std::process::exit(1);
}
