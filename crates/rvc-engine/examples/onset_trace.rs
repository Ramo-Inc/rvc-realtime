//! Short diagnostic capture; no audio devices, not a performance benchmark.
use anyhow::{ensure, Result};
use clap::Parser;
use half::f16;
use rvc_deiteris::{
    engine::Params,
    startup::Startup,
    tg::{Engine, Trial},
};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    assets: PathBuf,
    #[arg(long)]
    runtime: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    delayed_input: bool,
    #[arg(long, conflicts_with = "delayed_input")]
    precise_volume: bool,
    #[arg(long, conflicts_with_all = ["delayed_input", "precise_volume", "replay"])]
    aligned_pitch_history: bool,
    /// Replay saved c1/delayed generator inputs, changing pitch and features separately.
    #[arg(long)]
    replay: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    blocks: usize,
}

fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        !a.out.exists() && a.blocks > 0,
        "output must be new, blocks positive"
    );
    let runtime = a.runtime.canonicalize()?;
    let old = std::env::var_os("PATH").unwrap_or_default();
    std::env::set_var(
        "PATH",
        std::env::join_paths(std::iter::once(runtime.clone()).chain(std::env::split_paths(&old)))?,
    );
    let mut reader = hound::WavReader::open(&a.input)?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1 && spec.sample_format == hound::SampleFormat::Float,
        "mono float input required"
    );
    let samples = reader
        .samples::<f32>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(
        !samples.is_empty() && samples.iter().all(|x| x.is_finite()),
        "invalid input"
    );
    let startup = Startup {
        sample_rate: spec.sample_rate as usize,
        chunk: 60,
        block_frames: None,
        extra_ms: 3000.,
        crossfade_ms: 30.,
        formant: -0.2,
    };
    if let Some(root) = &a.replay {
        return replay(&a, root, &startup);
    }
    let mut engine = Engine::new_trial(
        &a.model,
        &a.assets.join("tg-fast-v1"),
        &a.assets.join("tg-gpu-pitch-v2"),
        &runtime,
        startup.clone(),
        true,
        Trial {
            delayed_input: a.delayed_input,
            precise_volume: a.precise_volume,
            aligned_pitch_history: a.aligned_pitch_history,
            ..Trial::default()
        },
    )?;
    engine.prepare(1)?;
    std::fs::create_dir_all(&a.out)?;
    let mut output = hound::WavWriter::create(a.out.join("output.wav"), spec)?;
    let mut rows = Vec::new();
    let mut input = vec![0.; engine.block_frames()];
    for (block, part) in samples.chunks(input.len()).take(a.blocks).enumerate() {
        input.fill(0.);
        input[..part.len()].copy_from_slice(part);
        let mut stats = BTreeMap::new();
        let out=engine.process_observed(&input,&Params{pitch:14.,threshold_db:-83.},&mut |name,value| {
            if !matches!(name,"audio16"|"volume"|"active"|"rawf0"|"features"|"pitch"|"pitchf"|"generated"|"pre_sola"|"sola_offset"|"output") {return Ok(());}
            let data=if let Ok((_,v))=value.try_extract_tensor::<f32>() {v.to_vec()}
                else if let Ok((_,v))=value.try_extract_tensor::<f16>() {v.iter().map(|v| v.to_f32()).collect()}
                else if let Ok((_,v))=value.try_extract_tensor::<i64>() {v.iter().map(|v| *v as f32).collect()}
                else {anyhow::bail!("unexpected type: {name}")};
            ensure!(!data.is_empty() && data.iter().all(|x|x.is_finite()),"invalid {name}");
            let rms=(data.iter().map(|x|(*x as f64).powi(2)).sum::<f64>()/data.len() as f64).sqrt();
            let bytes:Vec<_>=data.iter().flat_map(|x|x.to_le_bytes()).collect();
            std::fs::write(a.out.join(format!("{block}-{name}.f32")),bytes)?;
            stats.insert(name.to_string(),serde_json::json!({"len":data.len(),"rms":rms,"max_abs":data.iter().map(|x|x.abs()).fold(0f32,f32::max),"nonzero":data.iter().filter(|x|**x!=0.).count(),"scalar":if data.len()==1{Some(data[0])}else{None}}));
            Ok(())
        })?;
        for &x in out {
            output.write_sample(x)?;
        }
        rows.push(serde_json::json!({"block":block,"stages":stats}));
    }
    output.finalize()?;
    std::fs::write(
        a.out.join("trace.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"input":a.input,"startup":startup,"delayed_input":a.delayed_input,"precise_volume":a.precise_volume,"aligned_pitch_history":a.aligned_pitch_history,"seed":1,"rows":rows}),
        )?,
    )?;
    Ok(())
}

fn read_stage(root: &std::path::Path, case: &str, block: usize, stage: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(case).join(format!("{block}-{stage}.f32")))?;
    ensure!(bytes.len() % 4 == 0, "invalid stage bytes");
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(values.iter().all(|v| v.is_finite()), "invalid stage values");
    Ok(values)
}

fn replay(a: &Args, root: &std::path::Path, startup: &Startup) -> Result<()> {
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(a.model.join("model.json"))?)?;
    let rate = metadata["model_sr"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("missing model rate"))? as usize;
    let dims = startup.tg_dims(rate)?;
    let mut generator = rvc_deiteris::Generator::new_with_graph(
        &a.model.join("generator.onnx"),
        &a.model,
        rate,
        true,
    )?;
    std::fs::create_dir_all(&a.out)?;
    let mut rows = Vec::new();
    for (name, features_case, pitch_case) in [
        ("c1", "c1", "c1"),
        ("features-only", "delayed", "c1"),
        ("pitch-only", "c1", "delayed"),
        ("delayed", "delayed", "delayed"),
    ] {
        generator.seed(1)?;
        for block in 0..a.blocks {
            let features: Vec<_> = read_stage(root, features_case, block, "features")?
                .into_iter()
                .map(f16::from_f32)
                .collect();
            let pitch: Vec<_> = read_stage(root, pitch_case, block, "pitch")?
                .into_iter()
                .map(|v| v as i64)
                .collect();
            let pitchf: Vec<_> = read_stage(root, pitch_case, block, "pitchf")?
                .into_iter()
                .map(f16::from_f32)
                .collect();
            let generated = generator.infer(
                &features,
                &pitch,
                &pitchf,
                dims.skip,
                dims.ret,
                dims.formant_length,
            )?;
            if features_case == pitch_case {
                let expected = read_stage(root, features_case, block, "generated")?;
                ensure!(
                    generated == expected,
                    "replay differs from captured {name} block {block}"
                );
            }
            let rms = (generated.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                / generated.len() as f64)
                .sqrt();
            std::fs::write(
                a.out.join(format!("{name}-{block}.f32")),
                generated
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            rows.push(serde_json::json!({"case":name,"block":block,"generated_rms_db":20.*rms.max(1e-15).log10()}));
        }
    }
    std::fs::write(
        a.out.join("replay.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"diagonal_reproduces_captures_exactly":true,"seed":1,"rows":rows}),
        )?,
    )?;
    Ok(())
}
