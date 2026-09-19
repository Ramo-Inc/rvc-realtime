//! Same-input/shared-noise generator acceptance, without audio devices or Torch.
use clap::Parser;
use half::f16;
use serde_json::json;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    voice: PathBuf,
    #[arg(long)]
    assets: PathBuf,
    #[arg(long)]
    runtime: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    noise: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    uncaptured: bool,
    /// Each case has independent startup dimensions (synthetic matrix fixtures).
    #[arg(long)]
    matrix: bool,
}
fn npy(path: &Path, dtype: &str, width: usize) -> Result<Vec<u8>> {
    let data = fs::read(path)?;
    if data.len() < 10 || &data[..8] != b"\x93NUMPY\x01\x00" {
        return Err("unsupported NPY".into());
    }
    let start = 10 + u16::from_le_bytes(data[8..10].try_into()?) as usize;
    let header = std::str::from_utf8(data.get(10..start).ok_or("truncated header")?)?;
    if !header.contains(&format!("'descr': '{dtype}'"))
        || !header.contains("'fortran_order': False")
        || (data.len() - start) % width != 0
    {
        return Err(format!(
            "invalid fixture dtype/order: {} expected {dtype}",
            path.display()
        )
        .into());
    }
    Ok(data[start..].to_vec())
}
fn half(path: &Path) -> Result<Vec<f16>> {
    Ok(npy(path, "<f2", 2)?
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
        .collect())
}
fn ints(path: &Path) -> Result<Vec<i64>> {
    Ok(npy(path, "<i8", 8)?
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn scalar(dir: &Path, key: &str) -> Result<usize> {
    let path = dir.join(format!("{key}.npy"));
    let values = match npy(&path, "<i4", 4) {
        Ok(data) => data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64)
            .collect(),
        Err(_) => ints(&path)?,
    };
    if values.len() != 1 {
        return Err("expected scalar".into());
    }
    Ok(usize::try_from(values[0])?)
}
fn noise(file: &mut fs::File, shape: &[usize]) -> Result<Vec<f16>> {
    let mut word = [0u8; 4];
    file.read_exact(&mut word)?;
    if u32::from_le_bytes(word) as usize != shape.len() {
        return Err("noise rank mismatch".into());
    }
    for &size in shape {
        file.read_exact(&mut word)?;
        if u32::from_le_bytes(word) as usize != size {
            return Err("noise shape mismatch".into());
        }
    }
    let mut values = vec![0u8; shape.iter().product::<usize>() * 4];
    file.read_exact(&mut values)?;
    Ok(values
        .chunks_exact(4)
        .map(|b| f16::from_f32(f32::from_le_bytes(b.try_into().unwrap())))
        .collect())
}
fn main() -> Result<()> {
    let a = Args::parse();
    if a.out.exists() {
        return Err("report must be new".into());
    }
    let model = rvc_engine::voice_model::prepare_deiteris(
        &a.voice,
        &a.assets,
        &a.out.with_extension("models"),
    )?;
    let info: serde_json::Value = serde_json::from_slice(&fs::read(model.join("model.json"))?)?;
    let rate = info["model_sr"].as_u64().ok_or("missing rate")? as usize;
    let mut paths = vec![a.runtime.canonicalize()?];
    paths.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    std::env::set_var("PATH", std::env::join_paths(paths)?);
    let create = || {
        rvc_deiteris::Generator::new_with_graph(
            &model.join("generator.onnx"),
            &model,
            rate,
            !a.uncaptured,
        )
    };
    let mut generator = create().map_err(|e| e.to_string())?;
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(a.reference.join("manifest.json"))?)?;
    if receipt["complete"] != true {
        return Err("incomplete reference".into());
    }
    let count = receipt["blocks"].as_u64().ok_or("missing block count")? as usize;
    let mut file = fs::File::open(a.noise)?;
    let mut rows = Vec::new();
    let mut total_signal = 0.0;
    let mut total_error = 0.0;
    for i in 0..count {
        if a.matrix && i > 0 {
            drop(generator);
            generator = create().map_err(|e| e.to_string())?;
        }
        let dir = a.reference.join(format!("{i:06}"));
        let features = half(&dir.join("features.npy"))?;
        let pitch = ints(&dir.join("pitch.npy"))?;
        let pitchf = half(&dir.join("pitchf.npy"))?;
        let skip = scalar(&dir, "skip_head")?;
        let ret = scalar(&dir, "return_length")?;
        let formant = scalar(&dir, "formant_length")?;
        let rnd = noise(&mut file, &[1, 192, pitch.len() - skip.saturating_sub(24)])?;
        let sine = noise(&mut file, &[1, ret * (rate / 100), 1])?;
        let actual = generator
            .infer_with_noise(&features, &pitch, &pitchf, skip, ret, formant, &rnd, &sine)
            .map_err(|e| e.to_string())?;
        let expected: Vec<f32> = npy(&dir.join("generated.npy"), "<f4", 4)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if expected.len() != actual.len() {
            return Err("output shape mismatch".into());
        }
        let mut signal = 0.0;
        let mut error = 0.0;
        for (&x, &y) in expected.iter().zip(&actual) {
            if !x.is_finite() || !y.is_finite() {
                return Err("nonfinite result".into());
            }
            signal += (x as f64).powi(2);
            error += (x as f64 - y as f64).powi(2);
        }
        let snr = if error == 0.0 {
            None
        } else {
            Some(10.0 * (signal / error).log10())
        };
        let passed = error == 0.0 || snr.is_some_and(|v| v >= 25.0);
        rows.push(json!({"block":i,"snr_db":snr,"passed":passed,"frames":pitch.len(),"skip":skip,"ret":ret,"formant":formant}));
        total_signal += signal;
        total_error += error;
    }
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        return Err("trailing noise records".into());
    }
    let passed = rows.iter().all(|r| r["passed"] == true);
    let report = json!({"complete":true,"passed":passed,"blocks":count,"cuda_graph":!a.uncaptured,
        "scope":"product ONNX generator with original inputs and noise; not whole-pipeline quality",
        "reference":a.reference,"model":model,"snr_db":if total_error==0.0{None}else{Some(10.0*(total_signal/total_error).log10())},"rows":rows});
    fs::write(&a.out, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "passed={passed} blocks={count} snr={} report={}",
        report["snr_db"],
        a.out.display()
    );
    if !passed {
        return Err("generator parity failed".into());
    }
    Ok(())
}
