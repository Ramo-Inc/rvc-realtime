//! Voice model conversion without Python: an RVC v2 F0 model (`.pth` as the official training saves it, or
//! `.safetensors` with the config in its metadata) becomes a model directory the engine opens.
//!
//! Only the generator depends on the voice model. `tools/export_generator_template.py` writes one generator
//! graph per official training config with every weight in external data (`generator.weights`); conversion
//! checks the model against a template and writes its weights at the template's offsets, loaded like the
//! official `rtrvc.get_synthesizer` (fp32, `remove_weight_norm`, then half).
//!
//! Assets directory: `contentvec.onnx`, `rmvpe.onnx`, `fcpe.onnx`, `templates/<name>/{generator.onnx, template.json}`.

use std::path::Path;

use half::f16;
use rvc_model::{Source, Tensor};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// Why a voice model cannot be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {
    /// not RVC v2 (the engine takes 768-dim features)
    Version(String),
    /// trained without F0
    NoF0,
    /// generator settings, embedder or vocoder differ from the official training configs
    Config,
    /// a weight the generator needs is missing or has another shape
    Weights(String),
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unsupported::Version(v) => write!(f, "version {v} (only v2 is supported)"),
            Unsupported::NoF0 => write!(f, "model without F0"),
            Unsupported::Config => write!(f, "generator settings differ from the official 32k / 40k / 48k configs"),
            Unsupported::Weights(k) => write!(f, "weight {k} is missing or has another shape"),
        }
    }
}

/// Checks the voice model and returns the name of the template it fits ("32k", "40k", "48k").
pub fn check(src: &Path, assets_dir: &Path) -> Result<String> {
    inspect(src, assets_dir).map(|(name, _, _)| name)
}

/// Separate, content-addressed cache for the Deiteris graph contract. Template
/// changes invalidate it even when the source voice is unchanged.
pub fn prepare_deiteris(src: &Path, assets_dir: &Path, cache_root: &Path) -> Result<std::path::PathBuf> {
    prepare_deiteris_with_progress(src, assets_dir, cache_root, || {})
}

/// Calls `converting` only on a cache miss, immediately before weight conversion.
pub fn prepare_deiteris_with_progress(
    src: &Path, assets_dir: &Path, cache_root: &Path, converting: impl FnOnce(),
) -> Result<std::path::PathBuf> {
    let name = check(src, assets_dir)?;
    let template_dir = assets_dir.join("templates").join(name);
    let read = |path: &Path| std::fs::read(path).map_err(|e| Error::ModelFiles {
        path: path.into(), reason: e.to_string(),
    });
    let description = read(&template_dir.join("template.json"))?;
    let template: Template = serde_json::from_slice(&description).map_err(|e| Error::Runtime(e.to_string()))?;
    if template.model["generator_kind"] != "deiteris-onnx-v1" {
        return Err(Error::Runtime("not a Deiteris ONNX template".into()));
    }
    let graph = read(&template_dir.join("generator.onnx"))?;
    let mut digest = Sha256::new();
    digest.update(b"deiteris-onnx-v1\0");
    digest.update(read(src)?);
    digest.update(&description);
    digest.update(&graph);
    let key = format!("{:x}", digest.finalize());
    let output = cache_root.join("deiteris-onnx-v1").join(key);
    // Keep the lock file: unlinking it would allow two independent lock owners.
    // OS locks release on process exit, including crashes during conversion.
    let lock_path = output.with_extension("lock");
    let io = |e: std::io::Error| Error::ModelFiles { path: lock_path.clone(), reason: e.to_string() };
    std::fs::create_dir_all(output.parent().unwrap()).map_err(&io)?;
    let lock = std::fs::OpenOptions::new().read(true).write(true).create(true)
        .truncate(false).open(&lock_path).map_err(&io)?;
    lock.lock().map_err(&io)?;
    let valid = read(&output.join("generator.onnx")).is_ok_and(|v| v == graph)
        && std::fs::metadata(output.join("generator.weights")).is_ok_and(|v| v.len() == template.weights_len as u64)
        && read(&output.join("model.json")).ok()
            .and_then(|v| serde_json::from_slice::<Value>(&v).ok())
            .is_some_and(|v| v["generator_kind"] == "deiteris-onnx-v1" && v["model_sr"] == template.model["model_sr"]);
    if !valid {
        converting();
        convert(src, assets_dir, &output)?;
    }
    Ok(output)
}

/// Writes the converted model to `out_dir` (generator.onnx, generator.weights, model.json).
pub fn convert(src: &Path, assets_dir: &Path, out_dir: &Path) -> Result<()> {
    let (name, template, source) = inspect(src, assets_dir)?;
    let mut weights = vec![0u8; template.weights_len];
    for w in &template.weights {
        let data = match &w.weight_norm {
            Some([g, v]) => remove_weight_norm(&source.tensors[g], &source.tensors[v]),
            None => {
                let t = &source.tensors[&w.key];
                let n: usize = w.shape.iter().product();
                t.to_f32()[..n].iter().map(|&x| f16::from_f32(x)).collect()
            }
        };
        let bytes = &mut weights[w.offset..w.offset + data.len() * 2];
        for (b, x) in bytes.chunks_exact_mut(2).zip(&data) {
            b.copy_from_slice(&x.to_le_bytes());
        }
    }

    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |e: std::io::Error| Error::ModelFiles { path, reason: e.to_string() }
    };
    let tmp = out_dir.with_extension("tmp");
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp).map_err(io(&tmp))?;
    }
    std::fs::create_dir_all(&tmp).map_err(io(&tmp))?;
    let tdir = assets_dir.join("templates").join(&name);
    std::fs::copy(tdir.join("generator.onnx"), tmp.join("generator.onnx")).map_err(io(&tdir))?;
    std::fs::write(tmp.join("generator.weights"), &weights).map_err(io(&tmp))?;
    let mut model = template.model.clone();
    let shared = |file: &str| -> Result<Value> {
        let p = std::path::absolute(assets_dir.join(file)).map_err(io(assets_dir))?;
        Ok(Value::String(p.to_string_lossy().into_owned()))
    };
    model["files"] = serde_json::json!({
        "contentvec": shared("contentvec.onnx")?,
        "rmvpe": shared("rmvpe.onnx")?,
        "fcpe": shared("fcpe.onnx")?,
        "generator": "generator.onnx",
    });
    std::fs::write(tmp.join("model.json"), serde_json::to_string_pretty(&model).unwrap()).map_err(io(&tmp))?;
    if out_dir.exists() {
        std::fs::remove_dir_all(out_dir).map_err(io(out_dir))?;
    }
    std::fs::rename(&tmp, out_dir).map_err(io(out_dir))
}

#[derive(Deserialize)]
struct Template {
    config: Vec<Value>,
    model: Value,
    weights_len: usize,
    weights: Vec<TemplateWeight>,
}

#[derive(Deserialize)]
struct TemplateWeight {
    key: String,
    offset: usize,
    shape: Vec<usize>,
    weight_norm: Option<[String; 2]>,
    rows: Option<usize>,
}

/// Index of the speaker count in the official config list (`savee`); it differs per model.
const SPK_EMBED_DIM: usize = 15;

fn inspect(src: &Path, assets_dir: &Path) -> Result<(String, Template, Source)> {
    let source = Source::read(src).map_err(|reason| Error::ModelFiles { path: src.to_path_buf(), reason })?;
    let unsupported = |u| Err(Error::Unsupported(u));
    // rtrvc.get_synthesizer: cpt.get("version", "v1"), cpt.get("f0", 1)
    let version = source.version.clone().unwrap_or_else(|| "v1".into());
    if version != "v2" {
        return unsupported(Unsupported::Version(version));
    }
    if !source.f0 {
        return unsupported(Unsupported::NoF0);
    }
    if source.embedder.as_deref().is_some_and(|e| e != "contentvec") || source.vocoder.as_deref().is_some_and(|v| v != "HiFi-GAN") {
        return unsupported(Unsupported::Config);
    }
    let tdir = assets_dir.join("templates");
    let entries = std::fs::read_dir(&tdir).map_err(|e| Error::ModelFiles { path: tdir.clone(), reason: e.to_string() })?;
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for name in names {
        let path = tdir.join(&name).join("template.json");
        let text = std::fs::read_to_string(&path).map_err(|e| Error::ModelFiles { path: path.clone(), reason: e.to_string() })?;
        let template: Template = serde_json::from_str(&text).map_err(|e| Error::ModelFiles { path: path.clone(), reason: e.to_string() })?;
        if !same_config(&template.config, &source.config) {
            continue;
        }
        for w in &template.weights {
            let fits = |key: &str, shape: &[usize]| {
                source.tensors.get(key).is_some_and(|t| match w.rows {
                    Some(_) => t.shape.len() == shape.len() && t.shape[0] >= 1 && t.shape[1..] == shape[1..],
                    None => t.shape == shape,
                })
            };
            let ok = match &w.weight_norm {
                // weight_g has the output channels and ones elsewhere (torch weight_norm, dim 0)
                Some([g, v]) => {
                    let mut gshape = vec![1; w.shape.len()];
                    gshape[0] = w.shape[0];
                    fits(v, &w.shape) && fits(g, &gshape)
                }
                None => fits(&w.key, &w.shape),
            };
            if !ok {
                return unsupported(Unsupported::Weights(w.key.clone()));
            }
        }
        return Ok((name, template, source));
    }
    unsupported(Unsupported::Config)
}

/// Equal except the speaker count; numbers compare by value (0 == 0.0).
fn same_config(template: &[Value], model: &[Value]) -> bool {
    fn eq(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
            (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(a, b)| eq(a, b)),
            _ => a == b,
        }
    }
    template.len() == model.len() && template.iter().zip(model).enumerate().all(|(i, (a, b))| i == SPK_EMBED_DIM || eq(a, b))
}

/// `torch._weight_norm(v, g, 0)` in fp32 (`v * (g / ||v||)`, the norm over every dim but 0), then half.
fn remove_weight_norm(g: &Tensor, v: &Tensor) -> Vec<f16> {
    let (g, v) = (g.to_f32(), v.to_f32());
    let rows = g.len();
    let per = v.len() / rows;
    let mut out = Vec::with_capacity(v.len());
    for (i, row) in v.chunks_exact(per).enumerate() {
        let norm = row.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt() as f32;
        let scale = g[i] / norm;
        out.extend(row.iter().map(|&x| f16::from_f32(x * scale)));
    }
    out
}
