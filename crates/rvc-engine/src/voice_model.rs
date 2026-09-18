//! Voice model conversion without Python: an RVC v2 F0 model (`.pth` as the official training saves it, or
//! `.safetensors` with the config in its metadata) becomes a model directory the engine opens.
//!
//! Only the generator depends on the voice model. `tools/export_generator_template.py` writes one generator
//! graph per official training config with every weight in external data (`generator.weights`); conversion
//! checks the model against a template and writes its weights at the template's offsets, loaded like the
//! official `rtrvc.get_synthesizer` (fp32, `remove_weight_norm`, then half).
//!
//! Assets directory: `contentvec.onnx`, `rmvpe.onnx`, `fcpe.onnx`, `templates/<name>/{generator.onnx, template.json}`.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use half::f16;
use serde::Deserialize;
use serde_json::Value;

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
    let source = Source::read(src)?;
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

#[derive(Clone, Copy, PartialEq)]
enum Dtype {
    F16,
    F32,
}

struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    /// little-endian element data, exactly `shape.product()` elements
    data: Vec<u8>,
}

impl Tensor {
    fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            Dtype::F16 => self.data.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
            Dtype::F32 => self.data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        }
    }
}

struct Source {
    version: Option<String>,
    f0: bool,
    config: Vec<Value>,
    embedder: Option<String>,
    vocoder: Option<String>,
    tensors: HashMap<String, Tensor>,
}

impl Source {
    fn read(path: &Path) -> Result<Self> {
        let fail = |reason: String| Error::ModelFiles { path: path.to_path_buf(), reason };
        let bytes = std::fs::read(path).map_err(|e| fail(e.to_string()))?;
        if bytes.starts_with(b"PK") {
            read_pth(&bytes).map_err(fail)
        } else {
            read_safetensors(&bytes).map_err(fail)
        }
    }
}

fn read_safetensors(bytes: &[u8]) -> std::result::Result<Source, String> {
    let (_, meta) = safetensors::SafeTensors::read_metadata(bytes).map_err(|e| e.to_string())?;
    let meta = meta.metadata().clone().unwrap_or_default();
    let st = safetensors::SafeTensors::deserialize(bytes).map_err(|e| e.to_string())?;
    let mut tensors = HashMap::new();
    for (name, view) in st.tensors() {
        let dtype = match view.dtype() {
            safetensors::Dtype::F16 => Dtype::F16,
            safetensors::Dtype::F32 => Dtype::F32,
            other => return Err(format!("tensor {name} has dtype {other:?}")),
        };
        tensors.insert(name, Tensor { dtype, shape: view.shape().to_vec(), data: view.data().to_vec() });
    }
    let config = match meta.get("config") {
        Some(c) => serde_json::from_str(c).map_err(|e| format!("config: {e}"))?,
        None => return Err("no config in the metadata".into()),
    };
    Ok(Source {
        version: meta.get("version").cloned(),
        f0: meta.get("f0").is_none_or(|f| f == "True" || f == "true" || f == "1"),
        config,
        embedder: meta.get("embedder_model").cloned(),
        vocoder: meta.get("vocoder").cloned(),
        tensors,
    })
}

fn read_pth(bytes: &[u8]) -> std::result::Result<Source, String> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let pkl = zip.file_names().find(|n| n.ends_with("data.pkl")).ok_or("no data.pkl in the archive")?.to_string();
    let prefix = pkl.trim_end_matches("data.pkl").to_string();
    let mut data = Vec::new();
    zip.by_name(&pkl).map_err(|e| e.to_string())?.read_to_end(&mut data).map_err(|e| e.to_string())?;
    let root = pickle::load(&data)?;
    let get = |k: &str| root.get(k);

    let mut storages: HashMap<String, Vec<u8>> = HashMap::new();
    let mut tensors = HashMap::new();
    let weight = match get("weight") {
        Some(pickle::V::Dict(items)) => items,
        _ => return Err("no weight dict".into()),
    };
    for (k, v) in weight {
        let (pickle::V::Str(name), pickle::V::Tensor { dtype, key, offset, shape, stride }) = (k, v) else {
            continue;
        };
        let (dtype, size) = match dtype.as_str() {
            "HalfStorage" => (Dtype::F16, 2),
            "FloatStorage" => (Dtype::F32, 4),
            other => return Err(format!("tensor {name} has storage {other}")),
        };
        let mut contiguous = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            contiguous[i] = contiguous[i + 1] * shape[i + 1];
        }
        // torch ignores the stride of size-1 dims
        if shape.iter().zip(stride.iter().zip(&contiguous)).any(|(&n, (s, c))| n > 1 && s != c) {
            return Err(format!("tensor {name} is not contiguous"));
        }
        if !storages.contains_key(key) {
            let mut buf = Vec::new();
            zip.by_name(&format!("{prefix}data/{key}")).map_err(|e| e.to_string())?.read_to_end(&mut buf).map_err(|e| e.to_string())?;
            storages.insert(key.clone(), buf);
        }
        let storage = &storages[key];
        let (start, len) = (offset * size, shape.iter().product::<usize>() * size);
        let part = storage.get(start..start + len).ok_or(format!("tensor {name} is outside its storage"))?;
        tensors.insert(name.clone(), Tensor { dtype, shape: shape.clone(), data: part.to_vec() });
    }
    let config = match get("config") {
        Some(c) => c.to_json(),
        None => return Err("no config".into()),
    };
    let Value::Array(config) = config else { return Err("config is not a list".into()) };
    let text = |k: &str| match get(k) {
        Some(pickle::V::Str(s)) => Some(s.clone()),
        _ => None,
    };
    let f0 = match get("f0") {
        None => true,
        Some(pickle::V::Int(i)) => *i != 0,
        Some(pickle::V::Bool(b)) => *b,
        Some(pickle::V::Str(s)) => s == "True" || s == "1",
        Some(_) => false,
    };
    Ok(Source { version: text("version"), f0, config, embedder: text("embedder_model"), vocoder: text("vocoder"), tensors })
}

/// The subset of the pickle protocol `torch.save` writes for an RVC checkpoint.
mod pickle {
    use serde_json::Value;

    #[derive(Clone, Debug)]
    pub enum V {
        None,
        Bool(bool),
        Int(i64),
        Float(f64),
        Str(String),
        Bytes,
        List(Vec<V>),
        Tuple(Vec<V>),
        Dict(Vec<(V, V)>),
        Global(String, String),
        Storage { dtype: String, key: String },
        Tensor { dtype: String, key: String, offset: usize, shape: Vec<usize>, stride: Vec<usize> },
        Mark,
        Other,
    }

    impl V {
        pub fn get(&self, key: &str) -> Option<&V> {
            match self {
                V::Dict(items) => items.iter().find(|(k, _)| matches!(k, V::Str(s) if s == key)).map(|(_, v)| v),
                _ => None,
            }
        }

        pub fn to_json(&self) -> Value {
            match self {
                V::Bool(b) => Value::Bool(*b),
                V::Int(i) => Value::from(*i),
                V::Float(f) => serde_json::Number::from_f64(*f).map(Value::Number).unwrap_or(Value::Null),
                V::Str(s) => Value::String(s.clone()),
                V::List(xs) | V::Tuple(xs) => Value::Array(xs.iter().map(V::to_json).collect()),
                _ => Value::Null,
            }
        }

        fn usizes(&self) -> Option<Vec<usize>> {
            match self {
                V::Tuple(xs) | V::List(xs) => xs.iter().map(|x| if let V::Int(i) = x { usize::try_from(*i).ok() } else { None }).collect(),
                _ => None,
            }
        }
    }

    pub fn load(data: &[u8]) -> Result<V, String> {
        let mut stack: Vec<V> = Vec::new();
        let mut memo: std::collections::HashMap<u32, V> = std::collections::HashMap::new();
        let mut pos = 0;
        let take = |pos: &mut usize, n: usize| -> Result<&[u8], String> {
            let s = data.get(*pos..*pos + n).ok_or("truncated pickle")?;
            *pos += n;
            Ok(s)
        };
        let u32le = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        fn pop(stack: &mut Vec<V>) -> Result<V, String> {
            stack.pop().ok_or_else(|| "pickle stack underflow".to_string())
        }
        fn pop_mark(stack: &mut Vec<V>) -> Result<Vec<V>, String> {
            let at = stack.iter().rposition(|v| matches!(v, V::Mark)).ok_or("pickle mark missing")?;
            let items = stack.split_off(at + 1);
            stack.pop();
            Ok(items)
        }
        fn text(b: &[u8]) -> Result<String, String> {
            String::from_utf8(b.to_vec()).map_err(|e| e.to_string())
        }
        loop {
            let op = take(&mut pos, 1)?[0];
            match op {
                0x80 => pos += 1,         // PROTO
                0x95 => pos += 8,         // FRAME
                b'.' => return pop(&mut stack), // STOP
                b'}' => stack.push(V::Dict(Vec::new())),
                b']' => stack.push(V::List(Vec::new())),
                b')' => stack.push(V::Tuple(Vec::new())),
                b'(' => stack.push(V::Mark),
                b'N' => stack.push(V::None),
                0x88 => stack.push(V::Bool(true)),
                0x89 => stack.push(V::Bool(false)),
                b'J' => stack.push(V::Int(i32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as i64)),
                b'K' => stack.push(V::Int(take(&mut pos, 1)?[0] as i64)),
                b'M' => stack.push(V::Int(u16::from_le_bytes(take(&mut pos, 2)?.try_into().unwrap()) as i64)),
                0x8a => {
                    // LONG1: little-endian two's complement
                    let n = take(&mut pos, 1)?[0] as usize;
                    let b = take(&mut pos, n)?;
                    let mut v: i64 = if b.last().is_some_and(|x| x & 0x80 != 0) { -1 } else { 0 };
                    for &x in b.iter().rev() {
                        v = (v << 8) | x as i64;
                    }
                    stack.push(V::Int(v));
                }
                b'G' => stack.push(V::Float(f64::from_be_bytes(take(&mut pos, 8)?.try_into().unwrap()))),
                b'X' => {
                    let n = u32le(take(&mut pos, 4)?) as usize;
                    stack.push(V::Str(text(take(&mut pos, n)?)?));
                }
                0x8c => {
                    let n = take(&mut pos, 1)?[0] as usize;
                    stack.push(V::Str(text(take(&mut pos, n)?)?));
                }
                0x8d => {
                    let n = u64::from_le_bytes(take(&mut pos, 8)?.try_into().unwrap()) as usize;
                    stack.push(V::Str(text(take(&mut pos, n)?)?));
                }
                b'C' => {
                    let n = take(&mut pos, 1)?[0] as usize;
                    take(&mut pos, n)?;
                    stack.push(V::Bytes);
                }
                b'B' => {
                    let n = u32le(take(&mut pos, 4)?) as usize;
                    take(&mut pos, n)?;
                    stack.push(V::Bytes);
                }
                b'q' => {
                    let i = take(&mut pos, 1)?[0] as u32;
                    memo.insert(i, stack.last().cloned().ok_or("memo of empty stack")?);
                }
                b'r' => {
                    let i = u32le(take(&mut pos, 4)?);
                    memo.insert(i, stack.last().cloned().ok_or("memo of empty stack")?);
                }
                0x94 => {
                    let i = memo.len() as u32;
                    memo.insert(i, stack.last().cloned().ok_or("memo of empty stack")?);
                }
                b'h' => {
                    let i = take(&mut pos, 1)?[0] as u32;
                    stack.push(memo.get(&i).cloned().ok_or("unknown memo")?);
                }
                b'j' => {
                    let i = u32le(take(&mut pos, 4)?);
                    stack.push(memo.get(&i).cloned().ok_or("unknown memo")?);
                }
                b'c' => {
                    let line = |pos: &mut usize| -> Result<String, String> {
                        let end = data[*pos..].iter().position(|&b| b == b'\n').ok_or("truncated GLOBAL")?;
                        let s = text(&data[*pos..*pos + end])?;
                        *pos += end + 1;
                        Ok(s)
                    };
                    let module = line(&mut pos)?;
                    let name = line(&mut pos)?;
                    stack.push(V::Global(module, name));
                }
                0x93 => {
                    let name = pop(&mut stack)?;
                    let module = pop(&mut stack)?;
                    match (module, name) {
                        (V::Str(m), V::Str(n)) => stack.push(V::Global(m, n)),
                        _ => return Err("bad STACK_GLOBAL".into()),
                    }
                }
                b't' => {
                    let items = pop_mark(&mut stack)?;
                    stack.push(V::Tuple(items));
                }
                0x85..=0x87 => {
                    let n = (op - 0x84) as usize;
                    if stack.len() < n {
                        return Err("pickle stack underflow".into());
                    }
                    let items = stack.split_off(stack.len() - n);
                    stack.push(V::Tuple(items));
                }
                b'Q' => {
                    // BINPERSID: torch storage ('storage', storage type, key, location, numel)
                    let pid = pop(&mut stack)?;
                    let v = match pid {
                        V::Tuple(t) if t.len() >= 3 => match (&t[1], &t[2]) {
                            (V::Global(_, dtype), V::Str(key)) => V::Storage { dtype: dtype.clone(), key: key.clone() },
                            _ => V::Other,
                        },
                        _ => V::Other,
                    };
                    stack.push(v);
                }
                b'R' | 0x81 => {
                    let args = pop(&mut stack)?;
                    let func = pop(&mut stack)?;
                    let v = match (&func, &args) {
                        (V::Global(m, n), V::Tuple(a)) if m == "torch._utils" && n == "_rebuild_tensor_v2" && a.len() >= 4 => {
                            match (&a[0], &a[1], a[2].usizes(), a[3].usizes()) {
                                (V::Storage { dtype, key }, V::Int(offset), Some(shape), Some(stride)) => {
                                    V::Tensor { dtype: dtype.clone(), key: key.clone(), offset: *offset as usize, shape, stride }
                                }
                                _ => V::Other,
                            }
                        }
                        (V::Global(m, n), _) if m == "collections" && n == "OrderedDict" => V::Dict(Vec::new()),
                        _ => V::Other,
                    };
                    stack.push(v);
                }
                b'b' => {
                    pop(&mut stack)?; // BUILD state (object attributes) is not needed
                }
                b's' => {
                    let v = pop(&mut stack)?;
                    let k = pop(&mut stack)?;
                    if let Some(V::Dict(items)) = stack.last_mut() {
                        items.push((k, v));
                    }
                }
                b'u' => {
                    let items = pop_mark(&mut stack)?;
                    if let Some(V::Dict(d)) = stack.last_mut() {
                        let mut it = items.into_iter();
                        while let (Some(k), Some(v)) = (it.next(), it.next()) {
                            d.push((k, v));
                        }
                    }
                }
                b'a' => {
                    let v = pop(&mut stack)?;
                    if let Some(V::List(xs)) = stack.last_mut() {
                        xs.push(v);
                    }
                }
                b'e' => {
                    let items = pop_mark(&mut stack)?;
                    if let Some(V::List(xs)) = stack.last_mut() {
                        xs.extend(items);
                    }
                }
                other => return Err(format!("unsupported pickle opcode 0x{other:02x}")),
            }
        }
    }
}
