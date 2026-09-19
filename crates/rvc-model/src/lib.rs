//! Shared RVC checkpoint reader. Interprets data only; never executes pickle code.
use half::f16;
use serde_json::Value;
use std::{collections::HashMap, io::Read, path::Path};

#[derive(Clone, Copy, PartialEq)]
pub enum Dtype {
    F16,
    F32,
}

pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// little-endian element data, exactly `shape.product()` elements
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            Dtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        }
    }
}

pub struct Source {
    pub version: Option<String>,
    pub f0: bool,
    pub config: Vec<Value>,
    pub embedder: Option<String>,
    pub vocoder: Option<String>,
    pub tensors: HashMap<String, Tensor>,
}

impl Source {
    pub fn read(path: &Path) -> Result<Self, String> {
        let fail = |reason: String| format!("{}: {reason}", path.display());
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
        tensors.insert(
            name,
            Tensor {
                dtype,
                shape: view.shape().to_vec(),
                data: view.data().to_vec(),
            },
        );
    }
    let config = match meta.get("config") {
        Some(c) => serde_json::from_str(c).map_err(|e| format!("config: {e}"))?,
        None => return Err("no config in the metadata".into()),
    };
    Ok(Source {
        version: meta.get("version").cloned(),
        f0: meta
            .get("f0")
            .is_none_or(|f| f == "True" || f == "true" || f == "1"),
        config,
        embedder: meta.get("embedder_model").cloned(),
        vocoder: meta.get("vocoder").cloned(),
        tensors,
    })
}

fn read_pth(bytes: &[u8]) -> std::result::Result<Source, String> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let pkl = zip
        .file_names()
        .find(|n| n.ends_with("data.pkl"))
        .ok_or("no data.pkl in the archive")?
        .to_string();
    let prefix = pkl.trim_end_matches("data.pkl").to_string();
    let mut data = Vec::new();
    zip.by_name(&pkl)
        .map_err(|e| e.to_string())?
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    let root = pickle::load(&data)?;
    let get = |k: &str| root.get(k);

    let mut storages: HashMap<String, Vec<u8>> = HashMap::new();
    let mut tensors = HashMap::new();
    let weight = match get("weight") {
        Some(pickle::V::Dict(items)) => items,
        _ => return Err("no weight dict".into()),
    };
    for (k, v) in weight {
        let (
            pickle::V::Str(name),
            pickle::V::Tensor {
                dtype,
                key,
                offset,
                shape,
                stride,
            },
        ) = (k, v)
        else {
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
        if shape
            .iter()
            .zip(stride.iter().zip(&contiguous))
            .any(|(&n, (s, c))| n > 1 && s != c)
        {
            return Err(format!("tensor {name} is not contiguous"));
        }
        if !storages.contains_key(key) {
            let mut buf = Vec::new();
            zip.by_name(&format!("{prefix}data/{key}"))
                .map_err(|e| e.to_string())?
                .read_to_end(&mut buf)
                .map_err(|e| e.to_string())?;
            storages.insert(key.clone(), buf);
        }
        let storage = &storages[key];
        let (start, len) = (offset * size, shape.iter().product::<usize>() * size);
        let part = storage
            .get(start..start + len)
            .ok_or(format!("tensor {name} is outside its storage"))?;
        tensors.insert(
            name.clone(),
            Tensor {
                dtype,
                shape: shape.clone(),
                data: part.to_vec(),
            },
        );
    }
    let config = match get("config") {
        Some(c) => c.to_json(),
        None => return Err("no config".into()),
    };
    let Value::Array(config) = config else {
        return Err("config is not a list".into());
    };
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
    Ok(Source {
        version: text("version"),
        f0,
        config,
        embedder: text("embedder_model"),
        vocoder: text("vocoder"),
        tensors,
    })
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
        Storage {
            dtype: String,
            key: String,
        },
        Tensor {
            dtype: String,
            key: String,
            offset: usize,
            shape: Vec<usize>,
            stride: Vec<usize>,
        },
        Mark,
        Other,
    }

    impl V {
        pub fn get(&self, key: &str) -> Option<&V> {
            match self {
                V::Dict(items) => items
                    .iter()
                    .find(|(k, _)| matches!(k, V::Str(s) if s == key))
                    .map(|(_, v)| v),
                _ => None,
            }
        }

        pub fn to_json(&self) -> Value {
            match self {
                V::Bool(b) => Value::Bool(*b),
                V::Int(i) => Value::from(*i),
                V::Float(f) => serde_json::Number::from_f64(*f)
                    .map(Value::Number)
                    .unwrap_or(Value::Null),
                V::Str(s) => Value::String(s.clone()),
                V::List(xs) | V::Tuple(xs) => Value::Array(xs.iter().map(V::to_json).collect()),
                _ => Value::Null,
            }
        }

        fn usizes(&self) -> Option<Vec<usize>> {
            match self {
                V::Tuple(xs) | V::List(xs) => xs
                    .iter()
                    .map(|x| {
                        if let V::Int(i) = x {
                            usize::try_from(*i).ok()
                        } else {
                            None
                        }
                    })
                    .collect(),
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
            stack
                .pop()
                .ok_or_else(|| "pickle stack underflow".to_string())
        }
        fn pop_mark(stack: &mut Vec<V>) -> Result<Vec<V>, String> {
            let at = stack
                .iter()
                .rposition(|v| matches!(v, V::Mark))
                .ok_or("pickle mark missing")?;
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
                0x80 => pos += 1,               // PROTO
                0x95 => pos += 8,               // FRAME
                b'.' => return pop(&mut stack), // STOP
                b'}' => stack.push(V::Dict(Vec::new())),
                b']' => stack.push(V::List(Vec::new())),
                b')' => stack.push(V::Tuple(Vec::new())),
                b'(' => stack.push(V::Mark),
                b'N' => stack.push(V::None),
                0x88 => stack.push(V::Bool(true)),
                0x89 => stack.push(V::Bool(false)),
                b'J' => stack.push(V::Int(
                    i32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as i64,
                )),
                b'K' => stack.push(V::Int(take(&mut pos, 1)?[0] as i64)),
                b'M' => stack.push(V::Int(
                    u16::from_le_bytes(take(&mut pos, 2)?.try_into().unwrap()) as i64,
                )),
                0x8a => {
                    // LONG1: little-endian two's complement
                    let n = take(&mut pos, 1)?[0] as usize;
                    let b = take(&mut pos, n)?;
                    let mut v: i64 = if b.last().is_some_and(|x| x & 0x80 != 0) {
                        -1
                    } else {
                        0
                    };
                    for &x in b.iter().rev() {
                        v = (v << 8) | x as i64;
                    }
                    stack.push(V::Int(v));
                }
                b'G' => stack.push(V::Float(f64::from_be_bytes(
                    take(&mut pos, 8)?.try_into().unwrap(),
                ))),
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
                        let end = data[*pos..]
                            .iter()
                            .position(|&b| b == b'\n')
                            .ok_or("truncated GLOBAL")?;
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
                            (V::Global(_, dtype), V::Str(key)) => V::Storage {
                                dtype: dtype.clone(),
                                key: key.clone(),
                            },
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
                        (V::Global(m, n), V::Tuple(a))
                            if m == "torch._utils" && n == "_rebuild_tensor_v2" && a.len() >= 4 =>
                        {
                            match (&a[0], &a[1], a[2].usizes(), a[3].usizes()) {
                                (
                                    V::Storage { dtype, key },
                                    V::Int(offset),
                                    Some(shape),
                                    Some(stride),
                                ) => V::Tensor {
                                    dtype: dtype.clone(),
                                    key: key.clone(),
                                    offset: *offset as usize,
                                    shape,
                                    stride,
                                },
                                _ => V::Other,
                            }
                        }
                        (V::Global(m, n), _) if m == "collections" && n == "OrderedDict" => {
                            V::Dict(Vec::new())
                        }
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
