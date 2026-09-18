//! Start / stop on a worker thread: convert the voice model when no converted copy opens, then run `Realtime`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use rvc_engine::{voice_model, Devices, Model, Realtime, RealtimeOptions, Startup, Status};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Idle,
    Converting,
    Loading,
    Running,
}

pub struct Request {
    pub voice_model: PathBuf,
    pub assets_dir: PathBuf,
    pub models_dir: PathBuf,
    pub startup: Startup,
    pub devices: Devices,
    pub options: RealtimeOptions,
    pub pitch: f32,
    pub rms_mix: f32,
    pub threshold_db: f32,
    pub skip_silence: bool,
    pub monitor_volume: f32,
}

struct Inner {
    stage: Stage,
    /// the thread dropping a stopped `Realtime`; a new start waits for it
    dropping: Option<std::thread::JoinHandle<()>>,
    realtime: Option<Realtime>,
    error: Option<String>,
}

pub struct Session {
    inner: Arc<Mutex<Inner>>,
}

impl Session {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(Inner { stage: Stage::Idle, dropping: None, realtime: None, error: None })) }
    }

    pub fn stage(&self) -> Stage {
        self.inner.lock().unwrap().stage
    }

    pub fn error(&self) -> Option<String> {
        self.inner.lock().unwrap().error.clone()
    }

    /// Stops what runs (on the worker, so the window does not wait) and starts with `req`.
    pub fn start(&self, req: Request) {
        let (old, dropping) = {
            let mut i = self.inner.lock().unwrap();
            i.error = None;
            i.stage = Stage::Loading;
            (i.realtime.take(), i.dropping.take())
        };
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            if let Some(handle) = dropping {
                let _ = handle.join();
            }
            drop(old);
            if let Err(e) = run(&inner, req) {
                let mut i = inner.lock().unwrap();
                i.stage = Stage::Idle;
                i.error = Some(e);
            }
        });
    }

    pub fn stop(&self) {
        let mut i = self.inner.lock().unwrap();
        i.stage = Stage::Idle;
        let old = i.realtime.take();
        i.dropping = Some(std::thread::spawn(move || drop(old)));
    }

    /// Loading → Running once the stream runs; an engine error ends the run and is kept for the window.
    pub fn poll(&self) {
        let mut i = self.inner.lock().unwrap();
        let Some(rt) = &i.realtime else { return };
        match rt.status() {
            Status::Running if i.stage == Stage::Loading => i.stage = Stage::Running,
            Status::Error => {
                let what = if i.stage == Stage::Loading { "開始できませんでした" } else { "変換が止まりました" };
                i.error = Some(format!("{what}: {}", rt.status_text()));
                i.stage = Stage::Idle;
                let old = i.realtime.take();
                i.dropping = Some(std::thread::spawn(move || drop(old)));
            }
            _ => {}
        }
    }

    /// Runs `f` on the current `Realtime`, if any.
    pub fn with_realtime<T>(&self, f: impl FnOnce(&Realtime) -> T) -> Option<T> {
        self.inner.lock().unwrap().realtime.as_ref().map(f)
    }
}

fn run(inner: &Arc<Mutex<Inner>>, req: Request) -> Result<(), String> {
    let bytes = std::fs::read(&req.voice_model).map_err(|e| format!("声モデルを読めません: {e}"))?;
    let hash: String = Sha256::digest(&bytes).iter().take(8).map(|b| format!("{b:02x}")).collect();
    drop(bytes);
    let model_dir = req.models_dir.join(hash);
    if Model::open(&model_dir).is_err() || !model_dir.join("generator.weights").is_file() {
        inner.lock().unwrap().stage = Stage::Converting;
        voice_model::convert(&req.voice_model, &req.assets_dir, &model_dir).map_err(|e| format!("声モデルを変換できませんでした: {e}"))?;
        inner.lock().unwrap().stage = Stage::Loading;
    }
    let rt = Realtime::start(model_dir, req.startup, req.devices, req.options);
    rt.set_pitch(req.pitch);
    rt.set_rms_mix(req.rms_mix);
    rt.set_threshold_db(req.threshold_db);
    rt.set_skip_silence(req.skip_silence);
    // the same switch keeps silence out of the engine's context (PoC: the two together)
    rt.set_drop_silent_context(req.skip_silence);
    rt.set_monitor_volume(req.monitor_volume);
    let mut i = inner.lock().unwrap();
    i.stage = Stage::Loading;
    i.realtime = Some(rt);
    Ok(())
}
