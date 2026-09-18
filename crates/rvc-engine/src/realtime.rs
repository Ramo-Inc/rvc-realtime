//! `Realtime`: the audio side of the official realtime GUI (`realtime_gui.py` `start_vc` / `start_stream` /
//! `audio_callback`). One full-duplex PortAudio WASAPI stream with `blocksize = block_frame`; each callback
//! converts its block with `Engine::process` and writes the result to every output channel. An optional monitor
//! device gets the converted blocks on its own output stream, as VCClient's `run_with_monitor` does.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::config::{Model, Startup};
use crate::engine::{Engine, EngineOptions, Params};
use crate::error::{Error, Result};
use crate::portaudio::{PaStream, PortAudio, PA_ABORT, PA_CONTINUE};

pub struct Devices {
    /// device name as returned by `list_devices`
    pub input: String,
    pub output: String,
    /// device that also plays the converted voice, at the monitor volume
    pub monitor: Option<String>,
    /// the official GUI's "WASAPI exclusive" option
    pub wasapi_exclusive: bool,
}

pub struct RealtimeOptions {
    pub engine: EngineOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Loading,
    Running,
    Error,
}

pub struct DeviceEntry {
    pub name: String,
    /// the rate the official GUI uses for "device sample rate" (`query_devices(...)["default_samplerate"]`)
    pub default_sample_rate: u32,
}

pub struct DeviceList {
    pub inputs: Vec<DeviceEntry>,
    pub outputs: Vec<DeviceEntry>,
}

/// The WASAPI input and output devices.
pub fn list_devices(runtime_dir: &Path) -> Result<DeviceList> {
    let pa = PortAudio::open(runtime_dir)?;
    let devices = pa.wasapi_devices()?;
    let entry = |d: &crate::portaudio::Device| DeviceEntry { name: d.name.clone(), default_sample_rate: d.default_sample_rate as u32 };
    Ok(DeviceList {
        inputs: devices.iter().filter(|d| d.max_input_channels > 0).map(entry).collect(),
        outputs: devices.iter().filter(|d| d.max_output_channels > 0).map(entry).collect(),
    })
}

struct Shared {
    status: AtomicU8,
    status_text: Mutex<String>,
    pitch: AtomicU32,
    rms_mix: AtomicU32,
    monitor_volume: AtomicU32,
    infer_ms: AtomicU32,
    /// time of the last stream callback, in ms since `epoch`
    epoch: Instant,
    last_callback_ms: AtomicU64,
}

impl Shared {
    fn fail(&self, text: String) {
        if let Ok(mut t) = self.status_text.lock() {
            *t = text;
        }
        self.status.store(Status::Error as u8, Ordering::Relaxed);
    }
}

/// The latest converted block, handed from the conversion callback to the monitor callback. Only the newest
/// block is kept: the monitor plays it once, or silence when none has arrived.
struct Latest {
    block: Mutex<Vec<f32>>,
    fresh: AtomicBool,
}

/// State owned by the stream callback.
struct CallbackState {
    engine: Engine,
    shared: Arc<Shared>,
    channels: usize,
    mono: Vec<f32>,
    monitor: Option<Arc<Latest>>,
}

/// State owned by the monitor stream callback.
struct MonitorState {
    latest: Arc<Latest>,
    shared: Arc<Shared>,
    channels: usize,
}

/// The open streams and everything they need; dropped in field order after the streams are closed.
struct Running {
    pa: PortAudio,
    stream: *mut PaStream,
    state: *mut CallbackState,
    monitor: Option<(*mut PaStream, *mut MonitorState)>,
}

// SAFETY: the streams and callback states are only touched by PortAudio's callback threads while the streams are
// open, and by `Drop` after `Pa_AbortStream` / `Pa_CloseStream` have stopped those threads.
unsafe impl Send for Running {}

impl Drop for Running {
    fn drop(&mut self) {
        self.pa.close(self.stream);
        if let Some((stream, state)) = self.monitor {
            self.pa.close(stream);
            drop(unsafe { Box::from_raw(state) });
        }
        drop(unsafe { Box::from_raw(self.state) });
    }
}

pub struct Realtime {
    shared: Arc<Shared>,
    loader: Option<JoinHandle<Option<Running>>>,
}

impl Realtime {
    /// Returns at once; loads the model and opens the stream on a background thread.
    pub fn start(model_dir: PathBuf, startup: Startup, devices: Devices, opts: RealtimeOptions) -> Realtime {
        let shared = Arc::new(Shared {
            status: AtomicU8::new(Status::Loading as u8),
            status_text: Mutex::new("loading".into()),
            pitch: AtomicU32::new(12f32.to_bits()),
            rms_mix: AtomicU32::new(0.5f32.to_bits()),
            monitor_volume: AtomicU32::new(1f32.to_bits()),
            infer_ms: AtomicU32::new(0f32.to_bits()),
            epoch: Instant::now(),
            last_callback_ms: AtomicU64::new(0),
        });
        let s = shared.clone();
        let loader = std::thread::spawn(move || match load(&model_dir, &startup, &devices, &opts, &s) {
            Ok(running) => {
                if let Ok(mut t) = s.status_text.lock() {
                    *t = "running".into();
                }
                s.last_callback_ms.store(s.epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
                s.status.store(Status::Running as u8, Ordering::Relaxed);
                Some(running)
            }
            Err(e) => {
                s.fail(e.to_string());
                None
            }
        });
        Realtime { shared, loader: Some(loader) }
    }

    pub fn set_pitch(&self, semitones: f32) {
        self.shared.pitch.store(semitones.to_bits(), Ordering::Relaxed);
    }

    pub fn set_rms_mix(&self, value: f32) {
        self.shared.rms_mix.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Volume of the monitor device, 0..1; the output device is not affected.
    pub fn set_monitor_volume(&self, volume: f32) {
        self.shared.monitor_volume.store(volume.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    /// `Error` also when the stream stopped calling back (a device was removed or disabled while running).
    pub fn status(&self) -> Status {
        match self.shared.status.load(Ordering::Relaxed) {
            0 => Status::Loading,
            1 => {
                let s = &self.shared;
                let silent = (s.epoch.elapsed().as_millis() as u64).saturating_sub(s.last_callback_ms.load(Ordering::Relaxed));
                if silent > STREAM_STALL_MS {
                    s.fail(format!("audio: the stream stopped (no callback for {} s)", silent / 1000));
                    return Status::Error;
                }
                Status::Running
            }
            _ => Status::Error,
        }
    }

    pub fn status_text(&self) -> String {
        self.shared.status_text.lock().map(|t| t.clone()).unwrap_or_default()
    }

    /// Conversion time of the last block (the official GUI's inference time).
    pub fn infer_ms(&self) -> f32 {
        f32::from_bits(self.shared.infer_ms.load(Ordering::Relaxed))
    }
}

impl Drop for Realtime {
    fn drop(&mut self) {
        // Waits for a load in progress; dropping `Running` closes the stream, then frees the engine.
        if let Some(handle) = self.loader.take() {
            drop(handle.join());
        }
    }
}

fn load(model_dir: &Path, startup: &Startup, devices: &Devices, opts: &RealtimeOptions, shared: &Arc<Shared>) -> Result<Running> {
    let model = Model::open(model_dir)?;
    let engine = Engine::new(&model, startup, &opts.engine)?;
    let pa = PortAudio::open(&opts.engine.runtime_dir)?;
    let all = pa.wasapi_devices()?;
    let input = all
        .iter()
        .find(|d| d.max_input_channels > 0 && d.name == devices.input)
        .ok_or_else(|| Error::Audio(format!("input device not found: {}", devices.input)))?;
    let output = all
        .iter()
        .find(|d| d.max_output_channels > 0 && d.name == devices.output)
        .ok_or_else(|| Error::Audio(format!("output device not found: {}", devices.output)))?;
    // realtime_gui.py get_device_channels: min(max input channels, max output channels, 2)
    let channels = input.max_input_channels.min(output.max_output_channels).min(2);
    let block = engine.block_frames();
    let sample_rate = startup.sample_rate as f64;

    // The monitor is only for listening: shared mode (auto-convert) whatever the exclusive option says.
    let mut monitor = None;
    let mut latest = None;
    if let Some(name) = &devices.monitor {
        let device = all
            .iter()
            .find(|d| d.max_output_channels > 0 && &d.name == name)
            .ok_or_else(|| Error::Audio(format!("monitor device not found: {name}")))?;
        let l = Arc::new(Latest { block: Mutex::new(vec![0.0; block]), fresh: AtomicBool::new(false) });
        let ch = device.max_output_channels.min(2);
        let state = Box::into_raw(Box::new(MonitorState { latest: l.clone(), shared: shared.clone(), channels: ch as usize }));
        match pa.start(None, device, ch, sample_rate, block as u32, false, monitor_callback, state as *mut c_void) {
            Ok(stream) => monitor = Some((stream, state)),
            Err(e) => {
                drop(unsafe { Box::from_raw(state) });
                return Err(e);
            }
        }
        latest = Some(l);
    }

    let state = Box::into_raw(Box::new(CallbackState { engine, shared: shared.clone(), channels: channels as usize, mono: vec![0.0; block], monitor: latest }));
    let stream = match pa.start(Some(input), output, channels, sample_rate, block as u32, devices.wasapi_exclusive, callback, state as *mut c_void) {
        Ok(stream) => stream,
        Err(e) => {
            if let Some((stream, state)) = monitor {
                pa.close(stream);
                drop(unsafe { Box::from_raw(state) });
            }
            drop(unsafe { Box::from_raw(state) });
            return Err(e);
        }
    };
    Ok(Running { pa, stream, state, monitor })
}

/// Longer than any block (the official maximum is 1000 ms): no callback for this long means the stream ended.
const STREAM_STALL_MS: u64 = 3000;

/// `realtime_gui.py` `audio_callback`: to_mono, convert the block, repeat it to every output channel.
unsafe extern "C" fn callback(input: *const c_void, output: *mut c_void, frames: u32, _time: *const c_void, _flags: u32, user: *mut c_void) -> i32 {
    let st = &mut *(user as *mut CallbackState);
    st.shared.last_callback_ms.store(st.shared.epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
    let frames = frames as usize;
    let ch = st.channels;
    let out = std::slice::from_raw_parts_mut(output as *mut f32, frames * ch);
    if input.is_null() || frames != st.mono.len() {
        out.fill(0.0);
        return PA_CONTINUE;
    }
    let inp = std::slice::from_raw_parts(input as *const f32, frames * ch);
    for (m, frame) in st.mono.iter_mut().zip(inp.chunks_exact(ch)) {
        *m = frame.iter().sum::<f32>() / ch as f32;
    }
    let params = Params {
        pitch: f32::from_bits(st.shared.pitch.load(Ordering::Relaxed)),
        rms_mix: f32::from_bits(st.shared.rms_mix.load(Ordering::Relaxed)),
    };
    let t = Instant::now();
    match st.engine.process(&st.mono, &params) {
        Ok(converted) => {
            for (frame, v) in out.chunks_exact_mut(ch).zip(converted) {
                frame.fill(*v);
            }
            if let Some(latest) = &st.monitor {
                if let Ok(mut block) = latest.block.try_lock() {
                    block.copy_from_slice(converted);
                    latest.fresh.store(true, Ordering::Relaxed);
                }
            }
            st.shared.infer_ms.store((t.elapsed().as_secs_f64() as f32 * 1000.0).to_bits(), Ordering::Relaxed);
            PA_CONTINUE
        }
        Err(e) => {
            out.fill(0.0);
            st.shared.fail(e.to_string());
            PA_ABORT
        }
    }
}

/// Monitor stream callback: the latest converted block at the monitor volume on every channel, else silence.
unsafe extern "C" fn monitor_callback(_input: *const c_void, output: *mut c_void, frames: u32, _time: *const c_void, _flags: u32, user: *mut c_void) -> i32 {
    let st = &*(user as *const MonitorState);
    let out = std::slice::from_raw_parts_mut(output as *mut f32, frames as usize * st.channels);
    if let Ok(block) = st.latest.block.try_lock() {
        if block.len() == frames as usize && st.latest.fresh.swap(false, Ordering::Relaxed) {
            let volume = f32::from_bits(st.shared.monitor_volume.load(Ordering::Relaxed));
            for (frame, v) in out.chunks_exact_mut(st.channels).zip(block.iter()) {
                frame.fill(v * volume);
            }
            return PA_CONTINUE;
        }
    }
    out.fill(0.0);
    PA_CONTINUE
}
