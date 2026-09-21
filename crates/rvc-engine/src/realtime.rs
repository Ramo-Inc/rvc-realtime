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

use crate::config::Startup;
use crate::processor::{Conversion, Processor};
use crate::engine::{EngineOptions, Params};
use crate::error::{Error, Result};
use crate::monitor::MonitorBuffer;
use crate::onset::OnsetLatency;
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
    threshold_db: AtomicU32,
    drop_silent_context: AtomicBool,
    skip_silence: AtomicBool,
    monitor_volume: AtomicU32,
    infer_ms: AtomicU32,
    /// Packed elapsed time + block distance from the same detected utterance.
    onset_measurement: AtomicU64,
    /// time of the last stream callback, in ms since `epoch`
    epoch: Instant,
    last_callback_ms: AtomicU64,
    /// blocks whose conversion took longer than the block itself (the stream runs dry)
    late_blocks: AtomicU64,
    /// monitor callbacks with no new block to play (its device asked faster than blocks arrive)
    monitor_starved: AtomicU64,
    /// the monitor queue, so its counters can be read from outside the callbacks
    monitor_counts: Mutex<Option<Arc<MonitorBuffer>>>,
    blocks: AtomicU64,
    monitor_blocks: AtomicU64,
    /// callbacks PortAudio flagged: input overflow / output underflow (the stream lost samples)
    input_overflow: AtomicU64,
    output_underflow: AtomicU64,
}

impl Shared {
    fn fail(&self, text: String) {
        // Hold the message lock until Error is published, so startup completion
        // cannot interleave between the error message and its status.
        let mut message = self.status_text.lock().ok();
        if let Some(t) = message.as_mut() {
            **t = text;
        }
        self.status.store(Status::Error as u8, Ordering::Relaxed);
    }
}

fn publish_running(status: &AtomicU8, text: &Mutex<String>) {
    let mut message = text.lock().ok();
    if status.compare_exchange(Status::Loading as u8, Status::Running as u8, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        if let Some(text) = message.as_mut() {
            **text = "running".into();
        }
    }
}

/// State owned by the stream callback.
struct CallbackState {
    engine: Processor,
    onset: OnsetLatency,
    /// how long one block lasts, in seconds: conversion has to stay inside it
    block_seconds: f64,
    shared: Arc<Shared>,
    channels: usize,
    mono: Vec<f32>,
    monitor: Option<Arc<MonitorBuffer>>,
}

/// State owned by the monitor stream callback.
struct MonitorState {
    buffer: Arc<MonitorBuffer>,
    shared: Arc<Shared>,
    channels: usize,
    /// one value per frame, before it is written to every channel
    mono: Vec<f32>,
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
        Self::start_configured(Conversion::Legacy { model_dir, startup }, devices, opts,
            Params { pitch: 12.0, rms_mix: 0.5, threshold_db: -60.0, drop_silent_context: false, skip_silence: false }, 1.0)
    }

    /// Initial live values are installed before the first callback, including during model loading.
    pub fn start_configured(conversion: Conversion, devices: Devices, opts: RealtimeOptions, initial: Params, monitor_volume: f32) -> Realtime {
        let shared = Arc::new(Shared {
            status: AtomicU8::new(Status::Loading as u8),
            status_text: Mutex::new("loading".into()),
            pitch: AtomicU32::new(initial.pitch.to_bits()),
            rms_mix: AtomicU32::new(initial.rms_mix.to_bits()),
            threshold_db: AtomicU32::new(initial.threshold_db.to_bits()),
            drop_silent_context: AtomicBool::new(initial.drop_silent_context),
            skip_silence: AtomicBool::new(initial.skip_silence),
            monitor_volume: AtomicU32::new(monitor_volume.clamp(0.0, 1.0).to_bits()),
            infer_ms: AtomicU32::new(0f32.to_bits()),
            onset_measurement: AtomicU64::new(f32::NAN.to_bits() as u64),
            epoch: Instant::now(),
            last_callback_ms: AtomicU64::new(0),
            late_blocks: AtomicU64::new(0),
            monitor_starved: AtomicU64::new(0),
            monitor_counts: Mutex::new(None),
            blocks: AtomicU64::new(0),
            monitor_blocks: AtomicU64::new(0),
            input_overflow: AtomicU64::new(0),
            output_underflow: AtomicU64::new(0),
        });
        let s = shared.clone();
        let loader = std::thread::spawn(move || match load(conversion, &devices, &opts, &s) {
            Ok(running) => {
                s.last_callback_ms.store(s.epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
                // A callback may already have failed during Pa_StartStream.
                publish_running(&s.status, &s.status_text);
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

    /// Silence gate in dB; -60 and below is off, as in the official GUI.
    pub fn set_threshold_db(&self, value: f32) {
        self.shared.threshold_db.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Keep only speech in the engine's context (silence never enters it).
    pub fn set_drop_silent_context(&self, on: bool) {
        self.shared.drop_silent_context.store(on, Ordering::Relaxed);
    }

    /// Idle the models while the input stays below the threshold.
    pub fn set_skip_silence(&self, on: bool) {
        self.shared.skip_silence.store(on, Ordering::Relaxed);
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

    /// Last measured utterance onset: input callback delivery to generated output
    /// readiness, excluding hardware playback. None until a quiet-to-sound pair.
    pub fn onset_latency(&self) -> Option<(f32, u32)> {
        let packed = self.shared.onset_measurement.load(Ordering::Relaxed);
        let ms = f32::from_bits(packed as u32);
        ms.is_finite().then_some((ms, (packed >> 32) as u32))
    }

    /// Closes streams and joins the inference owner; counters remain readable afterward.
    pub fn stop(&mut self) {
        if let Some(handle) = self.loader.take() { drop(handle.join()); }
    }

    /// Callbacks PortAudio flagged: (input overflow, output underflow).
    pub fn glitches(&self) -> (u64, u64) {
        let s = &self.shared;
        (s.input_overflow.load(Ordering::Relaxed), s.output_underflow.load(Ordering::Relaxed))
    }

    /// What the streams could not deliver: (blocks, blocks that took longer than a block, monitor
    /// callbacks, monitor callbacks that ran dry, samples the monitor had to invent, samples it dropped).
    pub fn stats(&self) -> (u64, u64, u64, u64, u64, u64) {
        let s = &self.shared;
        let (missing, dropped) = s.monitor_counts.lock().ok().and_then(|b| b.as_ref().map(|b| b.counts())).unwrap_or((0, 0));
        (
            s.blocks.load(Ordering::Relaxed),
            s.late_blocks.load(Ordering::Relaxed),
            s.monitor_blocks.load(Ordering::Relaxed),
            s.monitor_starved.load(Ordering::Relaxed),
            missing,
            dropped,
        )
    }
}

impl Drop for Realtime {
    fn drop(&mut self) {
        // Waits for a load in progress; dropping `Running` closes the stream, then frees the engine.
        self.stop();
    }
}

fn load(conversion: Conversion, devices: &Devices, opts: &RealtimeOptions, shared: &Arc<Shared>) -> Result<Running> {
    let sample_rate = conversion.sample_rate() as f64;
    let engine = conversion.load(&opts.engine)?;
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

    // The monitor is only for listening: shared mode (auto-convert) whatever the exclusive option says.
    let mut monitor = None;
    let mut latest = None;
    if let Some(name) = &devices.monitor {
        let device = all
            .iter()
            .find(|d| d.max_output_channels > 0 && &d.name == name)
            .ok_or_else(|| Error::Audio(format!("monitor device not found: {name}")))?;
        let l = Arc::new(MonitorBuffer::new(block));
        let ch = device.max_output_channels.min(2);
        let state = Box::into_raw(Box::new(MonitorState { buffer: l.clone(), shared: shared.clone(), channels: ch as usize, mono: vec![0.0; block] }));
        match pa.start(None, device, ch, sample_rate, block as u32, false, monitor_callback, state as *mut c_void) {
            Ok(stream) => monitor = Some((stream, state)),
            Err(e) => {
                drop(unsafe { Box::from_raw(state) });
                return Err(e);
            }
        }
        if let Ok(mut slot) = shared.monitor_counts.lock() {
            *slot = Some(l.clone());
        }
        latest = Some(l);
    }

    let block_seconds = block as f64 / sample_rate;
    let state = Box::into_raw(Box::new(CallbackState { engine, onset: OnsetLatency::new(sample_rate as u32), block_seconds, shared: shared.clone(), channels: channels as usize, mono: vec![0.0; block], monitor: latest }));
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
unsafe extern "C" fn callback(input: *const c_void, output: *mut c_void, frames: u32, _time: *const c_void, flags: u32, user: *mut c_void) -> i32 {
    let st = &mut *(user as *mut CallbackState);
    let input_seen = st.shared.epoch.elapsed();
    if flags != 0 {
        st.onset.reset();
        st.shared.onset_measurement.store(f32::NAN.to_bits() as u64, Ordering::Relaxed);
    }
    // paInputOverflow: the device had more input than the callback took; paOutputUnderflow: the device
    // played something we did not deliver in time. Either one is an audible discontinuity.
    if flags & 0x2 != 0 {
        st.shared.input_overflow.fetch_add(1, Ordering::Relaxed);
    }
    if flags & 0x4 != 0 {
        st.shared.output_underflow.fetch_add(1, Ordering::Relaxed);
    }
    st.shared.last_callback_ms.store(st.shared.epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
    let frames = frames as usize;
    let ch = st.channels;
    let out = std::slice::from_raw_parts_mut(output as *mut f32, frames * ch);
    if input.is_null() || frames != st.mono.len() {
        st.onset.reset();
        st.shared.onset_measurement.store(f32::NAN.to_bits() as u64, Ordering::Relaxed);
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
        threshold_db: f32::from_bits(st.shared.threshold_db.load(Ordering::Relaxed)),
        drop_silent_context: st.shared.drop_silent_context.load(Ordering::Relaxed),
        skip_silence: st.shared.skip_silence.load(Ordering::Relaxed),
    };
    let t = Instant::now();
    match st.engine.process(&st.mono, &params) {
        Ok(converted) => {
            for (frame, v) in out.chunks_exact_mut(ch).zip(converted) {
                frame.fill(*v);
            }
            if let Some(buffer) = &st.monitor {
                buffer.push(converted);
            }
            // The primary output buffer has been filled. This is our handoff
            // boundary, not the time the hardware plays the buffer.
            let output_written = st.shared.epoch.elapsed();
            if flags == 0 {
                if let Some(measurement) = st.onset.observe(&st.mono, converted, input_seen, output_written) {
                    st.shared.onset_measurement.store(measurement.packed(), Ordering::Relaxed);
                }
            }
            let took = t.elapsed().as_secs_f64();
            st.shared.infer_ms.store((took as f32 * 1000.0).to_bits(), Ordering::Relaxed);
            st.shared.blocks.fetch_add(1, Ordering::Relaxed);
            if took > st.block_seconds {
                st.shared.late_blocks.fetch_add(1, Ordering::Relaxed);
            }
            PA_CONTINUE
        }
        Err(e) => {
            out.fill(0.0);
            st.shared.fail(e.to_string());
            PA_ABORT
        }
    }
}

/// Monitor stream callback: whatever the queue holds, at the monitor volume, on every channel. The queue
/// absorbs the two devices running on their own clocks; what it cannot deliver it fades out.
unsafe extern "C" fn monitor_callback(_input: *const c_void, output: *mut c_void, frames: u32, _time: *const c_void, _flags: u32, user: *mut c_void) -> i32 {
    let st = &mut *(user as *mut MonitorState);
    st.shared.monitor_blocks.fetch_add(1, Ordering::Relaxed);
    let frames = frames as usize;
    let out = std::slice::from_raw_parts_mut(output as *mut f32, frames * st.channels);
    st.mono.resize(frames, 0.0);
    let volume = f32::from_bits(st.shared.monitor_volume.load(Ordering::Relaxed));
    if !st.buffer.read(&mut st.mono, volume) {
        st.shared.monitor_starved.fetch_add(1, Ordering::Relaxed);
    }
    for (frame, v) in out.chunks_exact_mut(st.channels).zip(st.mono.iter()) {
        frame.fill(*v);
    }
    PA_CONTINUE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_completion_preserves_callback_error() {
        // Pa_StartStream may deliver a failed first callback before returning.
        let status = AtomicU8::new(Status::Error as u8);
        let text = Mutex::new("inference: first callback failed".to_string());
        publish_running(&status, &text);
        assert_eq!(status.load(Ordering::Relaxed), Status::Error as u8);
        assert_eq!(*text.lock().unwrap(), "inference: first callback failed");
    }

    #[test]
    fn successful_startup_publishes_running() {
        let status = AtomicU8::new(Status::Loading as u8);
        let text = Mutex::new("loading".to_string());
        publish_running(&status, &text);
        assert_eq!(status.load(Ordering::Relaxed), Status::Running as u8);
        assert_eq!(*text.lock().unwrap(), "running");
    }
}
