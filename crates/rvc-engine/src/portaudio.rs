//! Minimal bindings to the PortAudio DLL that sounddevice ships (`libportaudio64bit.dll`, PortAudio v19.7.0),
//! the library the official realtime GUI and VCClient open their streams with. Declarations follow
//! `portaudio.h` and `pa_win_wasapi.h` of v19.7.0.

use std::ffi::{c_char, c_void, CStr};
use std::path::Path;

use crate::error::{Error, Result};

pub(crate) const DLL: &str = "libportaudio64bit.dll";

pub(crate) type PaStream = c_void;
pub(crate) type Callback = unsafe extern "C" fn(*const c_void, *mut c_void, u32, *const c_void, u32, *mut c_void) -> i32;

pub(crate) const PA_FLOAT32: u32 = 0x0000_0001;
pub(crate) const PA_CONTINUE: i32 = 0;
pub(crate) const PA_ABORT: i32 = 2;
const PA_WASAPI: i32 = 13;
const WASAPI_EXCLUSIVE: u32 = 1 << 0;
const WASAPI_AUTO_CONVERT: u32 = 1 << 6;

#[repr(C)]
pub(crate) struct DeviceInfo {
    struct_version: i32,
    name: *const c_char,
    host_api: i32,
    max_input_channels: i32,
    max_output_channels: i32,
    default_low_input_latency: f64,
    default_low_output_latency: f64,
    default_high_input_latency: f64,
    default_high_output_latency: f64,
    default_sample_rate: f64,
}

#[repr(C)]
pub(crate) struct StreamParameters {
    device: i32,
    channel_count: i32,
    sample_format: u32,
    suggested_latency: f64,
    host_api_specific_stream_info: *mut c_void,
}

#[repr(C)]
pub(crate) struct WasapiStreamInfo {
    size: u32,
    host_api_type: i32,
    version: u32,
    flags: u32,
    channel_mask: u32,
    host_processor_output: *mut c_void,
    host_processor_input: *mut c_void,
    thread_priority: i32,
    stream_category: i32,
    stream_option: i32,
}

impl WasapiStreamInfo {
    /// sounddevice `WasapiSettings(exclusive=exclusive, auto_convert=not exclusive)` as VCClient opens it.
    pub(crate) fn new(exclusive: bool) -> Self {
        Self {
            size: std::mem::size_of::<Self>() as u32,
            host_api_type: PA_WASAPI,
            version: 1,
            flags: if exclusive { WASAPI_EXCLUSIVE } else { WASAPI_AUTO_CONVERT },
            channel_mask: 0,
            host_processor_output: std::ptr::null_mut(),
            host_processor_input: std::ptr::null_mut(),
            thread_priority: 0,
            stream_category: 0,
            stream_option: 0,
        }
    }
}

/// A WASAPI device: PortAudio index, name, channel counts and default low latencies.
pub(crate) struct Device {
    pub index: i32,
    pub name: String,
    pub max_input_channels: i32,
    pub max_output_channels: i32,
    pub low_input_latency: f64,
    pub low_output_latency: f64,
    pub default_sample_rate: f64,
}

/// The loaded DLL, initialised with `Pa_Initialize`; `Pa_Terminate` on drop.
pub(crate) struct PortAudio {
    _lib: libloading::Library,
    terminate: unsafe extern "C" fn() -> i32,
    error_text: unsafe extern "C" fn(i32) -> *const c_char,
    host_api_index: unsafe extern "C" fn(i32) -> i32,
    device_count: unsafe extern "C" fn() -> i32,
    device_info: unsafe extern "C" fn(i32) -> *const DeviceInfo,
    open_stream: unsafe extern "C" fn(*mut *mut PaStream, *const StreamParameters, *const StreamParameters, f64, u32, u32, Callback, *mut c_void) -> i32,
    start_stream: unsafe extern "C" fn(*mut PaStream) -> i32,
    abort_stream: unsafe extern "C" fn(*mut PaStream) -> i32,
    close_stream: unsafe extern "C" fn(*mut PaStream) -> i32,
}

fn audio<E: std::fmt::Display>(e: E) -> Error {
    Error::Audio(e.to_string())
}

impl PortAudio {
    pub(crate) fn open(runtime_dir: &Path) -> Result<Self> {
        unsafe {
            let lib = libloading::Library::new(runtime_dir.join(DLL)).map_err(|e| Error::Audio(format!("load {}: {e}", runtime_dir.join(DLL).display())))?;
            let initialize = *lib.get::<unsafe extern "C" fn() -> i32>(b"Pa_Initialize").map_err(audio)?;
            let pa = Self {
                terminate: *lib.get(b"Pa_Terminate").map_err(audio)?,
                error_text: *lib.get(b"Pa_GetErrorText").map_err(audio)?,
                host_api_index: *lib.get(b"Pa_HostApiTypeIdToHostApiIndex").map_err(audio)?,
                device_count: *lib.get(b"Pa_GetDeviceCount").map_err(audio)?,
                device_info: *lib.get(b"Pa_GetDeviceInfo").map_err(audio)?,
                open_stream: *lib.get(b"Pa_OpenStream").map_err(audio)?,
                start_stream: *lib.get(b"Pa_StartStream").map_err(audio)?,
                abort_stream: *lib.get(b"Pa_AbortStream").map_err(audio)?,
                close_stream: *lib.get(b"Pa_CloseStream").map_err(audio)?,
                _lib: lib,
            };
            let rc = initialize();
            if rc != 0 {
                return Err(Error::Audio(format!("Pa_Initialize: {}", pa.text(rc))));
            }
            Ok(pa)
        }
    }

    fn text(&self, rc: i32) -> String {
        unsafe { CStr::from_ptr((self.error_text)(rc)).to_string_lossy().into_owned() }
    }

    fn check(&self, what: &str, rc: i32) -> Result<()> {
        if rc < 0 {
            return Err(Error::Audio(format!("{what}: {}", self.text(rc))));
        }
        Ok(())
    }

    /// Devices of the WASAPI host API.
    pub(crate) fn wasapi_devices(&self) -> Result<Vec<Device>> {
        unsafe {
            let wasapi = (self.host_api_index)(PA_WASAPI);
            self.check("WASAPI host API", wasapi)?;
            let count = (self.device_count)();
            self.check("Pa_GetDeviceCount", count)?;
            let mut out = Vec::new();
            for index in 0..count {
                let info = (self.device_info)(index);
                if info.is_null() || (*info).host_api != wasapi {
                    continue;
                }
                let i = &*info;
                out.push(Device {
                    index,
                    name: CStr::from_ptr(i.name).to_string_lossy().into_owned(),
                    max_input_channels: i.max_input_channels,
                    max_output_channels: i.max_output_channels,
                    low_input_latency: i.default_low_input_latency,
                    low_output_latency: i.default_low_output_latency,
                    default_sample_rate: i.default_sample_rate,
                });
            }
            Ok(out)
        }
    }

    /// Opens and starts one float32 stream: full duplex (`sd.Stream(device=(input, output), blocksize=..., latency='low')`)
    /// or, without an input, output only (`sd.OutputStream`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start(
        &self,
        input: Option<&Device>,
        output: &Device,
        channels: i32,
        sample_rate: f64,
        frames_per_buffer: u32,
        exclusive: bool,
        callback: Callback,
        user_data: *mut c_void,
    ) -> Result<*mut PaStream> {
        let mut in_info = WasapiStreamInfo::new(exclusive);
        let mut out_info = WasapiStreamInfo::new(exclusive);
        let in_params = input.map(|input| StreamParameters {
            device: input.index,
            channel_count: channels,
            sample_format: PA_FLOAT32,
            suggested_latency: input.low_input_latency,
            host_api_specific_stream_info: &mut in_info as *mut _ as *mut c_void,
        });
        let out_params = StreamParameters {
            device: output.index,
            channel_count: channels,
            sample_format: PA_FLOAT32,
            suggested_latency: output.low_output_latency,
            host_api_specific_stream_info: &mut out_info as *mut _ as *mut c_void,
        };
        let mut stream: *mut PaStream = std::ptr::null_mut();
        unsafe {
            self.check("Pa_OpenStream", (self.open_stream)(&mut stream, in_params.as_ref().map_or(std::ptr::null(), |p| p as *const _), &out_params, sample_rate, frames_per_buffer, 0, callback, user_data))?;
            if let Err(e) = self.check("Pa_StartStream", (self.start_stream)(stream)) {
                (self.close_stream)(stream);
                return Err(e);
            }
        }
        Ok(stream)
    }

    /// `Pa_AbortStream` then `Pa_CloseStream`; after this the callback is no longer called.
    pub(crate) fn close(&self, stream: *mut PaStream) {
        unsafe {
            (self.abort_stream)(stream);
            (self.close_stream)(stream);
        }
    }
}

impl Drop for PortAudio {
    fn drop(&mut self) {
        unsafe {
            (self.terminate)();
        }
    }
}
