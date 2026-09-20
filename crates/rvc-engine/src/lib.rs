//! Realtime RVC voice conversion: `Engine` converts one block at a time (official `RVCStreamEngine`),
//! `Realtime` runs it on an input/output device pair the way the official realtime GUI does.

#[doc(hidden)]
pub mod config;
#[doc(hidden)]
pub mod dsp;
mod engine;
#[doc(hidden)]
pub mod noise;
mod error;
mod infer;
mod monitor;
#[doc(hidden)]
pub mod onset;
mod portaudio;
mod realtime;
mod processor;
#[cfg(any(feature = "deiteris", test))]
mod owner;
pub mod voice_model;

pub use config::{F0Method, F0Window, Model, Startup, Variant};
pub use engine::{Engine, EngineOptions, Params};
pub use error::Error;
pub use realtime::{list_devices, DeviceEntry, DeviceList, Devices, Realtime, RealtimeOptions, Status};
pub use processor::{Conversion, Processor};
#[cfg(feature = "deiteris")]
pub use rvc_deiteris::startup::Startup as DeiterisStartup;
