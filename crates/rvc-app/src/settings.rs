//! The saved configuration (`%APPDATA%\rvc-app\settings.json`), written whenever a value changes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    #[default]
    Legacy,
    Deiteris,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NativeSettings {
    pub chunk: usize,
    pub extra_ms: f64,
    pub crossfade_ms: f64,
    pub threshold_db: f32,
    pub cuda_graph: bool,
}
impl Default for NativeSettings {
    fn default() -> Self {
        Self { chunk: 19, extra_ms: 500.0, crossfade_ms: 100.0, threshold_db: -90.0, cuda_graph: true }
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    #[serde(default)]
    pub backend: Backend,
    pub native: NativeSettings,
    pub input: String,
    pub output: String,
    pub monitor: Option<String>,
    pub exclusive: bool,
    pub voice_model: Option<PathBuf>,
    /// "fcpe" | "rmvpe"
    pub f0: String,
    pub block_ms: f64,
    pub crossfade_ms: f64,
    pub extra_ms: f64,
    pub rms_mix: f32,
    /// what counts as silence, in dB; -60 is off
    pub threshold_db: f32,
    /// do not convert while the input is silent (and keep silence out of the engine's context)
    pub skip_silence: bool,
    /// percent
    pub monitor_volume: u32,
    /// pitch and formant per voice model file
    pub voices: BTreeMap<PathBuf, Voice>,
    /// configurations saved under a name; the first is always `DEFAULT_PRESET`
    pub presets: Vec<Preset>,
    /// the preset shown in the options window's dropdown
    #[serde(deserialize_with = "null_as_empty")]
    pub active_preset: String,
    /// the bundled voice model; the default preset always uses it
    #[serde(skip)]
    pub default_voice: PathBuf,
    /// which defaults the default preset was last brought up to (see `DEFAULTS_VERSION`)
    pub defaults_version: u32,
}

/// 1: FCPE, crossfade 80 ms, rms_mix 0.5 (up to 0.1.5). 2: RMVPE, crossfade 100 ms, rms_mix 1.0 (0.1.6,
/// with the Deiteris VCClient F0 handling in `main.rs`, docs/plans/quality-variants-poc/design.md).
const DEFAULTS_VERSION: u32 = 2;

/// Reads `null` as an empty string (earlier settings files stored no active preset as `null`).
fn null_as_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

/// The preset that always exists and cannot be renamed or deleted.
pub const DEFAULT_PRESET: &str = "デフォルト";

/// Everything the options window shows, saved under a name.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    #[serde(default)]
    pub backend: Backend,
    #[serde(default)]
    pub native: NativeSettings,
    pub name: String,
    pub input: String,
    pub output: String,
    pub monitor: Option<String>,
    pub exclusive: bool,
    pub voice_model: Option<PathBuf>,
    pub pitch: f32,
    pub formant: f64,
    pub rms_mix: f32,
    pub threshold_db: f32,
    pub skip_silence: bool,
    pub f0: String,
    pub monitor_volume: u32,
    pub block_ms: f64,
    pub crossfade_ms: f64,
    pub extra_ms: f64,
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Voice {
    pub pitch: f32,
    pub formant: f64,
}

impl Default for Settings {
    /// block 60 ms and context 1000 ms (lowlat), RMVPE, the whole 100 ms crossfaded, rms_mix off
    fn default() -> Self {
        Self {
            backend: Backend::Deiteris,
            native: NativeSettings::default(),
            input: String::new(),
            output: String::new(),
            monitor: None,
            exclusive: false,
            voice_model: None,
            f0: "rmvpe".into(),
            block_ms: 60.0,
            crossfade_ms: 100.0,
            extra_ms: 1000.0,
            rms_mix: 1.0,
            threshold_db: -60.0,
            skip_silence: true,
            monitor_volume: 50,
            voices: BTreeMap::new(),
            presets: Vec::new(),
            active_preset: DEFAULT_PRESET.into(),
            default_voice: PathBuf::new(),
            // a file without the field predates the versions
            defaults_version: 0,
        }
    }
}

fn app_dir(var: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(var).unwrap_or_else(|| ".".into())).join("rvc-app")
}

pub fn file() -> PathBuf {
    app_dir("APPDATA").join("settings.json")
}

/// Where converted voice models are kept.
pub fn models_dir() -> PathBuf {
    app_dir("LOCALAPPDATA").join("models")
}

impl Settings {
    /// Reads the saved settings. The default preset's voice model is always `default_voice` (the bundled model).
    pub fn load(default_voice: &Path) -> Self {
        let mut s: Settings = std::fs::read_to_string(file()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        s.default_voice = default_voice.to_path_buf();
        if !s.presets.iter().any(|p| p.name == s.active_preset) {
            s.active_preset = DEFAULT_PRESET.into();
        }
        if s.active_preset == DEFAULT_PRESET {
            s.voice_model = Some(s.default_voice.clone());
        }
        // the default preset starts as the options in use when it first appears
        if !s.presets.iter().any(|p| p.name == DEFAULT_PRESET) {
            let default = s.snapshot(DEFAULT_PRESET);
            s.presets.insert(0, default);
        }
        let voice = Some(s.default_voice.clone());
        let fresh = Settings::default();
        if let Some(p) = s.presets.iter_mut().find(|p| p.name == DEFAULT_PRESET) {
            p.voice_model = voice;
            // new defaults reach the default preset only where it still holds the old ones untouched
            if s.defaults_version < DEFAULTS_VERSION && p.f0 == "fcpe" && p.crossfade_ms == 80.0 && p.rms_mix == 0.5 {
                p.f0 = fresh.f0.clone();
                p.crossfade_ms = fresh.crossfade_ms;
                p.rms_mix = fresh.rms_mix;
            }
        }
        s.defaults_version = DEFAULTS_VERSION;
        // open on exactly what the default preset saved
        if s.active_preset == DEFAULT_PRESET {
            if let Some(p) = s.presets.iter().find(|p| p.name == DEFAULT_PRESET).cloned() {
                s.load_preset(&p);
            }
        }
        s
    }

    pub fn save(&self) {
        let path = file();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(path, serde_json::to_string_pretty(self).unwrap());
    }

    /// Whether `name` can be given to a new or renamed preset.
    pub fn name_is_free(&self, name: &str) -> bool {
        name != DEFAULT_PRESET && !self.presets.iter().any(|p| p.name == name)
    }

    /// Renames the active preset (not the default one).
    pub fn rename_active_preset(&mut self, name: &str) {
        if self.active_preset == DEFAULT_PRESET {
            return;
        }
        if let Some(p) = self.presets.iter_mut().find(|p| p.name == self.active_preset) {
            p.name = name.to_string();
        }
        self.active_preset = name.to_string();
    }

    /// Deletes the active preset (not the default one) and loads the default.
    pub fn delete_active_preset(&mut self) {
        if self.active_preset == DEFAULT_PRESET {
            return;
        }
        self.presets.retain(|p| p.name != self.active_preset);
        self.active_preset = DEFAULT_PRESET.into();
        if let Some(default) = self.presets.iter().find(|p| p.name == DEFAULT_PRESET).cloned() {
            self.load_preset(&default);
        }
    }

    /// Whether the options differ from what the active preset saved.
    pub fn is_modified(&mut self) -> bool {
        let now = self.snapshot(&self.active_preset.clone());
        self.presets.iter().find(|p| p.name == now.name) != Some(&now)
    }

    /// Saves the current options under `name`, replacing a preset with the same name.
    pub fn save_preset(&mut self, name: &str) {
        let preset = self.snapshot(name);
        match self.presets.iter_mut().find(|p| p.name == name) {
            Some(p) => *p = preset,
            None => self.presets.push(preset),
        }
    }

    /// The current options as a preset named `name`.
    fn snapshot(&mut self, name: &str) -> Preset {
        let voice = self.voice().copied().unwrap_or(Voice { pitch: 0.0, formant: 0.0 });
        let voice_model = if name == DEFAULT_PRESET { Some(self.default_voice.clone()) } else { self.voice_model.clone() };
        Preset {
            backend: self.backend,
            native: self.native.clone(),
            name: name.to_string(),
            voice_model,
            input: self.input.clone(),
            output: self.output.clone(),
            monitor: self.monitor.clone(),
            exclusive: self.exclusive,
            pitch: voice.pitch,
            formant: voice.formant,
            rms_mix: self.rms_mix,
            threshold_db: self.threshold_db,
            skip_silence: self.skip_silence,
            f0: self.f0.clone(),
            monitor_volume: self.monitor_volume,
            block_ms: self.block_ms,
            crossfade_ms: self.crossfade_ms,
            extra_ms: self.extra_ms,
        }
    }

    /// Makes every option the preset's value.
    pub fn load_preset(&mut self, p: &Preset) {
        self.backend = p.backend;
        self.native = p.native.clone();
        self.input = p.input.clone();
        self.output = p.output.clone();
        self.monitor = p.monitor.clone();
        self.exclusive = p.exclusive;
        self.voice_model = p.voice_model.clone();
        if let Some(path) = &p.voice_model {
            self.voices.insert(path.clone(), Voice { pitch: p.pitch, formant: p.formant });
        }
        self.rms_mix = p.rms_mix;
        self.threshold_db = p.threshold_db;
        self.skip_silence = p.skip_silence;
        self.f0 = p.f0.clone();
        self.monitor_volume = p.monitor_volume;
        self.block_ms = p.block_ms;
        self.crossfade_ms = p.crossfade_ms;
        self.extra_ms = p.extra_ms;
    }

    /// Pitch and formant of the selected voice model; a model used for the first time starts from VCClient's
    /// `params.json` next to it when that file describes it, else 0 / 0.
    pub fn voice(&mut self) -> Option<&mut Voice> {
        let path = self.voice_model.clone()?;
        Some(self.voices.entry(path.clone()).or_insert_with(|| vcclient_defaults(&path).unwrap_or(Voice { pitch: 0.0, formant: 0.0 })))
    }
}

fn vcclient_defaults(model: &Path) -> Option<Voice> {
    let text = std::fs::read_to_string(model.parent()?.join("params.json")).ok()?;
    let p: serde_json::Value = serde_json::from_str(&text).ok()?;
    if p["modelFile"].as_str()? != model.file_name()?.to_str()? {
        return None;
    }
    Some(Voice { pitch: p["defaultTune"].as_f64()? as f32, formant: p["defaultFormantShift"].as_f64()? })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_configuration_stays_legacy_and_native_preset_roundtrips() {
        let old: Settings = serde_json::from_str(r#"{"block_ms":50.0,"threshold_db":-42.0}"#).unwrap();
        assert_eq!(old.backend, Backend::Legacy);
        assert_eq!(old.block_ms, 50.0);
        assert_eq!(old.threshold_db, -42.0);
        let mut fresh = Settings::default();
        assert_eq!(fresh.backend, Backend::Deiteris);
        fresh.native.chunk = 17;
        fresh.native.threshold_db = -83.0;
        fresh.save_preset("native");
        let encoded = serde_json::to_string(&fresh).unwrap();
        let mut restored: Settings = serde_json::from_str(&encoded).unwrap();
        restored.backend = Backend::Legacy;
        restored.native.chunk = 99;
        let p = restored.presets[0].clone();
        restored.load_preset(&p);
        assert_eq!(restored.backend, Backend::Deiteris);
        assert_eq!(restored.native.chunk, 17);
        assert_eq!(restored.native.threshold_db, -83.0);
        let mut old_preset = serde_json::to_value(p).unwrap();
        old_preset.as_object_mut().unwrap().remove("backend");
        old_preset.as_object_mut().unwrap().remove("native");
        let old_preset: Preset = serde_json::from_value(old_preset).unwrap();
        assert_eq!(old_preset.backend, Backend::Legacy);
    }
}
