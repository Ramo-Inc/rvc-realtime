//! The saved configuration (`%APPDATA%\rvc-app\settings.json`), written whenever a value changes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
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
}

/// Reads `null` as an empty string (earlier settings files stored no active preset as `null`).
fn null_as_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

/// The preset that always exists and cannot be renamed or deleted.
pub const DEFAULT_PRESET: &str = "デフォルト";

/// Everything the options window shows, saved under a name.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    pub input: String,
    pub output: String,
    pub monitor: Option<String>,
    pub exclusive: bool,
    pub voice_model: Option<PathBuf>,
    pub pitch: f32,
    pub formant: f64,
    pub rms_mix: f32,
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
    /// lowlat-fcpe
    fn default() -> Self {
        Self {
            input: String::new(),
            output: String::new(),
            monitor: None,
            exclusive: false,
            voice_model: None,
            f0: "fcpe".into(),
            block_ms: 60.0,
            crossfade_ms: 80.0,
            extra_ms: 1000.0,
            rms_mix: 0.5,
            monitor_volume: 50,
            voices: BTreeMap::new(),
            presets: Vec::new(),
            active_preset: DEFAULT_PRESET.into(),
            default_voice: PathBuf::new(),
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
        if let Some(p) = s.presets.iter_mut().find(|p| p.name == DEFAULT_PRESET) {
            p.voice_model = voice;
        }
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
            name: name.to_string(),
            voice_model,
            input: self.input.clone(),
            output: self.output.clone(),
            monitor: self.monitor.clone(),
            exclusive: self.exclusive,
            pitch: voice.pitch,
            formant: voice.formant,
            rms_mix: self.rms_mix,
            f0: self.f0.clone(),
            monitor_volume: self.monitor_volume,
            block_ms: self.block_ms,
            crossfade_ms: self.crossfade_ms,
            extra_ms: self.extra_ms,
        }
    }

    /// Makes every option the preset's value.
    pub fn load_preset(&mut self, p: &Preset) {
        self.input = p.input.clone();
        self.output = p.output.clone();
        self.monitor = p.monitor.clone();
        self.exclusive = p.exclusive;
        self.voice_model = p.voice_model.clone();
        if let Some(path) = &p.voice_model {
            self.voices.insert(path.clone(), Voice { pitch: p.pitch, formant: p.formant });
        }
        self.rms_mix = p.rms_mix;
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
