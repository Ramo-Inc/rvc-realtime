//! RVC realtime voice changer: a Start button and options, on `rvc-engine`.
//! Next to the executable: `runtime/` (CUDA / cuDNN / ONNX Runtime / PortAudio DLLs) and `assets/`
//! (shared ONNX, generator templates).

#![windows_subsystem = "windows"]

mod session;
mod settings;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui;
use rvc_engine::voice_model::{self, Unsupported};
use rvc_engine::{list_devices, DeviceList, Devices, EngineOptions, Error, F0Method, F0Window, RealtimeOptions, Startup, Variant};
use session::{Request, Session, Stage};
use settings::{Backend, Settings};

/// The voice model bundled with the app (MIT, see `assets/voices/default_v2_40k.LICENSE.txt`), chosen until the user
/// picks one.
const DEFAULT_VOICE: &str = "voices/default_v2_40k.pth";
/// Space between the window edge and the content, in points.
const MARGIN: f32 = 8.0;
/// Fill of the Stop button while converting.
const RUNNING_COLOR: egui::Color32 = egui::Color32::from_rgb(128, 52, 52);
/// Space above and below the line between option groups.
const DIVIDER_SPACE: f32 = 12.0;
/// Width of the device and preset dropdowns.
const DEVICE_COMBO_WIDTH: f32 = 300.0;

/// The F0 handling of the Deiteris VCClient instead of the official 2.3 path: unvoiced frames stay unvoiced,
/// RMVPE voicing threshold 0.05, the crossfaded head re-estimated, the whole crossfade overlapped.
/// Measured against the official path in docs/plans/quality-variants-poc/design.md (`deiteris_all`).
const APP_VARIANT: Variant = Variant { f0_interp: false, rmvpe_threshold: 0.05, f0_window: F0Window::Head, full_crossfade: true };

fn main() -> eframe::Result {
    // each window's size follows its content every frame (main window; options in its own window)
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title(concat!("RVC リアルタイム変換 v", env!("CARGO_PKG_VERSION"), "-alpha")).with_inner_size([300.0, 80.0])
            .with_resizable(false)
            .with_maximize_button(false)
            .with_icon(Arc::new(egui::IconData::default())),
        ..Default::default()
    };
    eframe::run_native("rvc-app", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

enum PresetEditKind {
    New,
    Rename,
}

struct PresetEdit {
    kind: PresetEditKind,
    name: String,
    /// shown under the field when the name cannot be used
    problem: Option<&'static str>,
    /// move the keyboard focus to the field once, when it appears
    focus: bool,
}

enum Check {
    Pending,
    Ok(String),
    Unsupported(String),
}

/// Everything that needs the engine to be started again when it changes.
#[derive(Clone, PartialEq)]
struct StartKey {
    backend: Backend,
    native_chunk: usize,
    native_extra: f64,
    native_crossfade: f64,
    native_graph: bool,
    input: String,
    output: String,
    monitor: Option<String>,
    exclusive: bool,
    voice_model: Option<PathBuf>,
    f0: String,
    block_ms: f64,
    crossfade_ms: f64,
    extra_ms: f64,
    formant: f64,
}

struct App {
    settings: Settings,
    runtime_dir: PathBuf,
    assets_dir: PathBuf,
    devices: Result<DeviceList, String>,
    check: Arc<Mutex<(Option<PathBuf>, Check)>>,
    session: Session,
    /// configuration of the running (or starting) engine
    started: Option<StartKey>,
    options_open: bool,
    /// options content width of the last frame; the dividers between groups span it
    options_width: f32,
    /// the name field under the preset list, while creating or renaming
    preset_edit: Option<PresetEdit>,
    /// where the options window opens: next to the main window, taken when it is opened
    options_pos: egui::Pos2,
    /// the title-bar icon has been removed from the main / options window
    main_icon_hidden: bool,
    options_icon_hidden: bool,
    /// content width of the last frame; messages wrap at it so a long error does not widen the window
    content_width: f32,
    ctx: egui::Context,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        japanese_font(&cc.egui_ctx);
        let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(PathBuf::from)).unwrap_or_default();
        let runtime_dir = exe_dir.join("runtime");
        let devices = list_devices(&runtime_dir).map_err(|e| format!("音声デバイスを取得できません: {e}"));
        let assets_dir = exe_dir.join("assets");
        let app = Self {
            settings: Settings::load(&assets_dir.join(DEFAULT_VOICE)),
            runtime_dir,
            assets_dir,
            devices,
            check: Arc::new(Mutex::new((None, Check::Pending))),
            session: Session::new(),
            started: None,
            options_open: false,
            options_width: 0.0,
            preset_edit: None,
            options_pos: egui::Pos2::ZERO,
            main_icon_hidden: false,
            options_icon_hidden: false,
            content_width: 0.0,
            ctx: cc.egui_ctx.clone(),
        };
        app.check_voice_model();
        app
    }

    /// Checks the selected voice model on a worker thread and keeps the result.
    fn check_voice_model(&self) {
        let Some(path) = self.settings.voice_model.clone() else { return };
        *self.check.lock().unwrap() = (Some(path.clone()), Check::Pending);
        let (check, assets, ctx) = (self.check.clone(), self.assets_dir.clone(), self.ctx.clone());
        std::thread::spawn(move || {
            let result = match voice_model::check(&path, &assets) {
                Ok(template) => Check::Ok(template),
                Err(Error::Unsupported(u)) => Check::Unsupported(match u {
                    Unsupported::Version(v) => format!("対応外の声モデルです（RVC v2 ではありません: {v}）"),
                    Unsupported::NoF0 => "対応外の声モデルです（F0 なし）".into(),
                    Unsupported::Config => "対応外の声モデルです（公式の学習設定 32k / 40k / 48k と違う設定）".into(),
                    Unsupported::Weights(k) => format!("対応外の声モデルです（重み {k} が無いか形が違います）"),
                }),
                Err(e) => Check::Unsupported(format!("声モデルを読めません: {e}")),
            };
            let mut c = check.lock().unwrap();
            if c.0.as_ref() == Some(&path) {
                c.1 = result;
            }
            ctx.request_repaint();
        });
    }

    fn start_key(&self) -> StartKey {
        let s = &self.settings;
        StartKey {
            backend: s.backend,
            native_chunk: s.native.chunk,
            native_extra: s.native.extra_ms,
            native_crossfade: s.native.crossfade_ms,
            native_graph: s.native.cuda_graph,
            input: s.input.clone(),
            output: s.output.clone(),
            monitor: s.monitor.clone(),
            exclusive: s.exclusive,
            voice_model: s.voice_model.clone(),
            f0: s.f0.clone(),
            block_ms: s.block_ms,
            crossfade_ms: s.crossfade_ms,
            extra_ms: s.extra_ms,
            formant: s.voice_model.as_ref().and_then(|p| s.voices.get(p)).map_or(0.0, |v| v.formant),
        }
    }

    /// Why Start cannot be pressed, or None when the configuration is complete.
    fn missing(&self) -> Option<String> {
        let s = &self.settings;
        let Ok(devices) = &self.devices else { return Some("音声デバイスを取得できません".into()) };
        if !devices.inputs.iter().any(|d| d.name == s.input) || !devices.outputs.iter().any(|d| d.name == s.output) {
            return Some("オプションで入力デバイスと出力デバイスを選んでください".into());
        }
        if s.monitor.as_ref().is_some_and(|m| !devices.outputs.iter().any(|d| &d.name == m)) {
            return Some("オプションでモニター出力デバイスを選び直してください".into());
        }
        if s.voice_model.is_none() {
            return Some("オプションで声モデルを選んでください".into());
        }
        if s.backend == Backend::Deiteris {
            let startup = self.native_startup();
            if let Err(e) = startup.tg_dims(40000) { return Some(format!("Deiteris の設定: {e}")); }
            for file in ["tg-fast-v1/audio-contract.json", "tg-fast-v1/contentvec.onnx", "tg-fast-v1/post.onnx",
                "tg-fast-v1/rmvpe-salience.onnx", "tg-fast-v1/hann.f32", "tg-fast-v1/mel-basis.f32",
                "tg-gpu-pitch-v2/contract.json", "tg-gpu-pitch-v2/frame.onnx", "tg-gpu-pitch-v2/mel.onnx",
                "tg-gpu-pitch-v2/rmvpe-salience.onnx", "tg-gpu-pitch-v2/decode.onnx"] {
                if !self.assets_dir.join("deiteris").join(file).is_file() {
                    return Some(format!("Deiteris の資産がありません: {file}"));
                }
            }
        } else if let Some(problem) = over_pitch_cache(s.block_ms, s.crossfade_ms, s.extra_ms) {
            return Some(problem);
        }
        match &self.check.lock().unwrap().1 {
            Check::Pending => Some("声モデルを確認しています…".into()),
            Check::Unsupported(reason) => Some(reason.clone()),
            Check::Ok(_) => None,
        }
    }

    fn start(&mut self) {
        let s = self.settings.clone();
        let (Some(voice_model), Ok(devices)) = (s.voice_model.clone(), &self.devices) else { return };
        let voice = self.settings.voice().copied().unwrap_or(settings::Voice { pitch: 0.0, formant: 0.0 });
        let sample_rate = devices.inputs.iter().find(|d| d.name == s.input).map_or(48000, |d| d.default_sample_rate);
        let f0 = if s.f0 == "rmvpe" { F0Method::Rmvpe } else { F0Method::Fcpe };
        self.started = Some(self.start_key());
        self.session.start(Request {
            native: (s.backend == Backend::Deiteris).then(|| self.native_startup()),
            cuda_graph: s.native.cuda_graph,
            voice_model,
            assets_dir: self.assets_dir.clone(),
            models_dir: settings::models_dir(),
            startup: Startup { sample_rate, block_ms: s.block_ms, crossfade_ms: s.crossfade_ms, extra_ms: s.extra_ms, formant: voice.formant, f0, variant: APP_VARIANT },
            devices: Devices { input: s.input, output: s.output, monitor: s.monitor, wasapi_exclusive: s.exclusive },
            options: RealtimeOptions { engine: EngineOptions { runtime_dir: self.runtime_dir.clone(), seed: 1 } },
            pitch: voice.pitch,
            rms_mix: s.rms_mix,
            threshold_db: if s.backend == Backend::Deiteris { s.native.threshold_db } else { s.threshold_db },
            skip_silence: s.backend == Backend::Legacy && s.skip_silence,
            monitor_volume: s.monitor_volume as f32 / 100.0,
        });
    }

    fn native_startup(&self) -> rvc_engine::DeiterisStartup {
        let s = &self.settings;
        let sample_rate = self.devices.as_ref().ok()
            .and_then(|d| d.inputs.iter().find(|d| d.name == s.input)).map_or(48000, |d| d.default_sample_rate);
        rvc_engine::DeiterisStartup {
            sample_rate: sample_rate as usize, chunk: s.native.chunk,
            block_frames: None,
            extra_ms: s.native.extra_ms, crossfade_ms: s.native.crossfade_ms,
            formant: s.voice_model.as_ref().and_then(|p| s.voices.get(p)).map_or(0.0, |v| v.formant),
        }
    }

    fn effective_block_ms(&self) -> f64 {
        if self.settings.backend == Backend::Deiteris {
            let s = self.native_startup();
            s.chunk as f64 * 128000.0 / s.sample_rate as f64
        } else { self.settings.block_ms }
    }

    fn main_row(&mut self, ui: &mut egui::Ui, stage: Stage) {
        let missing = self.missing();
        let row = ui.horizontal(|ui| {
            let size = egui::vec2(160.0, 40.0);
            match stage {
                Stage::Idle => {
                    if ui.add_enabled(missing.is_none(), egui::Button::new("スタート").min_size(size)).clicked() {
                        self.start();
                    }
                }
                Stage::Converting => {
                    ui.add_enabled(false, egui::Button::new("変換中…").min_size(size));
                }
                Stage::Loading => {
                    ui.add_enabled(false, egui::Button::new("読み込み中…").min_size(size));
                }
                Stage::Running => {
                    // while converting, the Stop button shows it like an on-air light
                    let stop = egui::Button::new(egui::RichText::new("停止").color(egui::Color32::from_gray(230))).fill(RUNNING_COLOR).min_size(size);
                    if ui.add(stop).clicked() {
                        self.session.stop();
                        self.started = None;
                    }
                }
            }
            if ui.add(egui::Button::new("オプション").min_size(egui::vec2(100.0, 40.0)).selected(self.options_open)).clicked() {
                self.options_open = !self.options_open;
                self.options_icon_hidden = false;
                if let Some(main) = ui.ctx().input(|i| i.viewport().outer_rect) {
                    self.options_pos = egui::pos2(main.right(), main.top());
                }
            }
        });
        ui.set_max_width(row.response.rect.width().max(self.content_width));
        if let Some(error) = self.session.error() {
            ui.add(egui::Label::new(egui::RichText::new(error).color(ui.visuals().error_fg_color)).wrap());
        } else if stage == Stage::Running {
            let ms = self.session.with_realtime(|rt| rt.infer_ms()).unwrap_or(0.0);
            let text = format!("変換時間 {ms:.1} ms");
            // a block has to be converted within its own length, or the audio stream runs dry
            if ms as f64 > self.effective_block_ms() {
                ui.colored_label(ui.visuals().error_fg_color, format!("{text}（ブロック長 {:.2} ms に間に合っていません）", self.effective_block_ms()));
            } else {
                ui.label(text);
            }
            let latency = self.session.with_realtime(|rt| rt.onset_latency()).flatten();
            ui.label(match latency {
                Some((ms, _)) => format!("発声→出力（ブロック実測） {ms:.0} ms"),
                None => "発声→出力 — 無音からの発声待ち".to_owned(),
            }).on_hover_text(format!("発声を含む入力ブロックの受信から、その発声が現れた出力ブロックの書き込み完了までを計測します。直近の発声の値を保持します。\n{}\n両側300msの無音後、−45dBFSを5ms以上で検出。収録済みブロックの受信から測るため、受信前の収録待ち・機器の再生待ちは含みません。", match latency {
                Some((_, blocks)) => format!("今回の記録：入力検出から出力までのブロック間隔 {blocks}。"),
                None => "まだ発声と出力の組を検出していません。".to_owned(),
            }));
        } else if stage == Stage::Idle {
            if let Some(m) = missing {
                ui.add(egui::Label::new(m).wrap());
            }
        }
    }

    fn options(&mut self, ui: &mut egui::Ui) {
        let device_rate = self.native_startup().sample_rate;
        let voice_quality_unverified = matches!(&self.check.lock().unwrap().1, Check::Ok(name) if name == "32k" || name == "48k");
        let names = |f: fn(&DeviceList) -> &Vec<rvc_engine::DeviceEntry>| -> Vec<String> {
            self.devices.as_ref().map(|d| f(d).iter().map(|e| e.name.clone()).collect()).unwrap_or_default()
        };
        let (inputs, outputs) = (names(|d| &d.inputs), names(|d| &d.outputs));
        let s = &mut self.settings;
        let mut picked = false;

        ui.heading("設定");
        let mut load = None;
        egui::ComboBox::from_id_salt("preset").width(DEVICE_COMBO_WIDTH).selected_text(s.active_preset.clone()).show_ui(ui, |ui| {
            for p in &s.presets {
                let active = p.name == s.active_preset;
                if ui.selectable_label(active, &p.name).clicked() && !active {
                    load = Some(p.clone());
                }
            }
        });
        if let Some(p) = load {
            s.load_preset(&p);
            s.active_preset = p.name;
            self.preset_edit = None;
            picked = true;
        }
        ui.add_space(MARGIN);
        let modified = s.is_modified();
        let is_default = s.active_preset == settings::DEFAULT_PRESET;
        match &mut self.preset_edit {
            None => {
                ui.horizontal(|ui| {
                    // unsaved changes highlight the two save buttons
                    let save_button = |ui: &egui::Ui, text: &str| {
                        let v = ui.visuals();
                        if modified {
                            egui::Button::new(egui::RichText::new(text).color(v.selection.stroke.color)).fill(v.selection.bg_fill)
                        } else {
                            egui::Button::new(text)
                        }
                    };
                    if ui.add(save_button(ui, "新規作成")).clicked() {
                        self.preset_edit = Some(PresetEdit { kind: PresetEditKind::New, name: String::new(), problem: None, focus: true });
                    }
                    if ui.add_enabled(modified, save_button(ui, "上書き保存")).clicked() {
                        let name = s.active_preset.clone();
                        s.save_preset(&name);
                    }
                    if ui.add_enabled(!is_default, egui::Button::new("名前変更")).clicked() {
                        let name = s.active_preset.clone();
                        self.preset_edit = Some(PresetEdit { kind: PresetEditKind::Rename, name, problem: None, focus: true });
                    }
                    if ui.add_enabled(!is_default, egui::Button::new("削除")).clicked() {
                        s.delete_active_preset();
                        picked = true;
                    }
                });
            }
            Some(edit) => {
                let mut done = false;
                ui.horizontal(|ui| {
                    let field = ui.add(egui::TextEdit::singleline(&mut edit.name).hint_text("名前").desired_width(DEVICE_COMBO_WIDTH));
                    if std::mem::take(&mut edit.focus) {
                        field.request_focus();
                    }
                    let enter = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let name = edit.name.trim().to_string();
                    if (ui.add_enabled(!name.is_empty(), egui::Button::new("決定")).clicked() || enter) && !name.is_empty() {
                        let unchanged_rename = matches!(edit.kind, PresetEditKind::Rename) && name == s.active_preset;
                        if unchanged_rename {
                            done = true;
                        } else if !s.name_is_free(&name) {
                            edit.problem = Some("同じ名前の設定があります");
                        } else {
                            match edit.kind {
                                PresetEditKind::New => {
                                    // a new setting starts from the current options with a voice model picked right away
                                    let picked_model = rfd::FileDialog::new().add_filter("声モデル", &["pth", "safetensors"]).pick_file();
                                    if let Some(path) = picked_model {
                                        s.voice_model = Some(path);
                                        picked = true;
                                    }
                                    s.save_preset(&name);
                                    s.active_preset = name;
                                }
                                PresetEditKind::Rename => s.rename_active_preset(&name),
                            }
                            done = true;
                        }
                    }
                    if ui.button("キャンセル").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        done = true;
                    }
                });
                if let Some(problem) = edit.problem {
                    ui.colored_label(ui.visuals().error_fg_color, problem);
                }
                if done {
                    self.preset_edit = None;
                }
            }
        }

        section_divider(ui, self.options_width);
        ui.heading("声モデル");
        ui.horizontal(|ui| {
            let name = s.voice_model.as_ref().and_then(|p| p.file_name()).map_or("未選択".into(), |n| n.to_string_lossy().into_owned());
            ui.label(name);
            // the default setting always uses the bundled voice model
            let fixed = s.active_preset == settings::DEFAULT_PRESET;
            if ui.add_enabled(!fixed, egui::Button::new("選ぶ…")).clicked() {
                if let Some(path) = rfd::FileDialog::new().add_filter("声モデル", &["pth", "safetensors"]).pick_file() {
                    s.voice_model = Some(path);
                    picked = true;
                }
            }
            if fixed {
                ui.weak("デフォルト変更不可、新規作成をしてください");
            }
        });
        if let Check::Unsupported(reason) = &self.check.lock().unwrap().1 {
            if s.voice_model.is_some() {
                ui.colored_label(ui.visuals().error_fg_color, reason);
            }
        }

        section_divider(ui, self.options_width);
        ui.heading("音声デバイス");
        egui::Grid::new("devices").num_columns(2).show(ui, |ui| {
            ui.label("入力デバイス");
            combo(ui, "input", &mut s.input, &inputs);
            ui.end_row();
            ui.label("出力デバイス");
            combo(ui, "output", &mut s.output, &outputs);
            ui.end_row();
            ui.label("モニター出力デバイス");
            egui::ComboBox::from_id_salt("monitor").width(DEVICE_COMBO_WIDTH).selected_text(s.monitor.clone().unwrap_or_else(|| "なし".into())).show_ui(ui, |ui| {
                ui.selectable_value(&mut s.monitor, None, "なし");
                for n in &outputs {
                    ui.selectable_value(&mut s.monitor, Some(n.clone()), n);
                }
            });
            ui.end_row();
            ui.label("モニター音量");
            ui.add(egui::Slider::new(&mut s.monitor_volume, 0..=100).suffix(" %"));
            ui.end_row();
            ui.label("");
            ui.checkbox(&mut s.exclusive, "WASAPI 排他");
            ui.end_row();
        });

        section_divider(ui, self.options_width);
        ui.heading("声の調整");
        ui.horizontal(|ui| {
            ui.label("変換経路");
            ui.selectable_value(&mut s.backend, Backend::Deiteris, "Deiteris（検証中）");
            ui.selectable_value(&mut s.backend, Backend::Legacy, "従来");
        });
        let native = s.backend == Backend::Deiteris;
        if native {
            if device_rate == 44100 {
                ui.weak("44.1kHzは原本との有声判定差が未解決です。");
            }
            if voice_quality_unverified {
                ui.weak("32k/48k声モデルは構造のみ検証済みで、実声音質は未確認です。");
            }
        }
        egui::Grid::new("voice").num_columns(2).show(ui, |ui| {
            if let Some(v) = s.voice() {
                ui.label("pitch");
                ui.add(egui::Slider::new(&mut v.pitch, -24.0..=24.0).step_by(1.0));
                ui.end_row();
                ui.label("formant");
                ui.add(egui::Slider::new(&mut v.formant, -12.0..=12.0).step_by(0.1).max_decimals(1));
                ui.end_row();
            }
            ui.label("rms_mix");
            ui.add_enabled(!native, egui::Slider::new(&mut s.rms_mix, 0.0..=1.0).step_by(0.01));
            ui.end_row();
            ui.label("F0 方式");
            ui.add_enabled_ui(!native, |ui| {
                ui.radio_value(&mut s.f0, "rmvpe".to_string(), "RMVPE");
                ui.radio_value(&mut s.f0, "fcpe".to_string(), "FCPE");
            });
            ui.end_row();
        });

        section_divider(ui, self.options_width);
        ui.heading("雑音");
        if native {
            ui.horizontal(|ui| {
                ui.label("Deiteris 無音しきい値");
                ui.add(egui::Slider::new(&mut s.native.threshold_db, -120.0..=0.0).step_by(1.0).suffix(" dB"));
            });
            ui.weak("原本の無音処理（既定 −90dB）。無音でも推論は継続します。従来の省電力設定は適用しません。");
        } else {
        egui::Grid::new("noise").num_columns(2).show(ui, |ui| {
            ui.label("無音のしきい値");
            ui.add(egui::Slider::new(&mut s.threshold_db, -60.0..=0.0).step_by(1.0).suffix(" dB"));
            ui.end_row();
            ui.label("");
            ui.weak(if s.threshold_db <= -60.0 { "切（すべて変換します）" } else { "これより小さい音を無音とみなします" });
            ui.end_row();
            ui.label("");
            ui.add_enabled_ui(s.threshold_db > -60.0, |ui| {
                ui.checkbox(&mut s.skip_silence, "無音のときは変換しない（文脈にも入れない）");
            });
            ui.end_row();
        });
        }

        section_divider(ui, self.options_width);
        ui.heading("性能");
        if native {
            egui::Grid::new("native_perf").num_columns(2).show(ui, |ui| {
                ui.label("ブロック長（約）");
                let chunk_ms = 128000.0 / device_rate as f64;
                ui.add(egui::Slider::new(&mut s.native.chunk, 1..=256)
                    .custom_formatter(move |chunk, _| format!("{:.0}", chunk * chunk_ms))
                    .custom_parser(move |text| text.trim().parse::<f64>().ok()
                        .filter(|ms| ms.is_finite())
                        .map(|ms| (ms / chunk_ms).round().clamp(1.0, 256.0)))
                    .suffix(" ms"));
                ui.end_row();
                ui.label("クロスフェード");
                ui.add(egui::Slider::new(&mut s.native.crossfade_ms, 1.0..=1000.0).step_by(1.0).suffix(" ms"));
                ui.end_row();
                ui.label("文脈長");
                ui.add(egui::Slider::new(&mut s.native.extra_ms, 50.0..=5000.0).step_by(10.0).suffix(" ms"));
                ui.end_row();
                ui.label("生成器");
                ui.checkbox(&mut s.native.cuda_graph, "CUDA Graph");
                ui.end_row();
            });
        } else {
        egui::Grid::new("perf").num_columns(2).show(ui, |ui| {
            ui.label("ブロック長");
            ui.add(egui::Slider::new(&mut s.block_ms, 20.0..=1000.0).step_by(1.0).suffix(" ms"));
            ui.end_row();
            ui.label("クロスフェード");
            ui.add(egui::Slider::new(&mut s.crossfade_ms, 10.0..=100.0).step_by(1.0).suffix(" ms"));
            ui.end_row();
            ui.label("文脈長");
            ui.add(egui::Slider::new(&mut s.extra_ms, 500.0..=10000.0).step_by(10.0).suffix(" ms"));
            ui.end_row();
            if let Some(problem) = over_pitch_cache(s.block_ms, s.crossfade_ms, s.extra_ms) {
                ui.label("");
                ui.colored_label(ui.visuals().error_fg_color, problem);
                ui.end_row();
            }
        });
        }
        if picked {
            self.check_voice_model();
        }
    }
}

/// The engine keeps 1024 frames of 10 ms of pitch history; block, crossfade and context share it.
fn over_pitch_cache(block_ms: f64, crossfade_ms: f64, extra_ms: f64) -> Option<String> {
    const BUDGET_MS: f64 = 10_240.0;
    let need = (block_ms + crossfade_ms + extra_ms + 10.0).ceil();
    (need > BUDGET_MS).then(|| {
        format!("ブロック長・クロスフェード・文脈長の合計が {need:.0} ms です。{BUDGET_MS:.0} ms までにしてください")
    })
}

fn combo(ui: &mut egui::Ui, id: &str, value: &mut String, names: &[String]) {
    let text = if value.is_empty() { "未選択".to_string() } else { value.clone() };
    egui::ComboBox::from_id_salt(id).width(DEVICE_COMBO_WIDTH).selected_text(text).show_ui(ui, |ui| {
        for n in names {
            ui.selectable_value(value, n.clone(), n);
        }
    });
}

/// Space and a line between two groups of options. The line spans `width` (the content width of the last frame):
/// a plain `ui.separator()` takes all available width, which an Area does not bound.
fn section_divider(ui: &mut egui::Ui, width: f32) {
    ui.add_space(DIVIDER_SPACE);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 1.0), egui::Sense::hover());
    ui.painter().hline(rect.x_range(), rect.center().y, ui.visuals().widgets.noninteractive.bg_stroke);
    ui.add_space(DIVIDER_SPACE);
}

/// Draws `add` in an Area (which takes the size of what is drawn) and sets the window to that size plus margins.
fn fit_to_content(ctx: &egui::Context, add: impl FnOnce(&mut egui::Ui)) -> egui::Vec2 {
    let area = egui::Area::new(egui::Id::new("content")).fixed_pos(egui::pos2(MARGIN, MARGIN)).show(ctx, add);
    let content = area.response.rect.size();
    let size = (content + egui::vec2(2.0 * MARGIN, 2.0 * MARGIN)).ceil();
    if ctx.input(|i| i.viewport().inner_rect).is_none_or(|r| (r.size() - size).abs().max_elem() > 1.0) {
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
        // paint the resized window once more so the newly exposed area gets the background
        ctx.request_repaint();
    }
    content
}

/// Removes the title-bar icon of this process's window titled `title`: a dialog frame for the main window
/// (keeps its minimize button), a tool window for the options window (only a close button; a dialog frame
/// without minimize/maximize still shows the icon). Returns false while the window does not exist yet.
fn hide_title_icon(title: &str, tool: bool) -> bool {
    use windows_sys::Win32::Foundation::{HWND, LPARAM};
    use windows_sys::Win32::UI::WindowsAndMessaging::*;

    struct Search {
        title: Vec<u16>,
        found: HWND,
    }
    unsafe extern "system" fn visit(hwnd: HWND, lparam: LPARAM) -> i32 {
        let search = &mut *(lparam as *mut Search);
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if pid != std::process::id() {
            return 1;
        }
        let mut buf = [0u16; 128];
        let n = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32) as usize;
        if buf[..n] == search.title[..] {
            search.found = hwnd;
            return 0;
        }
        1
    }
    let mut search = Search { title: title.encode_utf16().collect(), found: std::ptr::null_mut() };
    unsafe {
        EnumWindows(Some(visit), &mut search as *mut Search as LPARAM);
        let hwnd = search.found;
        if hwnd.is_null() {
            return false;
        }
        let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        let add = if tool { WS_EX_TOOLWINDOW } else { WS_EX_DLGMODALFRAME };
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex | add as isize);
        // no small icon, so the title bar shows none; the main window keeps the exe's icon as its big icon for the
        // taskbar and Alt+Tab
        SendMessageW(hwnd, WM_SETICON, ICON_SMALL as usize, 0);
        let big = if tool {
            std::ptr::null_mut()
        } else {
            let module = windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null());
            LoadIconW(module, 1 as *const u16) // winresource embeds the app icon as resource 1
        };
        SendMessageW(hwnd, WM_SETICON, ICON_BIG as usize, big as isize);
        SetWindowPos(hwnd, std::ptr::null_mut(), 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
    }
    true
}

fn japanese_font(ctx: &egui::Context) {
    let Ok(bytes) = std::fs::read(r"C:\Windows\Fonts\YuGothM.ttc") else { return };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert("jp".into(), Arc::new(egui::FontData::from_owned(bytes)));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("jp".into());
    }
    ctx.set_fonts(fonts);
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.session.poll();
        let before = self.settings.clone();
        let stage = self.session.stage();
        egui::CentralPanel::default().show(ctx, |_| {});
        if !self.main_icon_hidden {
            self.main_icon_hidden = hide_title_icon("RVC リアルタイム変換", false);
        }
        let content = fit_to_content(ctx, |ui| self.main_row(ui, stage));
        self.content_width = content.x;
        if self.options_open {
            let viewport = egui::ViewportBuilder::default()
                .with_title("オプション")
                .with_inner_size([300.0, 300.0])
                .with_position(self.options_pos)
                .with_resizable(false)
                .with_minimize_button(false)
                .with_maximize_button(false);
            ctx.show_viewport_immediate(egui::ViewportId::from_hash_of("options"), viewport, |ctx, _| {
                egui::CentralPanel::default().show(ctx, |_| {});
                if !self.options_icon_hidden {
                    self.options_icon_hidden = hide_title_icon("オプション", true);
                }
                self.options_width = fit_to_content(ctx, |ui| self.options(ui)).x;
                if ctx.input(|i| i.viewport().close_requested()) {
                    self.options_open = false;
                    self.options_icon_hidden = false;
                    // Windows would otherwise activate another app's window and leave the main window behind it
                    ctx.send_viewport_cmd_to(egui::ViewportId::ROOT, egui::ViewportCommand::Focus);
                }
            });
        }
        if self.settings != before {
            self.settings.save();
        }

        let stage = self.session.stage();
        if stage == Stage::Idle {
            self.started = None;
        }
        // live values go straight to the engine; anything else restarts it once the value is set and the
        // current start has finished
        if let Some(voice) = self.settings.voice().copied() {
            let (rms_mix, volume) = (self.settings.rms_mix, self.settings.monitor_volume as f32 / 100.0);
            let native = self.started.as_ref().is_some_and(|k| k.backend == Backend::Deiteris);
            let (threshold, skip) = if native { (self.settings.native.threshold_db, false) }
                else { (self.settings.threshold_db, self.settings.skip_silence) };
            self.session.with_realtime(|rt| {
                rt.set_pitch(voice.pitch);
                rt.set_rms_mix(rms_mix);
                rt.set_threshold_db(threshold);
                rt.set_skip_silence(skip);
                rt.set_drop_silent_context(skip);
                rt.set_monitor_volume(volume);
            });
        }
        let dragging = ctx.input(|i| i.pointer.any_down());
        let checking = matches!(self.check.lock().unwrap().1, Check::Pending) && self.settings.voice_model.is_some();
        // an incomplete new configuration keeps the current run going
        if stage == Stage::Running && !dragging && !checking && self.started.as_ref().is_some_and(|k| *k != self.start_key()) && self.missing().is_none() {
            self.start();
        }
        if stage != Stage::Idle {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }
}
