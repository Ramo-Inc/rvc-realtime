//! Streaming engine: a port of the official `RVCStreamEngine.process` (RVCRealtimeVST/worker/rvc_worker.py)
//! and `RVC.infer` (infer/rtrvc.py), commit 81eed5e. Not ported: index retrieval, the threshold gate,
//! TorchGate noise reduction, PM F0.

use std::path::PathBuf;

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::config::{Dims, F0Method, Model, Setup, Startup};
use crate::dsp::{self, Resampler, Stft};
use crate::error::{Error, Result};
use crate::infer::{self, Models};
use crate::noise;

/// Slots of speech a run needs before the context keeps it: shorter bursts are noise, not voice.
const MIN_VOICED_SLOTS: usize = 20;
/// Slots of silence before the models stop converting (300 ms). One loud slot resumes at once.
const SKIP_AFTER_SLOTS: usize = 30;
/// Samples of the fade on both sides of a splice in the context (16 kHz).
const SEAM: usize = 40;

/// Next `randn_like` draw of `shape` (same call order as the official generator: z_p, then SineGen).
fn draw(rng: &mut rand::rngs::StdRng, shape: &[usize], out: &mut Vec<f32>) {
    let n: usize = shape.iter().product();
    out.resize(n, 0.0);
    for v in out.iter_mut() {
        *v = StandardNormal.sample(rng);
    }
}

pub struct EngineOptions {
    /// directory holding the CUDA 13 / cuDNN 9 DLLs; put first on PATH before the first session loads
    pub runtime_dir: PathBuf,
    /// seed for the generator noise (the official engine draws it with `torch.randn_like`)
    pub seed: u64,
}

/// Per-block parameters. `pitch` in semitones (-24..=24), `rms_mix` 0..=1; values outside are clamped.
/// `threshold_db` is the official silence gate: -60 and below turns it off (`realtime_gui.py`), and it is
/// also what `drop_silent_context` calls silence.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub pitch: f32,
    pub rms_mix: f32,
    pub threshold_db: f32,
    /// keep only speech in the context: silence and bursts shorter than 200 ms never enter it
    pub drop_silent_context: bool,
    /// do not convert while the input stays below the threshold (the models idle, the stream keeps running)
    pub skip_silence: bool,
}

pub struct Engine {
    d: Dims,
    fcpe: bool,
    upp: usize,
    formant: f32,
    models: Models,
    stft: Stft,
    rs_in: Resampler,
    rs_out: Option<Resampler>,
    rs_formant: Option<Resampler>,
    input_wav: Vec<f32>,
    /// 16 kHz input: the context (oldest) followed by the tail, which is what the block is converted from.
    /// Only the context is allowed to drop slots; `hist16` is where it ends.
    input_wav_res: Vec<f32>,
    /// samples of `input_wav_res` that belong to the context, and frames of the pitch cache that do
    hist16: usize,
    hist_frames: usize,
    /// slots that left the tail this block and are kept, gathered before they are overwritten
    keep16: Vec<f32>,
    keep_pitch: Vec<i64>,
    keep_pitchf: Vec<f32>,
    /// slots of the run of speech at the end of the context, and whether a gap precedes the next slot
    run_slots: usize,
    seam: bool,
    /// how long the input has been below the threshold, and whether the models are idling
    quiet_slots: usize,
    skipping: bool,
    sola_buffer: Vec<f32>,
    fade_in: Vec<f32>,
    cache_pitch: Vec<i64>,
    cache_pitchf: Vec<f32>,
    rng: rand::rngs::StdRng,
    rnd: Vec<f32>,
    sine: Vec<f32>,
    out: Vec<f32>,
}

impl Engine {
    /// Computes the buffer lengths for `startup`, creates the sessions and runs the official `prewarm`.
    pub fn new(model: &Model, startup: &Startup, opts: &EngineOptions) -> Result<Self> {
        infer::prepend_path(&opts.runtime_dir);
        let setup = Setup { model, startup, dims: Dims::compute(model, startup)? };
        let d = setup.dims.clone();
        let models = Models::load(&setup)?;
        let rs_out = (d.model_sr != d.device_sr).then(|| Resampler::new(d.model_sr, d.device_sr));
        let model_window = d.model_sr / 100;
        let rs_formant = (d.upp_res != model_window).then(|| Resampler::new(d.upp_res, model_window));
        let sb = d.sola_buffer_frame;
        // torch.sin(0.5 * pi * torch.linspace(0, 1, steps=sb)) ** 2 (float32)
        let fade_in = (0..sb)
            .map(|i| {
                let x = if sb == 1 { 0.0 } else { i as f32 / (sb - 1) as f32 };
                (0.5 * std::f32::consts::PI * x).sin().powi(2)
            })
            .collect();
        let mut engine = Self {
            fcpe: startup.f0 == F0Method::Fcpe,
            upp: model.upp,
            formant: startup.formant as f32,
            models,
            stft: Stft::new(),
            rs_in: Resampler::new(d.device_sr, 16000),
            rs_out,
            rs_formant,
            input_wav: vec![0.0; d.input_wav_len],
            input_wav_res: vec![0.0; d.n16],
            hist16: d.n16 - 160 * (d.crossfade_frame + d.sola_search_frame + d.block_frame) / d.zc,
            hist_frames: d.pitch_cache_len - (d.crossfade_frame + d.sola_search_frame + d.block_frame) / d.zc,
            keep16: Vec::new(),
            keep_pitch: Vec::new(),
            keep_pitchf: Vec::new(),
            run_slots: 0,
            seam: false,
            quiet_slots: 0,
            skipping: false,
            sola_buffer: vec![0.0; sb],
            fade_in,
            cache_pitch: vec![0; d.pitch_cache_len],
            cache_pitchf: vec![0.0; d.pitch_cache_len],
            rng: rand::rngs::StdRng::seed_from_u64(opts.seed),
            rnd: Vec::new(),
            sine: Vec::new(),
            out: Vec::new(),
            d,
        };
        engine.prewarm()?;
        Ok(engine)
    }

    pub fn block_frames(&self) -> usize {
        self.d.block_frame
    }

    /// Official `prewarm`: one block of a 220 Hz sine at the official default parameters, then all
    /// streaming state back to zero. The generator noise sequence is left untouched.
    fn prewarm(&mut self) -> Result<()> {
        let saved = std::mem::replace(&mut self.rng, rand::rngs::StdRng::seed_from_u64(12345));
        let sr = self.d.device_sr as f32;
        let probe: Vec<f32> = (0..self.d.block_frame)
            .map(|i| 0.05 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr).sin())
            .collect();
        self.process(&probe, &Params { pitch: 12.0, rms_mix: 0.5, threshold_db: -60.0, drop_silent_context: false, skip_silence: false })?;
        self.rng = saved;
        self.input_wav.fill(0.0);
        self.input_wav_res.fill(0.0);
        self.sola_buffer.fill(0.0);
        self.run_slots = 0;
        self.seam = false;
        self.quiet_slots = 0;
        self.skipping = false;
        self.cache_pitch.fill(0);
        self.cache_pitchf.fill(0.0);
        Ok(())
    }

    /// Appends one 10 ms slot to the context, dropping its oldest slot. After a gap, the seam is faded
    /// so the splice is not a step the generator has never seen.
    fn push_slot(&mut self, audio: &[f32; 160], pitch: i64, pitchf: f32) {
        let (h, hf) = (self.hist16, self.hist_frames);
        self.input_wav_res.copy_within(160..h, 0);
        self.input_wav_res[h - 160..h].copy_from_slice(audio);
        if std::mem::take(&mut self.seam) {
            for i in 0..SEAM {
                let w = i as f32 / SEAM as f32;
                self.input_wav_res[h - 160 + i] *= w;
                self.input_wav_res[h - 160 - SEAM + i] *= 1.0 - w;
            }
        }
        self.cache_pitch.copy_within(1..hf, 0);
        self.cache_pitchf.copy_within(1..hf, 0);
        self.cache_pitch[hf - 1] = pitch;
        self.cache_pitchf[hf - 1] = pitchf;
    }

    /// Takes the newest `n` slots back out of the context (the run turned out too short to be speech).
    fn drop_slots(&mut self, n: usize) {
        let (h, hf) = (self.hist16, self.hist_frames);
        let s = (n * 160).min(h);
        let n = (s / 160).min(hf);
        self.input_wav_res.copy_within(..h - s, s);
        self.input_wav_res[..s].fill(0.0);
        self.cache_pitch.copy_within(..hf - n, n);
        self.cache_pitchf.copy_within(..hf - n, n);
        self.cache_pitch[..n].fill(0);
        self.cache_pitchf[..n].fill(0.0);
    }

    /// One block of silence: no ContentVec, no F0, no generator. The pitch cache gets unvoiced frames,
    /// the noise draws continue so the sequence matches a converted run, and the first skipped block
    /// fades the overlap tail out instead of cutting to digital silence.
    fn skip_block(&mut self) -> &[f32] {
        let d = &self.d;
        let (bf, cl) = (d.block_frame, self.cache_pitch.len());
        let frames = d.f0_frames.min(cl);
        self.cache_pitch[cl - frames..].fill(0);
        self.cache_pitchf[cl - frames..].fill(0.0);
        let rnd_shape = [1usize, d.inter_channels, d.p_len - d.flow_head];
        let sine_shape = [1usize, d.return_length * self.upp, 1];
        draw(&mut self.rng, &rnd_shape, &mut self.rnd);
        draw(&mut self.rng, &sine_shape, &mut self.sine);
        self.out.clear();
        self.out.resize(bf, 0.0);
        if !self.skipping {
            self.skipping = true;
            let sb = self.sola_buffer.len().min(bf);
            for i in 0..sb {
                self.out[i] = self.sola_buffer[i] * (1.0 - self.fade_in[i]);
            }
            self.sola_buffer.fill(0.0);
        }
        &self.out
    }

    /// `RVCStreamEngine.process`: `block_frame` samples in, `block_frame` samples out.
    pub fn process(&mut self, block: &[f32], params: &Params) -> Result<&[f32]> {
        let pitch = params.pitch.clamp(-24.0, 24.0);
        let rms_mix = params.rms_mix.clamp(0.0, 1.0);
        let d = &self.d;
        let (zc, bf, bf16) = (d.zc, d.block_frame, d.block_frame_16k);
        if block.len() != bf {
            return Err(Error::BlockSize { expected: bf, got: block.len() });
        }

        // input buffers
        let taken = block.len();
        self.input_wav.copy_within(bf.., 0);
        let n = self.input_wav.len();
        self.input_wav[n - taken..].copy_from_slice(block);
        // The oldest block of the tail leaves it now. The context takes what it keeps of those slots,
        // then both the context and the tail move by their own amount.
        let slots = bf16 / 160;
        let (h, hf) = (self.hist16, self.hist_frames);
        let drop_silence = params.drop_silent_context && params.threshold_db > -60.0;
        if drop_silence {
            self.keep16.clear();
            self.keep16.extend_from_slice(&self.input_wav_res[h..h + bf16]);
            self.keep_pitch.clear();
            self.keep_pitchf.clear();
            self.keep_pitch.extend_from_slice(&self.cache_pitch[hf..hf + slots]);
            self.keep_pitchf.extend_from_slice(&self.cache_pitchf[hf..hf + slots]);
            for i in 0..slots {
                let audio: [f32; 160] = self.keep16[i * 160..(i + 1) * 160].try_into().unwrap();
                if noise::slot_db(&audio) >= params.threshold_db {
                    self.push_slot(&audio, self.keep_pitch[i], self.keep_pitchf[i]);
                    self.run_slots += 1;
                } else {
                    // a burst too short to be speech leaves the context again
                    if self.run_slots > 0 && self.run_slots < MIN_VOICED_SLOTS {
                        self.drop_slots(self.run_slots);
                    }
                    self.run_slots = 0;
                    self.seam = true;
                }
            }
        } else {
            self.keep16.clear();
            self.keep16.extend_from_slice(&self.input_wav_res[h..h + bf16]);
            let kept = self.keep16.len();
            self.input_wav_res.copy_within(kept..h, 0);
            self.input_wav_res[h - kept..h].copy_from_slice(&self.keep16);
            self.keep_pitch.clear();
            self.keep_pitchf.clear();
            self.keep_pitch.extend_from_slice(&self.cache_pitch[hf..hf + slots]);
            self.keep_pitchf.extend_from_slice(&self.cache_pitchf[hf..hf + slots]);
            let kept_frames = self.keep_pitch.len();
            self.cache_pitch.copy_within(kept_frames..hf, 0);
            self.cache_pitchf.copy_within(kept_frames..hf, 0);
            self.cache_pitch[hf - kept_frames..hf].copy_from_slice(&self.keep_pitch);
            self.cache_pitchf[hf - kept_frames..hf].copy_from_slice(&self.keep_pitchf);
            self.run_slots = usize::MAX / 2;
            self.seam = false;
        }
        self.input_wav_res.copy_within(h + bf16.., h);
        self.cache_pitch.copy_within(hf + slots.., hf);
        self.cache_pitchf.copy_within(hf + slots.., hf);
        let take = taken;
        let resampled = self.rs_in.process(&self.input_wav[n - take - 2 * zc..]);
        let resampled = &resampled[160..];
        let n16 = self.input_wav_res.len();
        // one block, or everything the gate rewrote (`realtime_gui.py`: 160 * (len // zc + 1) samples)
        let written = if take == bf { bf16 } else { 160 * (take / zc + 1) };
        self.input_wav_res[n16 - written..].copy_from_slice(&resampled[resampled.len() - written..]);

        // Silence: let the models idle instead of converting nothing. The buffers above stay current,
        // so the first loud block resumes with the same state a converted run would have had.
        if params.skip_silence && params.threshold_db > -60.0 {
            let loud = block.chunks(zc).any(|slot| noise::slot_db(slot) >= params.threshold_db);
            self.quiet_slots = if loud { 0 } else { self.quiet_slots + block.len() / zc };
            if self.quiet_slots >= SKIP_AFTER_SLOTS {
                return Ok(self.skip_block());
            }
        } else {
            self.quiet_slots = 0;
        }
        self.skipping = false;

        // ---- RVC.infer
        let mut feats = self.models.contentvec(&self.input_wav_res)?;
        let frames = feats.len() / 768;
        let last = feats[(frames - 1) * 768..].to_vec();
        feats.extend_from_slice(&last);

        let d = &self.d;
        let p_len = d.p_len;
        let (r, r2) = (d.return_length, d.return_length2);
        let key = pitch - self.formant;
        let x = &self.input_wav_res[n16 - d.f0_extractor_frame..];
        let f0 = if self.fcpe {
            let (mag, fr) = self.stft.fcpe(x);
            // On all-silent input ORT's FCPE sigmoid underflows to exact zeros and the local-argmax
            // decode yields 0/0 = NaN where torch yields an unvoiced 0; keep the official meaning.
            let mut f0 = self.models.f0(&mag, fr)?;
            for v in f0.iter_mut().filter(|v| !v.is_finite()) {
                *v = 0.0;
            }
            f0
        } else {
            let (mag, fr) = self.stft.rmvpe(x);
            let hidden = self.models.f0(&mag, fr)?;
            dsp::rmvpe_decode(&hidden, fr, 0.03)
        };
        let (pitch, pitchf) = dsp::f0_post(f0, key, d.f0_min, d.f0_max);
        let cl = self.cache_pitch.len();
        let m = pitch.len() - 4;
        self.cache_pitch[cl - m..].copy_from_slice(&pitch[3..pitch.len() - 1]);
        self.cache_pitchf[cl - m..].copy_from_slice(&pitchf[3..pitchf.len() - 1]);
        let pitch_in = self.cache_pitch[cl - p_len..].to_vec();
        let scale = r2 as f32 / r as f32;
        let pitchf_in: Vec<f32> = self.cache_pitchf[cl - p_len..].iter().map(|v| v * scale).collect();

        // feats: nearest x2, truncate to p_len
        let mut feats2 = vec![0f32; p_len * 768];
        for i in 0..p_len {
            let s = (i / 2) * 768;
            feats2[i * 768..(i + 1) * 768].copy_from_slice(&feats[s..s + 768]);
        }

        let rnd_shape = [1usize, 192, p_len - d.flow_head];
        let sine_shape = [1usize, r * self.upp, 1];
        draw(&mut self.rng, &rnd_shape, &mut self.rnd);
        draw(&mut self.rng, &sine_shape, &mut self.sine);
        let mut audio = self.models.generator(&feats2, p_len, &pitch_in, &pitchf_in, &self.rnd, rnd_shape, &self.sine)?;

        if let Some(rs) = &self.rs_formant {
            audio = rs.process(&audio[..r * d.upp_res]);
        }
        let mut infer = match &self.rs_out {
            Some(rs) => rs.process(&audio),
            None => audio,
        };

        // rms mix
        if rms_mix < 1.0 {
            let tail = &self.input_wav[d.extra_frame..];
            let l = infer.len();
            let rms1 = dsp::rms(&tail[..l], 4 * zc, zc);
            let rms1 = dsp::interp_linear_align_corners(&rms1, l + 1);
            let rms2 = dsp::rms(&infer, 4 * zc, zc);
            let rms2 = dsp::interp_linear_align_corners(&rms2, l + 1);
            let e = 1.0 - rms_mix;
            for i in 0..l {
                infer[i] *= (rms1[i] / rms2[i].max(1e-3)).powf(e);
            }
        }

        // SOLA
        let (sb, ss) = (d.sola_buffer_frame, d.sola_search_frame);
        let mut best = (0usize, f32::NEG_INFINITY);
        for k in 0..=ss {
            let (mut nom, mut den) = (0f32, 0f32);
            for i in 0..sb {
                let c = infer[k + i];
                nom += c * self.sola_buffer[i];
                den += c * c;
            }
            let score = nom / (den + 1e-8).sqrt();
            if score > best.1 {
                best = (k, score);
            }
        }
        let infer = &mut infer[best.0..];
        for i in 0..sb {
            infer[i] = infer[i] * self.fade_in[i] + self.sola_buffer[i] * (1.0 - self.fade_in[i]);
        }
        self.sola_buffer.copy_from_slice(&infer[bf..bf + sb]);
        self.out.clear();
        self.out.extend_from_slice(&infer[..bf]);
        Ok(&self.out)
    }
}
