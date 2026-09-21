//! Audio-device-independent conversion interface used by Realtime and headless validation.
use std::path::PathBuf;
use crate::{Engine, EngineOptions, Model, Params, Startup};
use crate::error::Result;
#[cfg(feature = "deiteris")]
use crate::error::Error;
pub struct Processor(Backend);
enum Backend {
    Legacy(Engine),
    #[cfg(feature = "deiteris")]
    Deiteris(crate::owner::OwnedProcessor),
}
impl Processor {
    pub fn block_frames(&self) -> usize {
        match &self.0 {
            Backend::Legacy(e) => e.block_frames(),
            #[cfg(feature = "deiteris")]
            Backend::Deiteris(e) => e.block_frames(),
        }
    }
    pub fn process(&mut self, block: &[f32], params: &Params) -> Result<&[f32]> {
        match &mut self.0 {
            Backend::Legacy(e) => e.process(block, params),
            #[cfg(feature = "deiteris")]
            Backend::Deiteris(e) => e.process(block, params),
        }
    }
}

/// Which conversion path to start. Deiteris has a distinct versioned ONNX cache.
pub enum Conversion {
    #[cfg(feature = "quality-evaluation")]
    DeiterisTrial {
        model_dir: PathBuf,
        assets: PathBuf,
        startup: rvc_deiteris::startup::Startup,
        cuda_graph: bool,
        trial: rvc_deiteris::tg::Trial,
    },
    Legacy { model_dir: PathBuf, startup: Startup },
    #[cfg(feature = "deiteris")]
    Deiteris {
        model_dir: PathBuf,
        assets: PathBuf,
        startup: rvc_deiteris::startup::Startup,
        cuda_graph: bool,
    },
    /// Comparison candidate only; never selected by the application.
    #[cfg(feature = "quality-evaluation")]
    DeiterisGpu {
        model_dir: PathBuf,
        assets: PathBuf,
        startup: rvc_deiteris::startup::Startup,
        cuda_graph: bool,
    },
}
#[cfg(feature = "deiteris")]
fn tg_runtime(assets: &std::path::Path, options: &EngineOptions) -> Result<PathBuf> {
    let gpu = assets.join("tg-gpu-pitch-v2");
    if !gpu.join("contract.json").is_file() {
        return Err(Error::Runtime(format!("missing Fast V2 GPU assets: {}", gpu.display())));
    }
    let runtime = options.runtime_dir.canonicalize().map_err(|e| Error::Runtime(e.to_string()))?;
    crate::infer::prepend_path(&runtime);
    Ok(runtime)
}
impl Conversion {
    pub(crate) fn sample_rate(&self) -> u32 {
        match self {
            Self::Legacy { startup, .. } => startup.sample_rate,
            #[cfg(feature = "deiteris")]
            Self::Deiteris { startup, .. } => startup.sample_rate as u32,
            #[cfg(feature = "quality-evaluation")]
            Self::DeiterisGpu { startup, .. } => startup.sample_rate as u32,
            #[cfg(feature = "quality-evaluation")]
            Self::DeiterisTrial { startup, .. } => startup.sample_rate as u32,
        }
    }
    pub fn load(self, options: &EngineOptions) -> Result<Processor> {
        match self {
            #[cfg(feature = "quality-evaluation")]
            Self::DeiterisTrial { model_dir, assets, startup, cuda_graph, trial } => {
                let runtime = tg_runtime(&assets, options)?;
                let seed = options.seed;
                Ok(Processor(Backend::Deiteris(crate::owner::OwnedProcessor::start(move || {
                    let mut engine = rvc_deiteris::tg::Engine::new_trial(
                        &model_dir, &assets.join("tg-fast-v1"), &assets.join("tg-gpu-pitch-v2"),
                        &runtime, startup, cuda_graph, trial
                    ).map_err(|e| Error::Runtime(e.to_string()))?;
                    engine.prepare(seed).map_err(|e| Error::Runtime(e.to_string()))?;
                    Ok((engine.block_frames(), move |block: &[f32], params: &Params, output: &mut Vec<f32>| {
                        let converted = engine.process(block, &rvc_deiteris::engine::Params {
                            pitch: params.pitch as f64, threshold_db: params.threshold_db as f64,
                        }).map_err(|e| Error::Inference(e.to_string()))?;
                        output.clear();
                        output.extend_from_slice(converted);
                        Ok(())
                    }))
                })?)))
            }
            #[cfg(feature = "quality-evaluation")]
            Self::DeiterisGpu { model_dir, assets, startup, cuda_graph } => {
                let pipeline = assets.join("tg-full-gpu-pipeline-v1");
                if !pipeline.join("contract.json").is_file() {
                    return Err(Error::Runtime(format!("missing experimental full-GPU assets: {}", pipeline.display())));
                }
                let runtime = tg_runtime(&assets, options)?;
                let seed = options.seed;
                Ok(Processor(Backend::Deiteris(crate::owner::OwnedProcessor::start(move || {
                    let mut engine = rvc_deiteris::gpu_pipeline::GpuPipeline::new(
                        &model_dir, &assets, &pipeline, &runtime, startup, cuda_graph
                    ).map_err(|e| Error::Runtime(e.to_string()))?;
                    engine.prepare(seed).map_err(|e| Error::Runtime(e.to_string()))?;
                    Ok((engine.block_frames(), move |block: &[f32], params: &Params, output: &mut Vec<f32>| {
                        let converted = engine.process(block, &rvc_deiteris::engine::Params {
                            pitch: params.pitch as f64, threshold_db: params.threshold_db as f64,
                        }).map_err(|e| Error::Inference(e.to_string()))?;
                        output.clear();
                        output.extend_from_slice(converted);
                        Ok(())
                    }))
                })?)))
            }
            Self::Legacy { model_dir, startup } => {
                let model = Model::open(&model_dir)?;
                Ok(Processor(Backend::Legacy(Engine::new(&model, &startup, options)?)))
            }
            #[cfg(feature = "deiteris")]
            Self::Deiteris { model_dir: model, assets, startup, cuda_graph } => {
                let runtime = tg_runtime(&assets, options)?;
                let seed = options.seed;
                Ok(Processor(Backend::Deiteris(crate::owner::OwnedProcessor::start(move || {
                    let mut engine = rvc_deiteris::tg::Engine::new_fast_v2(
                        &model, &assets.join("tg-fast-v1"), &assets.join("tg-gpu-pitch-v2"),
                        &runtime, startup, cuda_graph
                    ).map_err(|e| Error::Runtime(e.to_string()))?;
                    engine.prepare(seed).map_err(|e| Error::Runtime(e.to_string()))?;
                    Ok((engine.block_frames(), move |block: &[f32], params: &Params, output: &mut Vec<f32>| {
                        let converted = engine.process(block, &rvc_deiteris::engine::Params {
                            pitch: params.pitch as f64, threshold_db: params.threshold_db as f64,
                        }).map_err(|e| Error::Inference(e.to_string()))?;
                        output.clear();
                        output.extend_from_slice(converted);
                        Ok(())
                    }))
                })?)))
            }
        }
    }
}

#[cfg(all(test, feature = "deiteris"))]
mod tests {
    use super::*;

    #[test]
    fn missing_quality_bundle_is_reported_before_loading_runtime() {
        let absent = std::env::temp_dir().join(format!("rvc-quality-absent-{}", std::process::id()));
        assert!(!absent.exists());
        let startup = rvc_deiteris::startup::Startup {
            sample_rate: 48000, chunk: 60, block_frames: None,
            extra_ms: 3000., crossfade_ms: 30., formant: -0.2,
        };
        let options = EngineOptions { runtime_dir: absent.clone(), seed: 1 };
        let result = Conversion::Deiteris {
            model_dir: absent.clone(), assets: absent.clone(), startup: startup.clone(), cuda_graph: true,
        }.load(&options);
        assert!(result.err().unwrap().to_string().contains("missing Fast V2 GPU assets"));
        #[cfg(feature = "quality-evaluation")]
        {
            let result = Conversion::DeiterisGpu {
                model_dir: absent.clone(), assets: absent, startup, cuda_graph: true,
            }.load(&options);
            assert!(result.err().unwrap().to_string().contains("missing experimental full-GPU assets"));
        }
    }
}
