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
    Legacy { model_dir: PathBuf, startup: Startup },
    #[cfg(feature = "deiteris")]
    Deiteris {
        model_dir: PathBuf,
        assets: PathBuf,
        startup: rvc_deiteris::startup::Startup,
        cuda_graph: bool,
    },
}
impl Conversion {
    pub(crate) fn sample_rate(&self) -> u32 {
        match self {
            Self::Legacy { startup, .. } => startup.sample_rate,
            #[cfg(feature = "deiteris")]
            Self::Deiteris { startup, .. } => startup.sample_rate as u32,
        }
    }
    pub fn load(self, options: &EngineOptions) -> Result<Processor> {
        match self {
            Self::Legacy { model_dir, startup } => {
                let model = Model::open(&model_dir)?;
                Ok(Processor(Backend::Legacy(Engine::new(&model, &startup, options)?)))
            }
            #[cfg(feature = "deiteris")]
            Self::Deiteris { model_dir: model, assets, startup, cuda_graph } => {
                crate::infer::prepend_path(&options.runtime_dir);
                let seed = options.seed;
                Ok(Processor(Backend::Deiteris(crate::owner::OwnedProcessor::start(move || {
                    let metadata: serde_json::Value = serde_json::from_slice(
                        &std::fs::read(model.join("model.json")).map_err(|e| Error::Runtime(e.to_string()))?
                    ).map_err(|e| Error::Runtime(e.to_string()))?;
                    let rate = metadata["model_sr"].as_u64()
                        .ok_or_else(|| Error::Runtime("missing voice sample rate".into()))? as usize;
                    let assets = rvc_deiteris::engine::Assets {
                        pre: assets.join("pre.onnx"), prepare: assets.join("prepare.onnx"),
                        post: assets.join("post.onnx"), contentvec: assets.join("contentvec.onnx"),
                        rmvpe: assets.join("rmvpe.onnx"),
                        template: model.join("generator.onnx"),
                        model_rate: rate,
                    };
                    let mut engine = rvc_deiteris::engine::Engine::new_with_graph(&model, startup, &assets, cuda_graph)
                        .map_err(|e| Error::Runtime(e.to_string()))?;
                    engine.seed(seed).map_err(|e| Error::Runtime(e.to_string()))?;
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
