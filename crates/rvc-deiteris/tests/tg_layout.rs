use rvc_deiteris::startup::Startup;

#[cfg(feature = "onnx")]
#[test]
fn full_pitch_context_changes_analysis_only() {
    use rvc_deiteris::tg::Trial;
    for rate in [44100, 48000] {
        let startup = Startup { sample_rate:rate, chunk:60, block_frames:None,
            extra_ms:3000., crossfade_ms:30., formant:-0.2 };
        let normal = Trial::default().dimensions(&startup, 40000).unwrap();
        let full = Trial { full_pitch_context:true, ..Trial::default() }.dimensions(&startup, 40000).unwrap();
        assert!(normal.silence_front > 0);
        assert_eq!(full.silence_front, 0);
        let mut expected = serde_json::to_value(normal).unwrap();
        expected["silence_front"] = 0.into();
        assert_eq!(expected, serde_json::to_value(full).unwrap());
    }
}

#[test]
fn minimum_pitch_window_extends_context_not_returned_audio() {
    let startup = Startup {sample_rate: 48000, chunk: 19, block_frames: None,
        extra_ms: 0., crossfade_ms: 20., formant: 0.};
    let dims = startup.tg_dims(40000).unwrap();
    assert_eq!((dims.block,dims.convert,dims.features,dims.skip,dims.ret,dims.silence_front), (2432,3040,19,10,9,0));
    let long = Startup {crossfade_ms: 160., ..startup};
    let dims = long.tg_dims(40000).unwrap();
    assert_eq!((dims.convert,dims.skip,dims.ret,dims.silence_front), (3680,0,23,0));
}

#[test]
fn supported_layouts_keep_generation_and_estimation_ranges_valid() {
    for rate in [44100,48000] {
        for model_rate in [32000,40000,48000] {
            for extra in [0.,50.,500.,1000.,5000.] {
                for crossfade in [20.,40.,50.,70.,100.] {
                    for chunk in [19,23,24,60] {
                        for formant in [-12.,-0.2,0.,12.] {
                            let startup = Startup {sample_rate:rate,chunk,block_frames:None,extra_ms:extra,crossfade_ms:crossfade,formant};
                            let dims = startup.tg_dims(model_rate).unwrap();
                            assert!(dims.convert-dims.silence_front>=3040);
                            assert_eq!(dims.features, dims.skip+dims.ret);
                            assert!(dims.generated>=dims.trimmed);
                        }
                    }
                }
            }
        }
    }
}
