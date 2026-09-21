#![cfg(feature = "onnx")]
use half::f16;
use rvc_deiteris::pitch::{decode, PitchState};
use serde_json::Value;

#[cfg(feature = "evaluation")]
#[test]
fn elapsed_audio_hop_preserves_absolute_pitch_timestamps() {
    let mut aligned = PitchState::new(67).unwrap();
    let mut legacy = PitchState::new(67).unwrap();
    // One unique F0 value per absolute 10ms slot, overlapping estimates every 160ms.
    for end in [26, 42, 58, 74, 90] {
        let raw: Vec<_> = (end - 26..end).map(|slot| 100. + slot as f32).collect();
        aligned.update_aligned(&raw, 0., 0., 16).unwrap();
        legacy.update(&raw, 0., 0.).unwrap();
        let seen = end.min(68);
        let expected: Vec<_> = (end - seen..end)
            .map(|slot| f16::from_f32(100. + slot as f32))
            .collect();
        assert_eq!(&aligned.continuous()[68 - seen..], expected);
        assert_eq!(&aligned.continuous()[42..], &legacy.continuous()[42..]); // newest window unchanged
        if end > 26 {
            assert_ne!(legacy.continuous(), aligned.continuous()); // reproduces old timeline defect
        }
    }
    let mut oracle = PitchState::new(67).unwrap();
    oracle
        .update(
            &(22..90).map(|i| 100. + i as f32).collect::<Vec<_>>(),
            0.,
            0.,
        )
        .unwrap();
    assert_eq!(aligned.coarse(), oracle.coarse());
    assert_eq!(
        aligned.generator_pitch(67, 1.).unwrap().1,
        oracle.generator_pitch(67, 1.).unwrap().1
    );
    aligned.reset();
    assert!(aligned.continuous().iter().all(|v| *v == f16::ZERO));
}

#[cfg(feature = "evaluation")]
#[test]
fn aligned_history_rejects_invalid_hop_or_pitch_without_mutation() {
    let mut state = PitchState::from_cache(vec![20; 4], vec![f16::from_f32(123.); 4]).unwrap();
    for (raw, hop) in [(vec![250.], 0), (vec![250.], 2), (vec![250., f32::NAN], 1)] {
        assert!(state.update_aligned(&raw, 0., 0., hop).is_err());
        assert_eq!(state.coarse(), &[20; 4]);
        assert_eq!(state.continuous(), &[f16::from_f32(123.); 4]);
    }
}

#[cfg(feature = "evaluation")]
#[test]
fn aligned_history_poc_rejects_fractional_hops_before_loading_models() {
    use rvc_deiteris::{startup::Startup, tg::Trial};
    let mut startup = Startup {
        sample_rate: 48000,
        chunk: 60,
        block_frames: None,
        extra_ms: 3000.,
        crossfade_ms: 30.,
        formant: -0.2,
    };
    let trial = Trial {
        aligned_pitch_history: true,
        ..Trial::default()
    };
    assert!(trial.dimensions(&startup, 40000).is_ok());
    startup.chunk = 19;
    assert!(trial.dimensions(&startup, 40000).is_err());
    startup.sample_rate = 44100;
    startup.block_frames = Some(4410);
    assert!(trial.dimensions(&startup, 40000).is_ok());
    startup.block_frames = None;
    assert!(trial.dimensions(&startup, 40000).is_err());
    assert!(Trial::default().dimensions(&startup, 40000).is_ok());
}
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/tg-fast-cpu-v1.json")).unwrap()
}
fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn integers(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}

#[test]
fn decode_preserves_reference_half_denominator_and_voicing_boundary() {
    let data = fixture();
    let tolerance = data["tolerance"]["raw_f0_hz_abs"].as_f64().unwrap() as f32;
    for case in data["decode"].as_array().unwrap() {
        let hidden = floats(&case["salience"])
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let result = decode(&hidden, 0.05).unwrap();
        let expected = floats(&case["raw_f0"]);
        for (actual, expected) in result.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{}: {actual} != {expected}",
                case["name"]
            );
        }
    }
}

#[test]
fn prepare_matches_reference_entire_window_cache_and_half_storage() {
    for case in fixture()["pitch"].as_array().unwrap() {
        let mut state = PitchState::from_cache(
            integers(&case["old_pitch"]),
            floats(&case["old_pitchf"])
                .into_iter()
                .map(f16::from_f32)
                .collect(),
        )
        .unwrap();
        state
            .update(
                &floats(&case["raw_f0"]),
                case["tune"].as_f64().unwrap(),
                case["formant"].as_f64().unwrap(),
            )
            .unwrap();
        assert_eq!(state.coarse(), integers(&case["pitch"]), "{}", case["name"]);
        assert_eq!(
            state.continuous(),
            floats(&case["pitchf"])
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn invalid_input_does_not_partially_advance_the_pitch_state() {
    let mut state = PitchState::from_cache(vec![20; 4], vec![f16::from_f32(123.); 4]).unwrap();
    assert!(state.update(&[250., f32::NAN], 0., 0.).is_err());
    assert_eq!(state.coarse(), &[20; 4]);
    assert!(decode(&[f16::NAN; 360], 0.05).is_err());
}

#[test]
fn actual_decode_is_connected_to_cache_and_generator_inputs() {
    let data = fixture();
    let raw: Vec<f32> = data["decode"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|case| {
            let hidden: Vec<_> = floats(&case["salience"])
                .into_iter()
                .map(f16::from_f32)
                .collect();
            decode(&hidden, 0.05).unwrap()
        })
        .collect();
    for case in data["pitch"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["name"] == "decoded")
    {
        let mut state = PitchState::from_cache(
            integers(&case["old_pitch"]),
            floats(&case["old_pitchf"])
                .into_iter()
                .map(f16::from_f32)
                .collect(),
        )
        .unwrap();
        state
            .update(
                &raw,
                case["tune"].as_f64().unwrap(),
                case["formant"].as_f64().unwrap(),
            )
            .unwrap();
        let expected = integers(&case["pitch"]);
        let expected_f: Vec<_> = floats(&case["pitchf"])
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let (p, pf) = state.generator_pitch(11, 1.0).unwrap();
        assert_eq!(p, &expected[1..]);
        assert_eq!(pf, expected_f[1..]);
    }
}

#[test]
fn empty_and_oversized_estimation_windows_follow_circular_write_contract() {
    let mut state = PitchState::new(4).unwrap();
    state
        .update(&[0., 100., 200., 240., 250., 300., 440., 800.], 0., 0.)
        .unwrap();
    assert_eq!(state.coarse(), &[67, 70, 84, 122, 202]);
    let expected = [240., 250., 300., 440., 800.].map(f16::from_f32);
    assert_eq!(state.continuous(), &expected);
    state.update(&[], 0., 0.).unwrap();
    assert_eq!(state.continuous(), &expected);
    let (coarse, pitchf) = state.generator_pitch(4, 1.5).unwrap();
    assert_eq!(coarse, &[70, 84, 122, 202]);
    assert_eq!(pitchf, [375., 450., 660., 1200.].map(f16::from_f32));
    state.reset();
    assert_eq!(state.coarse(), &[0; 5]);
    assert_eq!(state.continuous(), &[f16::ZERO; 5]);
}

#[test]
fn successive_tagged_windows_match_reference_whole_cache_and_generator_slice() {
    let mut state = PitchState::new(67).unwrap();
    for case in fixture()["pitch_sequence"].as_array().unwrap() {
        state.update(&floats(&case["raw_f0"]), 0.0, 0.0).unwrap();
        assert_eq!(state.coarse(), integers(&case["pitch"]));
        assert_eq!(
            state.continuous(),
            floats(&case["pitchf"])
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>()
        );
        let (coarse, continuous) = state.generator_pitch(67, 1.25).unwrap();
        assert_eq!(coarse, &integers(&case["pitch"])[1..]);
        assert_eq!(
            continuous,
            floats(&case["generator_pitchf"])
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>()
        );
    }
}
