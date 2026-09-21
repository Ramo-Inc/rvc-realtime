use rvc_deiteris::input::InputTimeline;
use serde_json::Value;

#[test]
fn input_grid_counts_and_boundary_waveform_match_streaming_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/tg-fast-cpu-v1.json")).unwrap();
    for case in fixture["resample"].as_array().unwrap() {
        let rate = case["source_rate"].as_u64().unwrap() as usize;
        let block = case["block"].as_u64().unwrap() as usize;
        let mut input = InputTimeline::new(rate, block).unwrap();
        for (index, count) in case["counts"].as_array().unwrap().iter().enumerate() {
            let audio: Vec<_> = (index * block..(index + 1) * block)
                .map(|i| {
                    let time = i as f32 / rate as f32;
                    ((2.0 * std::f32::consts::PI * 997.0) * time).sin()
                        + 0.05 * ((2.0 * std::f32::consts::PI * 6013.0) * time).sin()
                })
                .collect();
            let actual = input.push(&audio).unwrap();
            assert_eq!(actual.len(), count.as_u64().unwrap() as usize);
            if index < 3 {
                for (i, value) in case["first20"][index]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    assert!(
                        (actual[i] - value.as_f64().unwrap() as f32).abs() <= 0.0001,
                        "rate {rate} block {block} index {index} sample {i}: {} vs {value}",
                        actual[i]
                    );
                }
            }
        }
        assert_eq!(input.total_output(), case["total_output"].as_u64().unwrap());
        input.reset();
        assert_eq!(input.total_output(), 0);
        assert!(input
            .push(&vec![0.0; block])
            .unwrap()
            .iter()
            .all(|v| *v == 0.0));
    }
}

#[cfg(feature = "evaluation")]
#[test]
fn delayed_input_is_independent_of_block_partition_and_resets() {
    // Equal source audio split into 50ms-ish and 160ms-ish blocks, with tones
    // and sparse impulses. No offline output from this implementation is used
    // as an oracle: partitioning alone must not introduce edges in the signal.
    for rate in [48000, 44100] {
        let input: Vec<_> = (0..145920).map(|i| {
            let tone = (std::f64::consts::TAU * 173. * i as f64 / rate as f64).sin() as f32 * 0.2;
            tone + if i % 12997 == 0 { 0.5 } else { 0. }
        }).collect();
        let run = |block, delayed| {
            let mut resampler = if delayed { InputTimeline::new_delayed(rate, block).unwrap() }
                else { InputTimeline::new(rate, block).unwrap() };
            let mut out = Vec::new();
            for chunk in input.chunks_exact(block) { out.extend_from_slice(resampler.push(chunk).unwrap()); }
            assert_eq!(out.len(), (input.len() * 16000).div_ceil(rate));
            assert_eq!(resampler.total_output(), out.len() as u64);
            resampler.reset();
            assert_eq!(resampler.total_output(), 0);
            let replay = resampler.push(&input[..block]).unwrap();
            assert_eq!(replay, &out[..replay.len()]);
            out
        };
        let max_delta = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a,b)|(a-b).abs()).fold(0f32, f32::max);
        let short = run(2432, true);
        let long = run(7680, true);
        assert!(short.iter().any(|v| v.abs() > 0.1));
        assert!(max_delta(&short, &long) < 1e-6);
        // Keep the reproduction explicit: the current undelayed reference path
        // remains different at block boundaries, and isn't silently replaced.
        assert!(max_delta(&run(2432, false), &run(7680, false)) > 0.01);
    }
}
