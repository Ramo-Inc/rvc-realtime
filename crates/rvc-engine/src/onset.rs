//! Passive onset measurement at the application's input/output boundary.
//! No audio modification, allocations in observe(), device timestamps, or estimated delays.
use std::time::Duration;

#[derive(Clone, Copy)]
struct Edge {
    window: u64,
    block: u64,
    time: Duration,
}

/// One detected utterance, paired with the first output block containing sound.
/// IDs refer to calls to observe(), not a count of how many times the user spoke.
#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub input_block: u64,
    pub output_block: u64,
    pub input_received: Duration,
    pub output_written: Duration,
}

impl Measurement {
    pub fn elapsed_ms(self) -> f32 {
        (self.output_written - self.input_received).as_secs_f32() * 1000.0
    }

    /// One atomic word keeps the displayed duration and block distance consistent.
    pub(crate) fn packed(self) -> u64 {
        ((self.output_block - self.input_block).min(u32::MAX as u64) << 32)
            | self.elapsed_ms().to_bits() as u64
    }
}

/// Detect a new utterance after 300 ms of quiet on BOTH input and output.
/// Measures from delivery of its input block to readiness of the first output
/// block containing sound. This excludes input-device collection before delivery
/// and playback after delivery, and has block-level timing resolution.
pub struct OnsetLatency {
    window_frames: usize,
    quiet_windows: usize,
    attack_windows: usize,
    count: usize,
    input_energy: f64,
    output_energy: f64,
    window: u64,
    next_block: u64,
    quiet: usize,
    armed: bool,
    input_run: usize,
    output_run: usize,
    input_edge: Option<Edge>,
    output_edge: Option<Edge>,
    pending: Option<Edge>,
}

impl OnsetLatency {
    pub fn new(sample_rate: u32) -> Self {
        let window_frames = (sample_rate as usize / 1000).max(1);
        Self {
            window_frames,
            quiet_windows: (sample_rate as usize * 300).div_ceil(1000 * window_frames),
            attack_windows: (sample_rate as usize * 5)
                .div_ceil(1000 * window_frames)
                .max(1),
            count: 0,
            input_energy: 0.0,
            output_energy: 0.0,
            window: 0,
            next_block: 0,
            quiet: 0,
            armed: false,
            input_run: 0,
            output_run: 0,
            input_edge: None,
            output_edge: None,
            pending: None,
        }
    }

    /// Cancel an ambiguous measurement on dropped/missing samples.
    pub fn reset(&mut self) {
        self.count = 0;
        self.input_energy = 0.0;
        self.output_energy = 0.0;
        self.quiet = 0;
        self.armed = false;
        self.input_run = 0;
        self.output_run = 0;
        self.input_edge = None;
        self.output_edge = None;
        self.pending = None;
    }

    /// Returns only a NEW measurement. Call after conversion, with timestamps
    /// taken before processing input and when output became available. Both must
    /// use the same monotonic clock (not wall/calendar time).
    pub fn observe(
        &mut self,
        input: &[f32],
        output: &[f32],
        input_seen: Duration,
        output_ready: Duration,
    ) -> Option<Measurement> {
        let block = self.next_block;
        self.next_block += 1;
        if input.len() != output.len() || output_ready < input_seen {
            self.reset();
            return None;
        }
        let mut measured = None;
        for (&i, &o) in input.iter().zip(output) {
            if !i.is_finite() || !o.is_finite() {
                self.reset();
                return None;
            }
            self.input_energy += (i as f64).powi(2);
            self.output_energy += (o as f64).powi(2);
            self.count += 1;
            if self.count != self.window_frames {
                continue;
            }
            // -45 dBFS RMS; 5 ms persistence rejects isolated clicks.
            const POWER: f64 = 0.000031622776601683795;
            let input_active = self.input_energy / self.count as f64 >= POWER;
            let output_active = self.output_energy / self.count as f64 >= POWER;
            self.input_energy = 0.0;
            self.output_energy = 0.0;
            self.count = 0;
            self.window += 1;

            if !input_active && !output_active {
                self.quiet = (self.quiet + 1).min(self.quiet_windows);
                if self.quiet == self.quiet_windows && self.pending.is_none() {
                    self.armed = true;
                }
            } else {
                self.quiet = 0;
            }

            if input_active {
                if self.input_run == 0 {
                    self.input_edge = Some(Edge {
                        window: self.window,
                        block,
                        time: input_seen,
                    });
                }
                self.input_run = (self.input_run + 1).min(self.attack_windows);
                if self.armed && self.input_run == self.attack_windows {
                    self.pending = self.input_edge;
                    self.armed = false;
                }
            } else {
                self.input_run = 0;
                self.input_edge = None;
            }

            if output_active {
                if self.output_run == 0 {
                    self.output_edge = Some(Edge {
                        window: self.window,
                        block,
                        time: output_ready,
                    });
                }
                self.output_run = (self.output_run + 1).min(self.attack_windows);
                if self.output_run == self.attack_windows {
                    self.armed = false;
                    if let (Some(input), Some(output)) = (self.pending.take(), self.output_edge) {
                        if output.window >= input.window {
                            if output
                                .time
                                .checked_sub(input.time)
                                .filter(|d| *d <= Duration::from_secs(3))
                                .is_some()
                            {
                                measured = Some(Measurement {
                                    input_block: input.block,
                                    output_block: output.block,
                                    input_received: input.time,
                                    output_written: output.time,
                                });
                            }
                        }
                    }
                }
            } else {
                self.output_run = 0;
                self.output_edge = None;
            }
            if self
                .pending
                .is_some_and(|edge| output_ready.saturating_sub(edge.time) > Duration::from_secs(3))
            {
                self.pending = None;
            }
        }
        measured
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn observe(
        d: &mut OnsetLatency,
        input: &[f32],
        output: &[f32],
        start: u64,
        end: u64,
    ) -> Option<f32> {
        d.observe(
            input,
            output,
            Duration::from_millis(start),
            Duration::from_millis(end),
        )
        .map(Measurement::elapsed_ms)
    }
    fn armed() -> OnsetLatency {
        let mut d = OnsetLatency::new(1000);
        assert!(observe(&mut d, &[0.0; 300], &[0.0; 300], 0, 20).is_none());
        d
    }
    #[test]
    fn measures_real_elapsed_time_across_blocks_and_updates_on_next_utterance() {
        let mut d = armed();
        assert!(observe(&mut d, &[0.1; 50], &[0.0; 50], 300, 340).is_none());
        assert!(observe(&mut d, &[0.1; 50], &[0.0; 50], 350, 390).is_none());
        assert_eq!(
            observe(&mut d, &[0.1; 50], &[0.1; 50], 400, 440),
            Some(140.0)
        );
        assert!(observe(&mut d, &[0.1; 50], &[0.1; 50], 450, 490).is_none());
        observe(&mut d, &[0.0; 300], &[0.0; 300], 500, 520);
        assert_eq!(
            observe(&mut d, &[0.1; 50], &[0.1; 50], 800, 825),
            Some(25.0)
        );
    }
    #[test]
    fn refuses_startup_sound_background_noise_and_old_output_tail() {
        let mut d = OnsetLatency::new(1000);
        assert!(observe(&mut d, &[0.1; 50], &[0.1; 50], 0, 20).is_none());
        assert!(observe(&mut d, &[0.001; 300], &[0.1; 300], 50, 70).is_none());
        assert!(observe(&mut d, &[0.1; 50], &[0.1; 50], 350, 370).is_none());
        observe(&mut d, &[0.001; 300], &[0.001; 300], 400, 420);
        assert_eq!(
            observe(&mut d, &[0.1; 50], &[0.1; 50], 700, 730),
            Some(30.0)
        );
    }
    #[test]
    fn rejects_clicks_and_output_that_precedes_input_in_the_same_block() {
        let mut d = armed();
        assert!(observe(&mut d, &[0.1; 1], &[0.1; 1], 300, 320).is_none());
        observe(&mut d, &[0.0; 300], &[0.0; 300], 301, 321);
        let mut input = [0.0; 50];
        input[20..].fill(0.1);
        assert!(observe(&mut d, &input, &[0.1; 50], 601, 621).is_none());
    }
    #[test]
    fn confirmation_across_a_block_boundary_uses_first_output_ready_time() {
        let mut d = armed();
        observe(&mut d, &[0.1; 50], &[0.0; 50], 300, 330);
        observe(&mut d, &[0.1; 2], &[0.1; 2], 350, 380);
        assert_eq!(
            observe(&mut d, &[0.1; 10], &[0.1; 10], 352, 400),
            Some(80.0)
        );
    }
    #[test]
    fn missing_output_times_out_and_discontinuity_requires_new_silence() {
        let mut d = armed();
        observe(&mut d, &[0.1; 50], &[0.0; 50], 300, 330);
        assert!(observe(&mut d, &[0.1; 50], &[0.1; 50], 3400, 3430).is_none());
        d.reset();
        assert!(observe(&mut d, &[0.1; 50], &[0.1; 50], 3500, 3530).is_none());
    }
    #[test]
    fn short_utterance_can_wait_through_silence_for_a_delayed_output() {
        let mut d = armed();
        observe(&mut d, &[0.1; 10], &[0.0; 10], 300, 330);
        observe(&mut d, &[0.0; 400], &[0.0; 400], 310, 340);
        assert_eq!(
            observe(&mut d, &[0.0; 50], &[0.1; 50], 710, 750),
            Some(450.0)
        );
    }
    #[test]
    fn detector_accepts_48k_and_44100_and_non_ms_blocks() {
        for sr in [48000, 44100] {
            let mut d = OnsetLatency::new(sr);
            let quiet = vec![0.0; sr as usize / 2];
            observe(&mut d, &quiet, &quiet, 0, 20);
            let sound = vec![0.1; 2176];
            assert_eq!(observe(&mut d, &sound, &sound, 500, 540), Some(40.0));
        }
    }

    #[test]
    fn tracks_the_utterance_block_until_its_output_not_the_immediate_return() {
        let mut d = armed(); // block 0
        assert!(d
            .observe(
                &[0.1; 50],
                &[0.0; 50],
                Duration::from_millis(300),
                Duration::from_millis(320)
            )
            .is_none());
        let m = d
            .observe(
                &[0.1; 50],
                &[0.1; 50],
                Duration::from_millis(350),
                Duration::from_millis(370),
            )
            .unwrap();
        assert_eq!((m.input_block, m.output_block), (1, 2));
        assert_eq!(m.input_received, Duration::from_millis(300));
        assert_eq!(m.output_written, Duration::from_millis(370));
        assert_eq!(m.elapsed_ms(), 70.0); // 50 ms arrival wait + 20 ms processing
        assert_eq!(m.packed() >> 32, 1);
        assert_eq!(f32::from_bits(m.packed() as u32), 70.0);
    }

    #[test]
    fn records_first_output_block_even_when_confirmation_happens_later() {
        let mut d = armed();
        d.observe(
            &[0.1; 50],
            &[0.0; 50],
            Duration::from_millis(300),
            Duration::from_millis(320),
        );
        d.observe(
            &[0.1; 2],
            &[0.1; 2],
            Duration::from_millis(350),
            Duration::from_millis(370),
        );
        let m = d
            .observe(
                &[0.1; 10],
                &[0.1; 10],
                Duration::from_millis(400),
                Duration::from_millis(420),
            )
            .unwrap();
        assert_eq!((m.input_block, m.output_block), (1, 2));
        assert_eq!(m.elapsed_ms(), 70.0);
    }
}
