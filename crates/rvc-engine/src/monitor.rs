//! The hand-off from the conversion callback to the monitor stream.
//!
//! The two run on different devices with their own clocks, so their callbacks never line up: one asks
//! for samples while the other has not produced them yet, or produces two blocks between two asks.
//! A queue of samples absorbs that. It is written by one thread and read by one thread, so the indices
//! are plain atomics; the delay it may add is bounded by `target`, and what it cannot deliver it fades
//! out instead of cutting to digital silence.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub(crate) struct MonitorBuffer {
    data: UnsafeCell<Vec<f32>>,
    /// total samples written and read; the difference is what is stored
    written: AtomicUsize,
    taken: AtomicUsize,
    /// samples kept at most after a write, so the monitor never falls behind the conversion
    target: usize,
    /// samples the reader had to invent, and samples dropped to keep the delay bounded
    missing: AtomicU64,
    dropped: AtomicU64,
    /// last sample handed out, so an underrun fades from it instead of stepping to zero
    last: UnsafeCell<f32>,
}

// SAFETY: one conversion callback writes, one monitor callback reads; the indices order the two.
unsafe impl Sync for MonitorBuffer {}
unsafe impl Send for MonitorBuffer {}

impl MonitorBuffer {
    /// `block` samples per conversion callback; the queue holds a few of them.
    pub(crate) fn new(block: usize) -> Self {
        let target = 2 * block;
        Self {
            data: UnsafeCell::new(vec![0.0; 4 * block.max(1)]),
            written: AtomicUsize::new(0),
            taken: AtomicUsize::new(0),
            target,
            missing: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            last: UnsafeCell::new(0.0),
        }
    }

    fn capacity(&self) -> usize {
        unsafe { (*self.data.get()).len() }
    }

    /// Samples waiting to be played.
    #[cfg(test)]
    pub(crate) fn stored(&self) -> usize {
        self.written.load(Ordering::Acquire).saturating_sub(self.taken.load(Ordering::Acquire))
    }

    /// (samples the monitor had to invent, samples dropped because it fell behind)
    pub(crate) fn counts(&self) -> (u64, u64) {
        (self.missing.load(Ordering::Relaxed), self.dropped.load(Ordering::Relaxed))
    }

    /// Conversion side: append one converted block.
    pub(crate) fn push(&self, block: &[f32]) {
        let cap = self.capacity();
        let data = unsafe { &mut *self.data.get() };
        let mut written = self.written.load(Ordering::Relaxed);
        for (i, &v) in block.iter().enumerate() {
            data[(written + i) % cap] = v;
        }
        written += block.len();
        self.written.store(written, Ordering::Release);
        // keep the monitor close to the conversion: drop whatever is older than `target`
        let taken = self.taken.load(Ordering::Acquire);
        let stored = written - taken;
        if stored > self.target {
            let drop = stored - self.target;
            self.taken.store(taken + drop, Ordering::Release);
            self.dropped.fetch_add(drop as u64, Ordering::Relaxed);
        }
    }

    /// Monitor side: fill `out` (one value per frame) at `volume`. Returns false when the queue ran dry.
    pub(crate) fn read(&self, out: &mut [f32], volume: f32) -> bool {
        let cap = self.capacity();
        let data = unsafe { &*self.data.get() };
        let last = unsafe { &mut *self.last.get() };
        let written = self.written.load(Ordering::Acquire);
        let mut taken = self.taken.load(Ordering::Relaxed);
        let have = written.saturating_sub(taken).min(out.len());
        for (i, o) in out[..have].iter_mut().enumerate() {
            *o = data[(taken + i) % cap] * volume;
        }
        taken += have;
        self.taken.store(taken, Ordering::Release);
        if have > 0 {
            *last = out[have - 1];
        }
        if have == out.len() {
            return true;
        }
        // nothing left: fade from the last sample so the gap is not a step
        let rest = out.len() - have;
        self.missing.fetch_add(rest as u64, Ordering::Relaxed);
        let start = *last;
        let fade = rest.min(FADE);
        for (i, o) in out[have..].iter_mut().enumerate() {
            *o = if i < fade { start * (1.0 - (i + 1) as f32 / fade as f32) } else { 0.0 };
        }
        *last = 0.0;
        false
    }
}

/// Samples an underrun fades over (about 2 ms at 48 kHz).
const FADE: usize = 96;

#[cfg(test)]
mod tests {
    use super::*;

    /// The two sides run at the same average rate but never line up: the monitor asks in its own chunk
    /// size and sometimes before, sometimes after the conversion produced its block. Nothing may be lost.
    #[test]
    fn jitter_does_not_break_the_stream() {
        let block = 2880;
        let buf = MonitorBuffer::new(block);
        let mut next_in = 0f32;
        let mut expect = 0f32;
        let mut push = |from: &mut f32| {
            let b: Vec<f32> = (0..block).map(|i| *from + i as f32).collect();
            *from += block as f32;
            buf.push(&b);
        };
        push(&mut next_in); // one block of head start, as the stream has when it opens
        for round in 0..60 {
            // the monitor asks for one block in three chunks, before or after the conversion callback
            let chunks = [1024usize, 1024, 832];
            if round % 2 == 0 {
                push(&mut next_in);
            }
            for n in chunks {
                let mut out = vec![0.0; n];
                assert!(buf.read(&mut out, 1.0), "round {round}: ran dry");
                for v in &out {
                    assert_eq!(*v, expect, "round {round}");
                    expect += 1.0;
                }
            }
            if round % 2 == 1 {
                push(&mut next_in);
            }
        }
        assert_eq!(buf.counts(), (0, 0));
    }

    /// The queue never grows into a long delay: what the monitor plays stays close to the newest block.
    #[test]
    fn delay_stays_bounded() {
        let block = 2880;
        let buf = MonitorBuffer::new(block);
        for i in 0..100 {
            buf.push(&vec![i as f32; block]);
        }
        assert!(buf.stored() <= 2 * block, "{}", buf.stored());
        let mut out = vec![0.0; block];
        buf.read(&mut out, 1.0);
        assert!(out[0] >= 98.0, "played {} while 99 was the newest", out[0]);
        let (_, dropped) = buf.counts();
        assert!(dropped > 0);
    }

    /// An empty queue fades out instead of stepping to zero, and the volume applies.
    #[test]
    fn underrun_fades_out() {
        let block = 480;
        let buf = MonitorBuffer::new(block);
        buf.push(&vec![1.0; block]);
        let mut out = vec![0.0; block + 200];
        assert!(!buf.read(&mut out, 0.5));
        assert_eq!(out[block - 1], 0.5);
        assert!(out[block] < 0.5 && out[block] > 0.0, "{}", out[block]);
        assert_eq!(out[block + 150], 0.0);
        let (missing, _) = buf.counts();
        assert_eq!(missing, 200);
    }
}
