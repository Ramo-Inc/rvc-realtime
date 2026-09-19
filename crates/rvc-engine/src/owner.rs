//! Synchronous block handoff to an engine that must never leave its owning thread.
use crate::{Error, Params};
use crate::error::Result;
use std::{sync::mpsc::{self, Receiver, SyncSender}, thread::JoinHandle};

type Response = Result<(Vec<f32>, Vec<f32>)>;
pub(crate) struct OwnedProcessor {
    sender: Option<SyncSender<(Vec<f32>, Vec<f32>, Params)>>,
    receiver: Receiver<Response>,
    thread: Option<JoinHandle<()>>,
    frames: usize,
    input: Vec<f32>,
    output: Vec<f32>,
    failed: bool,
}
impl OwnedProcessor {
    pub fn start<F, P>(factory: F) -> Result<Self>
    where F: FnOnce() -> Result<(usize, P)> + Send + 'static,
          P: FnMut(&[f32], &Params, &mut Vec<f32>) -> Result<()> + 'static {
        let (sender, requests) = mpsc::sync_channel::<(Vec<f32>, Vec<f32>, Params)>(1);
        let (responses, receiver) = mpsc::sync_channel(1);
        let (ready, initialized) = mpsc::sync_channel(1);
        let thread = std::thread::spawn(move || {
            let (frames, mut process) = match factory() {
                Ok(value) => value,
                Err(e) => { let _ = ready.send(Err(e)); return; }
            };
            if ready.send(Ok(frames)).is_err() { return; }
            while let Ok((input, mut output, params)) = requests.recv() {
                let result = process(&input, &params, &mut output).and_then(|()| {
                    if output.len() != frames || output.iter().any(|v| !v.is_finite()) {
                        return Err(Error::Inference("invalid owner output".into()));
                    }
                    Ok((input, output))
                });
                let failed = result.is_err();
                if responses.send(result).is_err() || failed { break; }
            }
            // The captured !Send processor is dropped here, on the thread that created it.
        });
        let frames = match initialized.recv() {
            Ok(Ok(frames)) => frames,
            result => {
                drop(sender);
                let _ = thread.join();
                return Err(match result {
                    Ok(Err(e)) => e,
                    _ => Error::Runtime("inference owner ended during startup".into()),
                });
            }
        };
        Ok(Self { sender: Some(sender), receiver, thread: Some(thread), frames,
            input: Vec::with_capacity(frames), output: Vec::with_capacity(frames), failed: false })
    }
    pub fn block_frames(&self) -> usize { self.frames }
    pub fn process(&mut self, input: &[f32], params: &Params) -> Result<&[f32]> {
        if self.failed { return Err(Error::Inference("inference owner failed; restart required".into())); }
        if input.len() != self.frames { return Err(Error::BlockSize { expected: self.frames, got: input.len() }); }
        self.input.clear();
        self.input.extend_from_slice(input);
        self.failed = true;
        self.sender.as_ref().unwrap().try_send((std::mem::take(&mut self.input), std::mem::take(&mut self.output), *params))
            .map_err(|_| Error::Inference("inference owner disconnected or busy".into()))?;
        // Like Engine::process, this waits for completion. A device watchdog belongs
        // to Realtime, not the device-independent processor (first CUDA capture can
        // take seconds). A timeout here cannot cancel CUDA or bound Drop's join.
        let (reusable, output) = self.receiver.recv()
            .map_err(|e| Error::Inference(format!("inference owner response: {e}")))??;
        self.input = reusable;
        self.output = output;
        self.failed = false;
        Ok(&self.output)
    }
}
impl Drop for OwnedProcessor {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{rc::Rc, sync::{Arc, Mutex}, thread};
    #[test]
    fn engine_first_call_is_not_an_audio_device_watchdog() {
        let mut engine = OwnedProcessor::start(|| {
            let not_send = Rc::new(());
            Ok((1, move |input: &[f32], _params: &Params, output: &mut Vec<f32>| {
                let _ = &not_send;
                thread::sleep(std::time::Duration::from_millis(3100));
                output.clear();
                output.extend_from_slice(input);
                Ok(())
            }))
        }).unwrap();
        let params = Params { pitch: 0.0, threshold_db: -90.0, rms_mix: 1.0, skip_silence: false, drop_silent_context: false };
        assert_eq!(engine.process(&[0.25], &params).unwrap(), &[0.25]);
    }
    #[test]
    fn non_send_engine_lifecycle_and_live_values_stay_on_owner() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let record = events.clone();
        struct Local { _not_send: Rc<()>, events: Arc<Mutex<Vec<thread::ThreadId>>> }
        impl Drop for Local {
            fn drop(&mut self) { self.events.lock().unwrap().push(thread::current().id()); }
        }
        let mut client = OwnedProcessor::start(move || {
            record.lock().unwrap().push(thread::current().id());
            let local = Local { _not_send: Rc::new(()), events: record };
            Ok((2, move |audio: &[f32], params: &Params, output: &mut Vec<f32>| {
                local.events.lock().unwrap().push(thread::current().id());
                if params.pitch < 0.0 { return Err(Error::Inference("test failure".into())); }
                output.clear();
                output.extend(audio.iter().map(|v| v + params.pitch + params.threshold_db));
                Ok(())
            }))
        }).unwrap();
        let mut p = Params { pitch: 14.0, threshold_db: -90.0, rms_mix: 1.0, skip_silence: false, drop_silent_context: false };
        assert_eq!(client.process(&[1.0, 2.0], &p).unwrap(), &[-75.0, -74.0]);
        p.pitch = -1.0;
        assert!(client.process(&[1.0, 2.0], &p).is_err());
        assert!(client.process(&[1.0, 2.0], &p).is_err());
        drop(client);
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 4); // create, success, failure, drop (no processing after failure)
        assert_ne!(events[0], thread::current().id());
        assert!(events.iter().all(|id| *id == events[0]));
    }
}
