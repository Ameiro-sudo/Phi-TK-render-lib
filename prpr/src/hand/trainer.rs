//! Background PPO trainer.
//!
//! Keeps heavy weight updates off the AI worker thread (and therefore off the
//! frame budget): `submit` hands a cloned network plus a batch of experiences
//! to a dedicated thread, `poll` picks the trained network back up. Only one
//! job is ever in flight, so results can never be mismatched with their owner.

use super::network::DeepNeuralNetwork;
use super::Experience;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use once_cell::sync::OnceCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
use std::thread;
use std::time::Duration;

struct Job {
    net: DeepNeuralNetwork,
    exps: Vec<Experience>,
    epochs: usize,
    budget: Duration,
}

static JOB_TX: OnceCell<Sender<Job>> = OnceCell::new();
static DONE_TX: OnceCell<Sender<DeepNeuralNetwork>> = OnceCell::new();
static DONE_RX: OnceCell<Receiver<DeepNeuralNetwork>> = OnceCell::new();
static PENDING: AtomicBool = AtomicBool::new(false);
static INIT: Once = Once::new();

fn init() {
    INIT.call_once(|| {
        let (job_tx, job_rx) = bounded::<Job>(1);
        let (done_tx, done_rx) = bounded::<DeepNeuralNetwork>(1);
        let spawned = thread::Builder::new()
            .name("phitk-ai-trainer".into())
            .spawn(move || run(job_rx));
        if spawned.is_ok() {
            let _ = DONE_TX.set(done_tx);
            let _ = DONE_RX.set(done_rx);
            let _ = JOB_TX.set(job_tx);
        }
        // on failure the cells stay unset, so submit() fails cleanly
    });
}

fn run(job_rx: Receiver<Job>) {
    // half of the cores, at most 4, so the render/game threads keep headroom
    let threads = thread::available_parallelism()
        .map(|n| n.get() / 2)
        .unwrap_or(2)
        .clamp(1, 4);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("phitk-train-{i}"))
        .build();

    while let Ok(job) = job_rx.recv() {
        let Job { mut net, exps, epochs, budget } = job;
        let trained = catch_unwind(AssertUnwindSafe(|| {
            match &pool {
                Ok(p) => p.install(|| net.train_with_ppo(&exps, epochs, budget)),
                Err(_) => net.train_with_ppo(&exps, epochs, budget),
            }
        }));
        match trained {
            Ok(()) => {
                // drop any stale result, keep the channel at most one deep
                if let Some(tx) = DONE_TX.get() {
                    let _ = tx.try_send(net);
                }
            }
            Err(_) => {
                // training panicked: release the slot instead of wedging forever
                PENDING.store(false, Ordering::SeqCst);
            }
        }
    }
}

/// Queue a PPO update on the trainer thread.
///
/// Returns `false` when nothing was queued (no work, or a job is already in
/// flight) — the caller should then keep its current network untouched.
pub fn submit(net: &DeepNeuralNetwork, exps: Vec<Experience>, epochs: usize, budget_ms: u64) -> bool {
    if epochs == 0 || exps.is_empty() {
        return false;
    }
    if PENDING.swap(true, Ordering::SeqCst) {
        return false;
    }
    init();
    let tx = match JOB_TX.get() {
        Some(tx) => tx,
        None => {
            PENDING.store(false, Ordering::SeqCst);
            return false;
        }
    };
    match tx.try_send(Job {
        net: net.clone(),
        exps,
        epochs,
        budget: Duration::from_millis(budget_ms),
    }) {
        Ok(()) => true,
        Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
            PENDING.store(false, Ordering::SeqCst);
            false
        }
    }
}

/// Take a freshly trained network, if the trainer finished one.
pub fn poll() -> Option<DeepNeuralNetwork> {
    let rx = DONE_RX.get()?;
    match rx.try_recv() {
        Ok(net) => {
            PENDING.store(false, Ordering::SeqCst);
            Some(net)
        }
        Err(_) => None,
    }
}

/// True while a training job is queued or running.
#[allow(dead_code)]
pub fn is_pending() -> bool {
    PENDING.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;


    fn exps(n: usize) -> Vec<Experience> {
        (0..n)
            .map(|i| Experience {
                state: vec![0.05 * (i as f32 + 1.0); crate::hand::AI_IMAGE_SIZE],
                action: i,
                log_prob: 0.0,
                reward: 0.5,
                value: 0.1,
                next_value: 0.0,
                done: false,
                advantage: 0.3,
                return_: 0.4,
                old_out: vec![0.2; crate::hand::AI_IMAGE_W],
            })
            .collect()
    }

    #[test]
    fn submit_rejects_empty_job() {
        let net = DeepNeuralNetwork::new();
        assert!(!submit(&net, vec![], 4, 50));
        assert!(!submit(&net, exps(8), 0, 50));
    }

    /// Round trip through the background thread: submit never blocks the
    /// caller, and the trained network comes back through `poll`.
    #[test]
    fn trainer_roundtrip() {
        let net = DeepNeuralNetwork::new();
        assert!(submit(&net, exps(8), 1, 30_000), "job should be queued");
        assert!(is_pending());

        let deadline = Instant::now() + Duration::from_secs(30);
        let trained = loop {
            if let Some(n) = poll() {
                break Some(n);
            }
            if Instant::now() >= deadline {
                break None;
            }
            thread::sleep(Duration::from_millis(5));
        };
        let trained = trained.expect("trainer did not finish in time");
        assert!(trained.validate());
        assert!(!is_pending(), "slot must be released after poll");
    }
}

