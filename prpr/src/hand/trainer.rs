//! Background PPO trainer. One job is in flight at a time, so a trained network
//! can never be mismatched with its owner: `submit` hands a cloned network plus
//! a batch of experiences to the worker thread, `poll` picks the result up.

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
    network: DeepNeuralNetwork,
    experiences: Vec<Experience>,
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
        // On failure the cells stay unset, so `submit` just reports "not queued".
    });
}

fn run(job_rx: Receiver<Job>) {
    // Half of the cores, at most four, so render and game threads keep headroom.
    let threads = thread::available_parallelism()
        .map(|n| n.get() / 2)
        .unwrap_or(2)
        .clamp(1, 4);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("phitk-train-{i}"))
        .build();

    while let Ok(job) = job_rx.recv() {
        let Job { mut network, experiences, epochs, budget } = job;
        let trained = catch_unwind(AssertUnwindSafe(|| match &pool {
            Ok(pool) => pool.install(|| network.train_with_ppo(&experiences, epochs, budget)),
            Err(_) => network.train_with_ppo(&experiences, epochs, budget),
        }));
        match trained {
            Ok(()) => {
                // Keep the channel one deep and drop a stale result instead of blocking.
                if let Some(tx) = DONE_TX.get() {
                    let _ = tx.try_send(network);
                }
            }
            Err(_) => PENDING.store(false, Ordering::SeqCst),
        }
    }
}

/// Queues a PPO update. Returns `false` when nothing was queued (no work, or a
/// job is already in flight), in which case the caller keeps its network.
pub fn submit(network: &DeepNeuralNetwork, experiences: Vec<Experience>, epochs: usize, budget_ms: u64) -> bool {
    if epochs == 0 || experiences.is_empty() {
        return false;
    }
    if PENDING.swap(true, Ordering::SeqCst) {
        return false;
    }
    init();
    let Some(tx) = JOB_TX.get() else {
        PENDING.store(false, Ordering::SeqCst);
        return false;
    };
    match tx.try_send(Job {
        network: network.clone(),
        experiences,
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

/// Takes a freshly trained network, if the trainer finished one.
pub fn poll() -> Option<DeepNeuralNetwork> {
    let receiver = DONE_RX.get()?;
    match receiver.try_recv() {
        Ok(network) => {
            PENDING.store(false, Ordering::SeqCst);
            Some(network)
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

    fn experiences(count: usize) -> Vec<Experience> {
        (0..count)
            .map(|i| Experience {
                state: vec![0.05 * (i as f32 + 1.0); crate::hand::AI_IMAGE_SIZE],
                actions: vec![(i % 2) as u8; crate::hand::AI_IMAGE_W],
                old_out: vec![0.2; crate::hand::AI_IMAGE_W],
                reward: 0.5,
                value: 0.1,
                advantage: 0.3,
                return_: 0.4,
            })
            .collect()
    }

    #[test]
    fn submit_rejects_empty_job() {
        let network = DeepNeuralNetwork::new();
        assert!(!submit(&network, Vec::new(), 4, 50));
        assert!(!submit(&network, experiences(8), 0, 50));
    }

    #[test]
    fn trainer_roundtrip() {
        let network = DeepNeuralNetwork::new();
        assert!(submit(&network, experiences(8), 1, 30_000), "job should be queued");
        assert!(is_pending());

        let deadline = Instant::now() + Duration::from_secs(30);
        let trained = loop {
            if let Some(network) = poll() {
                break Some(network);
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
