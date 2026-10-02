pub mod ai;
pub(crate) mod network;
pub(crate) mod trainer;
pub mod vit;

pub use ai::{Finger, PhiTKAdvancedAI};

pub const AI_IMAGE_W: usize = 80;
pub const AI_IMAGE_H: usize = 45;
pub const AI_IMAGE_CHANNELS: usize = 3;
pub const AI_IMAGE_SIZE: usize = AI_IMAGE_W * AI_IMAGE_H * AI_IMAGE_CHANNELS;

pub(crate) const MODEL_PATH: &str = "phitk_ai_model.bin";

use crate::config::Config;
use crate::core::{Note, Vector};
use crossbeam_channel::{unbounded, Sender};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{info, warn};

const FULL_UPDATE_INTERVAL_MS: u64 = 8;
const REQUEST_TIMEOUT_MS: u64 = 5000;
const MAX_PENDING_PER_LINE: usize = 2;
const RESPONSES_PER_POLL: usize = 5;
const BATCH_SIZE: usize = 8;
const BATCH_TIMEOUT_MS: u64 = 16;

/// One rollout sample for the background PPO trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: Vec<f32>,
    /// Per-image-column right-hand bit, length `AI_IMAGE_W`.
    pub actions: Vec<u8>,
    /// Policy logits recorded at rollout time, needed for the PPO ratio.
    pub old_out: Vec<f32>,
    pub reward: f32,
    pub value: f32,
    pub advantage: f32,
    pub return_: f32,
}

struct AiRequest {
    id: u64,
    line_id: usize,
    version: u64,
    time: f32,
    created: Instant,
    notes: Vec<Note>,
    rotation: f32,
    config: Arc<Config>,
    image_data: Vec<f32>,
}

struct AiResponse {
    id: u64,
    line_id: usize,
    version: u64,
    notes: Vec<Note>,
}

#[derive(Default)]
struct LineState {
    last_full: Option<Instant>,
    pending: HashMap<u64, (Instant, u64)>,
}

static AI_REQ_TX: OnceCell<Sender<AiRequest>> = OnceCell::new();
static LINE_STATES: OnceCell<Mutex<HashMap<usize, LineState>>> = OnceCell::new();
static LINE_RESP_QUEUES: OnceCell<Mutex<HashMap<usize, VecDeque<AiResponse>>>> = OnceCell::new();
static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);
static VERSION_COUNTER: AtomicU64 = AtomicU64::new(1);
static START_ONCE: Once = Once::new();

fn expired(created: Instant) -> bool {
    created.elapsed() > Duration::from_millis(REQUEST_TIMEOUT_MS)
}

/// Writes the worker's hands back onto the live notes. Positions are compared in
/// the same rotated space the worker received them in.
fn match_and_merge_notes(original: &mut [Note], original_pos: &[Vector], updated: &[Note]) -> bool {
    const TIME_THRESHOLD_SEC: f32 = 0.05;
    const POS_THRESHOLD_SQ: f32 = 400.0;

    let mut used = vec![false; updated.len()];
    let mut order: Vec<usize> = (0..updated.len()).collect();
    order.sort_by(|&a, &b| {
        updated[a]
            .time
            .partial_cmp(&updated[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut merged = false;
    for (orig, orig_pos) in original.iter_mut().zip(original_pos.iter()) {
        let mut best: Option<usize> = None;
        let mut best_score = f64::INFINITY;

        let ot = orig.time;
        let start = order.partition_point(|&i| updated[i].time < ot - TIME_THRESHOLD_SEC);
        let end = order.partition_point(|&i| updated[i].time <= ot + TIME_THRESHOLD_SEC);

        for &idx in &order[start..end] {
            if used[idx] {
                continue;
            }
            if std::mem::discriminant(&orig.kind) != std::mem::discriminant(&updated[idx].kind) {
                continue;
            }

            let dx = orig_pos.x - updated[idx].object.translation.0.now();
            let dy = orig_pos.y - updated[idx].object.translation.1.now();
            let dist_sq = dx * dx + dy * dy;
            if dist_sq > POS_THRESHOLD_SQ {
                continue;
            }

            let dt = (ot - updated[idx].time).abs();
            let score = dt as f64 * 1000.0 + (dist_sq as f64).sqrt();
            if score < best_score {
                best_score = score;
                best = Some(idx);
            }
        }

        if let Some(i) = best {
            orig.hand = updated[i].hand;
            used[i] = true;
            merged = true;
        }
    }
    merged
}

fn cleanup_expired_requests(states: &mut HashMap<usize, LineState>) {
    let now = Instant::now();
    for state in states.values_mut() {
        state
            .pending
            .retain(|_, (created, _)| now.duration_since(*created) < Duration::from_millis(REQUEST_TIMEOUT_MS));
    }
}

fn start_ai_worker_if_needed() {
    START_ONCE.call_once(|| {
        let (req_tx, req_rx) = unbounded::<AiRequest>();
        let (resp_tx, resp_rx) = unbounded::<AiResponse>();

        let _ = AI_REQ_TX.set(req_tx);
        LINE_RESP_QUEUES.get_or_init(|| Mutex::new(HashMap::new()));

        let dispatcher_rx = resp_rx.clone();
        let dispatcher = thread::Builder::new()
            .name("phitk-ai-dispatcher".into())
            .spawn(move || {
                while let Ok(resp) = dispatcher_rx.recv() {
                    let queues = LINE_RESP_QUEUES.get().expect("queues initialized above");
                    let mut guard = queues.lock().unwrap_or_else(|p| p.into_inner());
                    guard.entry(resp.line_id).or_default().push_back(resp);
                }
            });

        let worker = thread::Builder::new()
            .name("phitk-ai-worker".into())
            .spawn(move || {
                let mut worker_ai = std::panic::catch_unwind(|| {
                    ai::PhiTKAdvancedAI::load_or_create(MODEL_PATH, 0.0)
                })
                .unwrap_or_else(|_| {
                    warn!("hand AI model init failed, retrying on the CPU path");
                    ai::PhiTKAdvancedAI::new(0.0)
                });
                info!("hand AI worker ready");

                loop {
                    let first = match req_rx.recv() {
                        Ok(req) => req,
                        Err(_) => break,
                    };
                    if expired(first.created) {
                        continue;
                    }
                    worker_ai.poll_trainer();

                    let mut batch = vec![first];
                    let batch_start = Instant::now();
                    while batch.len() < BATCH_SIZE {
                        match req_rx.try_recv() {
                            Ok(req) if !expired(req.created) => batch.push(req),
                            Ok(_) => {}
                            Err(_) => break,
                        }
                        if batch_start.elapsed() >= Duration::from_millis(BATCH_TIMEOUT_MS) {
                            break;
                        }
                    }

                    if worker_ai.rotation != batch[0].rotation {
                        worker_ai.rotation = batch[0].rotation;
                        worker_ai.update_hand_positions();
                    }

                    for req in batch {
                        let AiRequest {
                            id,
                            line_id,
                            version,
                            time,
                            mut notes,
                            config,
                            mut image_data,
                            ..
                        } = req;
                        image_data.resize(AI_IMAGE_SIZE, 0.0);
                        let analyzed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            worker_ai.assign_frame(&mut notes, &image_data, time, &config);
                        }));
                        match analyzed {
                            Ok(()) => {
                                let _ = resp_tx.send(AiResponse {
                                    id,
                                    line_id,
                                    version,
                                    notes,
                                });
                            }
                            Err(_) => warn!("hand AI worker panicked on line {line_id}"),
                        }
                    }
                }
            });

        let cleanup = thread::Builder::new()
            .name("phitk-ai-cleanup".into())
            .spawn(move || loop {
                thread::sleep(Duration::from_secs(30));
                if let Some(states) = LINE_STATES.get() {
                    let mut guard = states.lock().unwrap_or_else(|p| p.into_inner());
                    cleanup_expired_requests(&mut guard);
                }
            });

        if dispatcher.is_err() || worker.is_err() || cleanup.is_err() {
            warn!("hand AI worker threads could not be spawned");
        }
    });
}

/// Assign a hand to every note of a judge line from the current frame plus the
/// line's world-space note positions. Requests are throttled and merged
/// asynchronously by the worker thread.
pub fn assign_hands_unified_perspective(
    notes: &mut [Note],
    config: &Config,
    line_id: usize,
    rotation: f32,
    time: f32,
    world_positions: &[Vector],
    make_image: impl FnOnce() -> Vec<f32>,
) {
    if notes.is_empty() || !config.hand_split {
        return;
    }

    let rad = rotation.to_radians();
    let (sin_r, cos_r) = (rad.sin(), rad.cos());

    // The worker sees the frame as rendered, so notes are rotated into that
    // space and matched back against the same rotated positions.
    let rotated: Vec<Vector> = world_positions
        .iter()
        .map(|p| Vector::new(p.x * cos_r - p.y * sin_r, p.x * sin_r + p.y * cos_r))
        .collect();

    start_ai_worker_if_needed();

    let now = Instant::now();
    let states = LINE_STATES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = states.lock().unwrap_or_else(|p| {
        warn!("line state mutex poisoned, recovering");
        p.into_inner()
    });
    let state = guard.entry(line_id).or_default();

    if let Some(queues) = LINE_RESP_QUEUES.get() {
        let mut responses = Vec::with_capacity(RESPONSES_PER_POLL);
        {
            let mut queues = queues.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(queue) = queues.get_mut(&line_id) {
                for _ in 0..RESPONSES_PER_POLL {
                    match queue.pop_front() {
                        Some(resp) => responses.push(resp),
                        None => break,
                    }
                }
            }
        }
        for resp in responses {
            if resp.line_id != line_id {
                continue;
            }
            if let Some((_, req_ver)) = state.pending.remove(&resp.id) {
                if resp.version == req_ver && match_and_merge_notes(notes, &rotated, &resp.notes) {
                    state.last_full = Some(now);
                }
            }
        }
    }

    let due = state.last_full.map_or(true, |last| {
        now.duration_since(last) >= Duration::from_millis(FULL_UPDATE_INTERVAL_MS)
    });
    if due && state.pending.len() < MAX_PENDING_PER_LINE {
        let version = VERSION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let req_id = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
        state.pending.insert(req_id, (now, version));

        if let Some(tx) = AI_REQ_TX.get() {
            let snapshot: Vec<Note> = notes
                .iter()
                .enumerate()
                .map(|(i, note)| {
                    let pos = rotated.get(i).copied().unwrap_or_default();
                    let mut note = note.clone();
                    note.object.translation.0 = crate::core::AnimFloat::fixed(pos.x);
                    note.object.translation.1 = crate::core::AnimFloat::fixed(pos.y);
                    note
                })
                .collect();

            // Reading the framebuffer back is the expensive part, so only do it
            // when a request actually goes out.
            let image_data = make_image();
            let sent = tx.send(AiRequest {
                id: req_id,
                line_id,
                version,
                time,
                created: now,
                notes: snapshot,
                rotation,
                config: Arc::new(config.clone()),
                image_data,
            });
            if sent.is_err() {
                state.pending.remove(&req_id);
            }
        }
    }
}
