pub(crate) mod network;
pub mod ai;
pub(crate) mod trainer;
pub mod vit;

pub use ai::PhiTKAdvancedAI;
pub use ai::Finger;

pub const AI_IMAGE_W: usize = 80;
pub const AI_IMAGE_H: usize = 45;
pub const AI_IMAGE_CHANNELS: usize = 3;
pub const AI_IMAGE_SIZE: usize = AI_IMAGE_W * AI_IMAGE_H * AI_IMAGE_CHANNELS;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: usize,
    pub log_prob: f32,
    pub reward: f32,
    pub value: f32,
    pub next_value: f32,
    pub done: bool,
    pub advantage: f32,
    pub return_: f32,
    pub old_out: Vec<f32>,
}
use crate::config::Config;
use crate::core::{BpmList, Note, Vector};
use crossbeam_channel::{unbounded, Receiver as CbReceiver, Sender as CbSender};
use once_cell::sync::OnceCell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::thread;
use std::time::{Duration, Instant};

struct AiRequest {
    id: u64,
    line_id: usize,
    version: u64,
    timestamp: Instant,
    notes: Vec<Note>,
    rotation: f32,
    config: Arc<Config>,
    bpm_list: Arc<BpmList>,
    image_data: Vec<f32>,
}

#[derive(Clone)]
struct AiResponse {
    id: u64,
    line_id: usize,
    version: u64,
    timestamp: Instant,
    notes: Vec<Note>,
    checksum: u64,
}

struct LineState {
    ver: u64,
    last_full: Instant,
    pending: HashMap<u64, (Instant, u64)>,
}

impl Default for LineState {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            ver: 0,
            last_full: now,
            pending: HashMap::new(),
        }
    }
}

static AI_REQ_TX: OnceCell<CbSender<AiRequest>> = OnceCell::new();
static AI_RESP_RX: OnceCell<CbReceiver<AiResponse>> = OnceCell::new();
static LINE_STATES: OnceCell<Mutex<HashMap<usize, LineState>>> = OnceCell::new();
static LINE_RESP_QUEUES: OnceCell<Mutex<HashMap<usize, VecDeque<AiResponse>>>> = OnceCell::new();
static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);
static VERSION_COUNTER: AtomicU64 = AtomicU64::new(1);

const FULL_UPDATE_INTERVAL_MS: u64 = 8;
const REQUEST_TIMEOUT_MS: u64 = 5000;

static START_ONCE: Once = Once::new();

fn calculate_checksum(notes: &[Note]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();

    notes.len().hash(&mut hasher);
    for note in notes {
        note.time.to_bits().hash(&mut hasher);
        let x = note.object.translation.0.now();
        let y = note.object.translation.1.now();
        x.to_bits().hash(&mut hasher);
        y.to_bits().hash(&mut hasher);
        std::mem::discriminant(&note.kind).hash(&mut hasher);
        std::mem::discriminant(&note.hand).hash(&mut hasher);
    }
    hasher.finish()
}

fn match_and_merge_notes(original: &mut [Note], original_pos: &[Vector], updated: &[Note]) -> bool {
    const TIME_THRESHOLD_SEC: f32 = 0.05;
    const POS_THRESHOLD_SQ: f32 = 400.0;

    let mut used = vec![false; updated.len()];
    let mut ui: Vec<usize> = (0..updated.len()).collect();
    ui.sort_by(|&a, &b| {
        updated[a]
            .time
            .partial_cmp(&updated[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (orig, orig_pos) in original.iter_mut().zip(original_pos.iter()) {
        let mut bi: Option<usize> = None;
        let mut bs = std::f64::INFINITY;

        let ot = orig.time;
        let ss = ui.partition_point(|&i| updated[i].time < ot - TIME_THRESHOLD_SEC);
        let se = ui.partition_point(|&i| updated[i].time <= ot + TIME_THRESHOLD_SEC);

        for &idx in &ui[ss..se] {
            if used[idx] { continue; }
            if std::mem::discriminant(&orig.kind) != std::mem::discriminant(&updated[idx].kind) { continue; }

            let ox = orig_pos.x;
            let oy = orig_pos.y;
            let ux = updated[idx].object.translation.0.now();
            let uy = updated[idx].object.translation.1.now();

            let dx = ox - ux;
            let dy = oy - uy;
            let ds = dx * dx + dy * dy;
            if ds > POS_THRESHOLD_SQ { continue; }

            let dt = (ot - updated[idx].time).abs();
            let score = (dt as f64) * 1000.0 + (ds as f64).sqrt();
            if score < bs { bs = score; bi = Some(idx); }
        }

        if let Some(i) = bi {
            orig.hand = updated[i].hand;
            used[i] = true;
        }
    }
    true
}

fn cleanup_expired_requests(states: &mut HashMap<usize, LineState>) {
    let now = Instant::now();
    let timeout = Duration::from_millis(REQUEST_TIMEOUT_MS);
    for (_, state) in states.iter_mut() {
        state
            .pending
            .retain(|_, (ts, _)| now.duration_since(*ts) < timeout);
    }
}

fn start_ai_worker_if_needed(config: &Config) {
    static HAND_SPLIT: OnceCell<bool> = OnceCell::new();
    HAND_SPLIT.get_or_init(|| config.hand_split);

    START_ONCE.call_once(|| {
        let (tx_req, rx_req) = unbounded::<AiRequest>();
        let (tx_resp, rx_resp) = unbounded::<AiResponse>();

        AI_REQ_TX.set(tx_req.clone()).ok();
        AI_RESP_RX.set(rx_resp.clone()).ok();
        LINE_RESP_QUEUES.get_or_init(|| Mutex::new(HashMap::new()));

        let resp_rx = rx_resp.clone();
        thread::spawn(move || loop {
            match resp_rx.recv() {
                Ok(resp) => {
                    let map = LINE_RESP_QUEUES.get().unwrap();
                    let mut guard = map.lock().unwrap_or_else(|p| p.into_inner());
                    guard.entry(resp.line_id).or_default().push_back(resp);
                }
                Err(_) => break,
            }
        });

        let req_rx = rx_req;
        thread::spawn(move || {
            let hand_split = *HAND_SPLIT.get().unwrap_or(&false);
            let mut worker_config = Config::default();
            worker_config.hand_split = hand_split;
            let mut worker_ai =
                ai::PhiTKAdvancedAI::load_or_create("phitk_ai_model.bin", 0.0, &worker_config);
            println!("[GPU] Initializing GPU in AI worker thread...");
            worker_ai.init_gpu_support();
            println!("[GPU] Worker GPU init done");

            const BATCH_SIZE: usize = 8;
            const BATCH_TIMEOUT_MS: u64 = 16;

            loop {
                let first = match req_rx.recv() {
                    Ok(r) => r,
                    Err(_) => break,
                };
                if Instant::now().duration_since(first.timestamp) > Duration::from_millis(REQUEST_TIMEOUT_MS) {
                    continue;
                }
                worker_ai.poll_trainer();

                let mut batch = vec![first];
                let batch_start = Instant::now();
                while batch.len() < BATCH_SIZE {
                    match req_rx.try_recv() {
                        Ok(r) => {
                            if Instant::now().duration_since(r.timestamp) <= Duration::from_millis(REQUEST_TIMEOUT_MS) {
                                batch.push(r);
                            }
                        }
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

                let images: Vec<Vec<f32>> = batch.iter().map(|r| {
                    let mut v = r.image_data.clone();
                    v.resize(AI_IMAGE_SIZE, 0.0);
                    v
                }).collect();

                for (bi, req) in batch.iter().enumerate() {
                    let img = images[bi].as_slice();
                    let mut notes_copy = req.notes.clone();
                    let t = Instant::now().elapsed().as_secs_f32();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        worker_ai.assign_frame(&mut notes_copy, img, req.line_id, t, &req.config);
                        notes_copy
                    }));

                    if let Ok(analyzed) = result {
                        println!("[AI] line={} notes={} batch={}/{}", req.line_id, analyzed.len(), bi+1, batch.len());
                        let checksum = calculate_checksum(&analyzed);
                        let _ = tx_resp.send(AiResponse {
                            id: req.id,
                            line_id: req.line_id,
                            version: req.version,
                            timestamp: Instant::now(),
                            notes: analyzed,
                            checksum,
                        });
                    }
                }
            }
        });

        thread::spawn(|| {
            let cleanup_interval = Duration::from_secs(30);
            loop {
                thread::sleep(cleanup_interval);
                if let Some(ls) = LINE_STATES.get() {
                    if let Ok(mut guard) = ls.lock() {
                        cleanup_expired_requests(&mut guard);
                    }
                }
            }
        });
    });
}

pub fn assign_hands(
    notes: &mut [Note],
    config: &Config,
    line_id: usize,
    rotation: f32,
    bpm_list: &BpmList,
) {
    if notes.is_empty() || !config.hand_split {
        return;
    }

    start_ai_worker_if_needed(config);

    let now = Instant::now();
    let ls = LINE_STATES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = ls.lock().unwrap_or_else(|p| {
        eprintln!("Line states mutex poisoned, recovering...");
        p.into_inner()
    });
    let state = guard.entry(line_id).or_default();

    if let Some(map) = LINE_RESP_QUEUES.get() {
        let mut responses = Vec::with_capacity(5);
        {
            let mut mq = map.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(queue) = mq.get_mut(&line_id) {
                for _ in 0..5 {
                    if let Some(r) = queue.pop_front() {
                        responses.push(r);
                    } else {
                        break;
                    }
                }
            }
        }
        for resp in responses {
            if resp.line_id != line_id {
                continue;
            }
            if let Some((_, req_ver)) =             state.pending.remove(&resp.id) {
                if resp.version == req_ver && resp.checksum == calculate_checksum(&resp.notes) {
                    // Legacy path: no world positions available, fall back to the
                    // notes' own local translation (preserves previous behaviour).
                    let local_pos: Vec<Vector> = notes
                        .iter()
                        .map(|n| Vector::new(n.object.translation.0.now(), n.object.translation.1.now()))
                        .collect();
                    if match_and_merge_notes(notes, &local_pos, &resp.notes) {
                        state.ver = resp.version;
                        state.last_full = now;
                    }
                }
            }
        }
    }

    let should_update =
        now.duration_since(state.last_full) >= Duration::from_millis(FULL_UPDATE_INTERVAL_MS);
    if should_update && state.pending.len() < 3 {
        let version = VERSION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let req_id = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
        state.pending.insert(req_id, (now, version));

        let cfg = Arc::new(config.clone());
        let bpm = Arc::new(bpm_list.clone());
        let snapshot: Vec<Note> = notes.to_vec();

        let req = AiRequest {
            id: req_id,
            line_id,
            version,
            timestamp: now,
            notes: snapshot,
            rotation,
            config: cfg,
            bpm_list: bpm,
            image_data: vec![0.0; AI_IMAGE_SIZE],
        };

        if let Some(tx) = AI_REQ_TX.get() {
            if tx.send(req).is_err() {
                state.pending.remove(&req_id);
            }
        }
    }
    drop(guard);
}

pub fn assign_hands_unified_perspective(
    notes: &mut [Note],
    config: &Config,
    line_id: usize,
    rotation: f32,
    bpm_list: &BpmList,
    world_positions: &[Vector],
    make_image: impl FnOnce() -> Vec<f32>,
) {
    if notes.is_empty() {
        return;
    }

    start_ai_worker_if_needed(config);

    let now = Instant::now();
    let rad = rotation.to_radians();
    let cos_r = rad.cos();
    let sin_r = rad.sin();

    let ls = LINE_STATES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = ls.lock().unwrap_or_else(|p| {
        eprintln!("Line states mutex poisoned, recovering...");
        p.into_inner()
    });
    let state = guard.entry(line_id).or_default();

    if let Some(map) = LINE_RESP_QUEUES.get() {
        let mut responses = Vec::with_capacity(3);
        {
            let mut mq = map.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(queue) = mq.get_mut(&line_id) {
                for _ in 0..3 {
                    if let Some(r) = queue.pop_front() {
                        responses.push(r);
                    } else {
                        break;
                    }
                }
            }
        }
        for resp in responses {
            if resp.line_id != line_id {
                continue;
            }
            if let Some((_, req_ver)) = state.pending.remove(&resp.id) {
                if resp.version == req_ver {
                    if match_and_merge_notes(notes, world_positions, &resp.notes) {
                        state.ver = resp.version;
                        state.last_full = now;
                    }
                }
            }
        }
    }

    let should_update =
        now.duration_since(state.last_full) >= Duration::from_millis(FULL_UPDATE_INTERVAL_MS);
    if should_update && state.pending.len() < 2 {
        let version = VERSION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let req_id = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
        state.pending.insert(req_id, (now, version));

        if let Some(tx) = AI_REQ_TX.get() {
            let pn: Vec<Note> = notes
                .iter()
                .enumerate()
                .map(|(i, note)| {
                    let true_pos = world_positions.get(i).copied().unwrap_or_default();
                    let px = true_pos.x * cos_r - true_pos.y * sin_r;
                    let py = true_pos.x * sin_r + true_pos.y * cos_r;
                    let mut n = note.clone();
                    n.object.translation.0 = crate::core::AnimFloat::fixed(px);
                    n.object.translation.1 = crate::core::AnimFloat::fixed(py);
                    n
                })
                .collect();

            let cfg = Arc::new(config.clone());
            let bpm = Arc::new(bpm_list.clone());

            // only grab the frame when a request actually goes out: the
            // glReadPixels readback is the expensive part
            let image_data = make_image();
            if tx.send(AiRequest {
                id: req_id,
                line_id,
                version,
                timestamp: now,
                notes: pn,
                rotation,
                config: cfg,
                bpm_list: bpm,
                image_data,
            }).is_err() {
                state.pending.remove(&req_id);
            }
        }
    }
    drop(guard);
}
