use crate::config::Config;
use crate::core::note::Hand;
use crate::core::{BpmList, Note};
use crate::hand_model::{ErgonomicHandSystem, FingerType, Vector2};
use super::network::DeepNeuralNetwork;
use super::vit::{go as vgo, V};
use super::Experience;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

pub(crate) const CONSECUTIVE_LIMIT: usize = 3;
pub(crate) const BUFFER_SIZE: usize = 256;
pub(crate) const TRAIN_INTERVAL: u64 = 8;
pub(crate) const GAE_GAMMA: f32 = 0.99;
pub(crate) const GAE_LAMBDA: f32 = 0.95;
pub(crate) const PPO_EPS: f32 = 0.2;
pub(crate) const PPO_VF_COEF: f32 = 0.5;

pub type Finger = crate::hand_model::FingerType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiTKAdvancedAI {
    /// Vision Transformer: frame + note layout -> L/R
    pub(crate) v: V,
    /// value/aux head (legacy PPO path; ViT drives decisions)
    pub(crate) net: DeepNeuralNetwork,
    pub(crate) hs: ErgonomicHandSystem,
    pub(crate) rotation: f32,
    eps: f32,
    total: u64,
    ncorrect: u64,
    recent: VecDeque<(Hand, f32, f32)>,
    buf: VecDeque<Experience>,
    /// last rendered frame (HxWx3, row-major) — ViT input cache
    #[serde(default)]
    last_frame: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct PNote {
    pub position: Vector2,
    pub time: f32,
    pub kind: crate::core::NoteKind,
    pub hand: Option<Hand>,
    pub ok: bool,
}

fn x_to_col(x: f32, w: usize) -> usize {
    let c = ((x + 1.0) * 0.5 * w as f32) as isize;
    c.clamp(0, w as isize - 1) as usize
}

impl PhiTKAdvancedAI {
    pub fn new(rotation: f32) -> Self {
        let mut ai = Self {
            v: V::n(),
            net: DeepNeuralNetwork::new(),
            hs: ErgonomicHandSystem::new(),
            rotation,
            eps: 0.25,
            total: 0,
            ncorrect: 0,
            recent: VecDeque::with_capacity(100),
            buf: VecDeque::with_capacity(BUFFER_SIZE),
            last_frame: vec![0.0; super::AI_IMAGE_SIZE],
        };
        ai.v.wake();
        ai.net.init_gpu_sync();
        ai
    }

    pub fn load_or_create(filepath: &str, rotation: f32, _config: &Config) -> Self {
        if let Ok(bytes) = fs::read(Path::new(filepath)) {
            if let Ok(mut ai) = bincode::deserialize::<Self>(&bytes) {
                if ai.validate() {
                    ai.rotation = rotation;
                    ai.update_hand_positions();
                    ai.v.wake();
                    ai.net.init_gpu_sync();
                    return ai;
                }
            }
        }
        let ai = Self::new(rotation);
        ai.save(filepath);
        ai
    }

    pub(crate) fn init_gpu_support(&mut self) {
        self.v.wake();
        self.net.init_gpu_sync();
    }

    fn validate(&self) -> bool {
        self.v.val() && self.net.validate() && self.eps.is_finite()
    }

    fn save(&self, filepath: &str) {
        if let Ok(data) = bincode::serialize(self) {
            let _ = fs::write(filepath, data);
        }
    }

    pub(crate) fn update_hand_positions(&mut self) {
        let r = self.rotation.to_radians();
        let (c, s) = (r.cos(), r.sin());
        self.hs.left_hand.position = Vector2::new(-0.3 * c, -0.3 * s);
        self.hs.right_hand.position = Vector2::new(0.3 * c, 0.3 * s);
    }

    /// Primary entry: ViT scores the frame against each note, distribution
    /// policy emits hands, physical model re-checks feasibility.
    pub fn analyze_and_assign(
        &mut self,
        notes: &mut [Note],
        config: &Config,
        _bpm_list: &BpmList,
        line_id: usize,
        time: f32,
        image_data: &[f32],
    ) {
        if notes.is_empty() {
            return;
        }
        self.hs.update(time);
        self.assign_vision(notes, image_data, line_id, time, config);
    }

    /// Worker-thread entry: same path as analyze_and_assign without Bpm.
    pub(crate) fn assign_frame(
        &mut self,
        notes: &mut [Note],
        image_data: &[f32],
        line_id: usize,
        time: f32,
        config: &Config,
    ) {
        if notes.is_empty() {
            return;
        }
        self.assign_vision(notes, image_data, line_id, time, config);
    }

    /// Adopt a network the background trainer finished, if any.
    pub(crate) fn poll_trainer(&mut self) {
        if let Some(net) = super::trainer::poll() {
            self.net = net;
        }
    }

    fn assign_vision(&mut self, notes: &mut [Note], image_data: &[f32], line_id: usize, time: f32, config: &Config) {
        self.poll_trainer();

        // cache frame for fallback paths
        if image_data.len() >= super::AI_IMAGE_SIZE {
            self.last_frame.clear();
            self.last_frame
                .extend_from_slice(&image_data[..super::AI_IMAGE_SIZE]);
        } else {
            self.last_frame.clear();
            self.last_frame.extend_from_slice(image_data);
            self.last_frame.resize(super::AI_IMAGE_SIZE, 0.0);
        }

        let mut img = self.last_frame.clone();
        img.resize(super::AI_IMAGE_SIZE, 0.0);

        // ViT: remember respack templates + attend frame -> per-note L/R
        let hands = vgo(&self.v, &img, notes);

        // decay exploration on ViT
        self.v.eps = (self.v.eps * 0.999).max(0.03);
        self.eps = self.v.eps;

        let mut pn = self.preprocess(notes);
        self.apply_hands(&mut pn, &hands, line_id, time);
        self.post_process(&mut pn);
        self.update_physical(&mut pn, time);
        self.apply(notes, &pn);

        self.store_exp(&img, &hands, notes);
        self.total += notes.len() as u64;
        if self.total % TRAIN_INTERVAL == 0 {
            self.train(config.hand_ai_epochs as usize, config.hand_ai_train_budget_ms as u64);
        }
        if self.total % 500 == 0 {
            self.save("phitk_ai_model.bin");
        }
    }

    fn preprocess(&self, notes: &[Note]) -> Vec<PNote> {
        notes
            .iter()
            .map(|n| {
                let pos = Vector2::new(
                    n.object.translation.0.now(),
                    n.object.translation.1.now(),
                );
                PNote {
                    position: pos,
                    time: n.time,
                    kind: n.kind.clone(),
                    hand: None,
                    ok: false,
                }
            })
            .collect()
    }

    fn apply_hands(&mut self, notes: &mut [PNote], hands: &[Hand], line_id: usize, time: f32) {
        for (n, &h) in notes.iter_mut().zip(hands.iter()) {
            n.hand = Some(h);
            let conf = if h == self.ideal_hand(n.position) {
                1.0
            } else {
                0.4
            };
            n.ok = conf > 0.5;
            self.hs
                .update_finger_state(h, FingerType::Index, n.position, n.time, n.ok, &n.kind);
            if h == self.ideal_hand(n.position) {
                self.ncorrect += 1;
            }
            self.recent.push_back((h, n.position.x, n.time));
            if self.recent.len() > 100 {
                self.recent.pop_front();
            }
        }
        let _ = (line_id, time);
    }

    fn ideal_hand(&self, pos: Vector2) -> Hand {
        if pos.x < 0.0 {
            Hand::Left
        } else {
            Hand::Right
        }
    }

    fn post_process(&self, notes: &mut [PNote]) {
        self.smooth(notes);
        self.limit_consecutive(notes);
    }

    fn smooth(&self, notes: &mut [PNote]) {
        if notes.len() < 3 {
            return;
        }
        for i in 1..notes.len() - 1 {
            if let (Some(p), Some(c), Some(n)) =
                (notes[i - 1].hand, notes[i].hand, notes[i + 1].hand)
            {
                if c != p && c != n && p == n {
                    let gp = notes[i].time - notes[i - 1].time;
                    let gn = notes[i + 1].time - notes[i].time;
                    if gp > 0.15 && gn > 0.15 {
                        let nx = notes[i].position.x;
                        let pr = match p {
                            Hand::Left => nx < 0.3,
                            Hand::Right => nx > -0.3,
                        };
                        if pr {
                            notes[i].hand = Some(p);
                        }
                    }
                }
            }
        }
    }

    fn limit_consecutive(&self, notes: &mut [PNote]) {
        let mut cc = 0;
        let mut lh = None;
        for n in notes.iter_mut() {
            if let Some(h) = n.hand {
                if lh == Some(h) {
                    cc += 1;
                } else {
                    cc = 1;
                    lh = Some(h);
                }
                if cc > CONSECUTIVE_LIMIT {
                    n.hand = Some(match h {
                        Hand::Left => Hand::Right,
                        Hand::Right => Hand::Left,
                    });
                    cc = 1;
                    lh = n.hand;
                }
            }
        }
    }

    fn update_physical(&mut self, notes: &mut [PNote], t: f32) {
        for n in notes.iter_mut() {
            if let Some(h) = n.hand {
                let (ok, _, _, _) =
                    self.hs
                        .evaluate_note_success(h, &n.position, n.time, t, &n.kind);
                n.ok = ok;
            }
        }
    }

    fn apply(&self, original: &mut [Note], processed: &[PNote]) {
        for (i, p) in processed.iter().enumerate() {
            if let Some(h) = p.hand {
                original[i].hand = h;
            }
        }
    }

    fn store_exp(&mut self, input: &[f32], hands: &[Hand], notes: &[Note]) {
        let mut column_rewards = vec![0.0f32; super::AI_IMAGE_W];
        for (n, &h) in notes.iter().zip(hands.iter()) {
            let col = x_to_col(n.object.translation.0.now(), super::AI_IMAGE_W);
            let correct = (h == Hand::Left) == (n.object.translation.0.now() < 0.0);
            column_rewards[col] += if correct { 1.0 } else { -0.5 };
        }
        for v in column_rewards.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }

        let value = self.net.value_head(input);
        let action = hands
            .iter()
            .fold(0usize, |a, &h| (a << 1) | (h == Hand::Right) as usize);
        let log_prob = 0.0;

        self.buf.push_back(Experience {
            state: input.to_vec(),
            action,
            log_prob,
            reward: column_rewards.iter().sum::<f32>() / column_rewards.len() as f32,
            value,
            next_value: 0.0,
            done: false,
            advantage: 0.0,
            return_: 0.0,
            old_out: column_rewards.clone(),
        });
        if self.buf.len() > BUFFER_SIZE {
            self.buf.pop_front();
        }
    }

    fn train(&mut self, epochs: usize, budget_ms: u64) {
        if self.buf.len() < 64 {
            return;
        }
        let n = 128.min(self.buf.len());
        let mut exps: Vec<Experience> = self.buf.range(self.buf.len() - n..).cloned().collect();

        let mut last_val = 0.0f32;
        let mut last_adv = 0.0f32;
        for i in (0..exps.len()).rev() {
            let next_val = if exps[i].done { 0.0 } else { last_val };
            let delta = exps[i].reward + GAE_GAMMA * next_val - exps[i].value;
            let adv = if exps[i].done {
                delta
            } else {
                delta + GAE_GAMMA * GAE_LAMBDA * last_adv
            };
            exps[i].advantage = adv;
            exps[i].return_ = adv + exps[i].value;
            last_val = exps[i].value;
            last_adv = adv;
        }

        let mean_adv = exps.iter().map(|e| e.advantage).sum::<f32>() / exps.len() as f32;
        let std_adv = (exps.iter().map(|e| (e.advantage - mean_adv).powi(2)).sum::<f32>()
            / exps.len() as f32)
            .sqrt()
            .max(1e-8);
        for e in exps.iter_mut() {
            e.advantage = (e.advantage - mean_adv) / std_adv;
        }

        // hand the update to the background trainer instead of blocking here
        if super::trainer::submit(&self.net, exps, epochs, budget_ms) {
            self.eps = (self.eps * 0.999).max(0.05);
            self.v.eps = self.eps;
        }
    }
}
