use super::network::DeepNeuralNetwork;
use super::vit::HandVisionTransformer;
use super::{Experience, AI_IMAGE_SIZE, AI_IMAGE_W, MODEL_PATH};
use crate::config::Config;
use crate::core::note::{Hand, NoteKind};
use crate::core::Note;
use crate::hand_model::{ErgonomicHandSystem, FingerType, Vector2};
use crate::judge::Judgement;
use crate::loss;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::Path;

pub(crate) const CONSECUTIVE_LIMIT: usize = 3;
pub(crate) const TRAIN_INTERVAL: u64 = 8;
pub(crate) const GAE_GAMMA: f32 = 0.99;
pub(crate) const GAE_LAMBDA: f32 = 0.95;
pub(crate) const PPO_EPS: f32 = 0.2;
pub(crate) const PPO_VF_COEF: f32 = 0.5;

const BUFFER_SIZE: usize = 256;
const MIN_SAMPLES: usize = 64;
const SAMPLE_LIMIT: usize = 128;
const SAVE_EVERY: u64 = 500;
const EPS_DECAY: f32 = 0.999;
const EPS_FLOOR: f32 = 0.03;

/// Model files carry a header so a file written by an older layout is rejected
/// instead of being misread as garbage weights.
const MODEL_MAGIC: [u8; 8] = *b"PHITKAI\0";
const MODEL_VERSION: u32 = 3;
const MODEL_HEADER: usize = 12;

pub type Finger = crate::hand_model::FingerType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiTKAdvancedAI {
    pub(crate) vision: HandVisionTransformer,
    /// Column-wise value/policy head, trained by the background PPO trainer.
    pub(crate) net: DeepNeuralNetwork,
    pub(crate) hs: ErgonomicHandSystem,
    pub(crate) rotation: f32,
    total: u64,
    buffer: VecDeque<Experience>,
    #[serde(default)]
    last_frame: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct PNote {
    pub position: Vector2,
    pub time: f32,
    pub kind: NoteKind,
    pub hand: Option<Hand>,
    pub feasible: bool,
    pub judgement: Judgement,
    pub time_error: f32,
}

fn column_of(x: f32, width: usize) -> usize {
    let column = ((x + 1.0) * 0.5 * width as f32) as isize;
    column.clamp(0, width as isize - 1) as usize
}

impl PhiTKAdvancedAI {
    pub fn new(rotation: f32) -> Self {
        let mut ai = Self {
            vision: HandVisionTransformer::new(),
            net: DeepNeuralNetwork::new(),
            hs: ErgonomicHandSystem::new(),
            rotation,
            total: 0,
            buffer: VecDeque::with_capacity(BUFFER_SIZE),
            last_frame: vec![0.0; AI_IMAGE_SIZE],
        };
        ai.vision.wake();
        ai
    }

    pub fn load_or_create(filepath: &str, rotation: f32) -> Self {
        if let Some(mut ai) = read_model(filepath) {
            if ai.validate() {
                ai.rotation = rotation;
                ai.update_hand_positions();
                ai.vision.wake();
                return ai;
            }
        }
        let ai = Self::new(rotation);
        ai.save(filepath);
        ai
    }

    fn validate(&self) -> bool {
        self.vision.validate() && self.net.validate()
    }

    fn save(&self, filepath: &str) {
        let Ok(data) = bincode::serialize(self) else { return };
        let mut file = Vec::with_capacity(MODEL_HEADER + data.len());
        file.extend_from_slice(&MODEL_MAGIC);
        file.extend_from_slice(&MODEL_VERSION.to_le_bytes());
        file.extend_from_slice(&data);
        let _ = fs::write(filepath, file);
    }

    pub(crate) fn update_hand_positions(&mut self) {
        let radians = self.rotation.to_radians();
        let (cos, sin) = (radians.cos(), radians.sin());
        self.hs.left_hand.position = Vector2::new(-0.3 * cos, -0.3 * sin);
        self.hs.right_hand.position = Vector2::new(0.3 * cos, 0.3 * sin);
    }

    /// Assign one hand per note from the frame, then fold the decision back into
    /// the experience buffer.
    pub(crate) fn assign_frame(
        &mut self,
        notes: &mut [Note],
        image_data: &[f32],
        time: f32,
        config: &Config,
    ) {
        if notes.is_empty() {
            return;
        }
        self.poll_trainer();
        self.cache_frame(image_data);

        self.hs.update(time);
        let hands = self.vision.decide(&self.last_frame, notes);
        self.vision.eps = (self.vision.eps * EPS_DECAY).max(EPS_FLOOR);

        let mut processed = self.preprocess(notes);
        self.apply_hands(&mut processed, &hands);
        self.post_process(&mut processed);
        self.update_physical(&mut processed, time);
        self.apply(notes, &processed);

        self.store_experience(&processed);
        self.total += notes.len() as u64;
        if self.total % TRAIN_INTERVAL == 0 {
            self.train(config.hand_ai_epochs as usize, config.hand_ai_train_budget_ms as u64);
        }
        if self.total % SAVE_EVERY == 0 {
            self.save(MODEL_PATH);
        }
    }

    /// Adopt a network the background trainer finished, if any.
    pub(crate) fn poll_trainer(&mut self) {
        if let Some(net) = super::trainer::poll() {
            self.net = net;
        }
    }

    fn cache_frame(&mut self, image_data: &[f32]) {
        self.last_frame.clear();
        let available = image_data.len().min(AI_IMAGE_SIZE);
        self.last_frame.extend_from_slice(&image_data[..available]);
        self.last_frame.resize(AI_IMAGE_SIZE, 0.0);
    }

    fn preprocess(&self, notes: &[Note]) -> Vec<PNote> {
        notes
            .iter()
            .map(|note| PNote {
                position: Vector2::new(note.object.translation.0.now(), note.object.translation.1.now()),
                time: note.time,
                kind: note.kind.clone(),
                hand: None,
                feasible: false,
                judgement: Judgement::Perfect,
                time_error: 0.0,
            })
            .collect()
    }

    fn apply_hands(&mut self, notes: &mut [PNote], hands: &[Hand]) {
        for (note, &hand) in notes.iter_mut().zip(hands.iter()) {
            note.hand = Some(hand);
            let matches_spatial_prior = hand == spatial_prior(note.position);
            self.hs.update_finger_state(
                hand,
                FingerType::Index,
                note.position,
                note.time,
                matches_spatial_prior,
                &note.kind,
            );
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
            let (Some(previous), Some(current), Some(next)) =
                (notes[i - 1].hand, notes[i].hand, notes[i + 1].hand)
            else {
                continue;
            };
            if current == previous || current == next || previous != next {
                continue;
            }
            let gap_previous = notes[i].time - notes[i - 1].time;
            let gap_next = notes[i + 1].time - notes[i].time;
            if gap_previous <= 0.15 || gap_next <= 0.15 {
                continue;
            }
            let x = notes[i].position.x;
            let close_to_previous = match previous {
                Hand::Left => x < 0.3,
                Hand::Right => x > -0.3,
            };
            if close_to_previous {
                notes[i].hand = Some(previous);
            }
        }
    }

    fn limit_consecutive(&self, notes: &mut [PNote]) {
        let mut run = 0;
        let mut last: Option<Hand> = None;
        for note in notes.iter_mut() {
            let Some(hand) = note.hand else { continue };
            if last == Some(hand) {
                run += 1;
            } else {
                run = 1;
                last = Some(hand);
            }
            if run > CONSECUTIVE_LIMIT {
                note.hand = Some(match hand {
                    Hand::Left => Hand::Right,
                    Hand::Right => Hand::Left,
                });
                run = 1;
                last = note.hand;
            }
        }
    }

    fn update_physical(&mut self, notes: &mut [PNote], current_time: f32) {
        for note in notes.iter_mut() {
            let Some(hand) = note.hand else { continue };
            let (feasible, position_error, time_error, _) =
                self.hs
                    .evaluate_note_success(hand, &note.position, note.time, current_time, &note.kind);
            note.feasible = feasible;
            note.time_error = time_error;
            note.judgement = loss::judgement_from_errors(feasible, position_error, time_error, &note.kind);
        }
    }

    fn apply(&self, notes: &mut [Note], processed: &[PNote]) {
        for (note, processed) in notes.iter_mut().zip(processed.iter()) {
            if let Some(hand) = processed.hand {
                note.hand = hand;
            }
        }
    }

    /// The reward is the negative mean note loss of the physical model, so the
    /// policy head and `loss.rs` score the same way.
    fn store_experience(&mut self, notes: &[PNote]) {
        let state = self.last_frame.clone();
        let old_out = self.net.forward(&state);
        let value = self.net.value_from(&old_out);
        let actions = column_actions(notes);

        let mut loss_sum = 0.0f32;
        let mut counted = 0usize;
        for note in notes {
            if note.hand.is_none() {
                continue;
            }
            loss_sum += loss::note_loss(note.judgement, note.time_error, note.feasible);
            counted += 1;
        }
        let reward = if counted == 0 { 0.0 } else { -loss_sum / counted as f32 };

        self.buffer.push_back(Experience {
            state,
            actions,
            old_out,
            reward,
            value,
            advantage: 0.0,
            return_: 0.0,
        });
        if self.buffer.len() > BUFFER_SIZE {
            self.buffer.pop_front();
        }
    }

    fn train(&mut self, epochs: usize, budget_ms: u64) {
        if self.buffer.len() < MIN_SAMPLES {
            return;
        }
        let count = SAMPLE_LIMIT.min(self.buffer.len());
        let mut samples: Vec<Experience> = self.buffer.range(self.buffer.len() - count..).cloned().collect();

        let mut next_value = 0.0f32;
        let mut next_advantage = 0.0f32;
        for sample in samples.iter_mut().rev() {
            let delta = sample.reward + GAE_GAMMA * next_value - sample.value;
            sample.advantage = delta + GAE_GAMMA * GAE_LAMBDA * next_advantage;
            sample.return_ = sample.advantage + sample.value;
            next_value = sample.value;
            next_advantage = sample.advantage;
        }

        let mean = samples.iter().map(|s| s.advantage).sum::<f32>() / samples.len() as f32;
        let deviation = (samples
            .iter()
            .map(|s| (s.advantage - mean) * (s.advantage - mean))
            .sum::<f32>()
            / samples.len() as f32)
            .sqrt()
            .max(1e-8);
        for sample in samples.iter_mut() {
            sample.advantage = (sample.advantage - mean) / deviation;
        }

        let _ = super::trainer::submit(&self.net, samples, epochs, budget_ms);
    }
}

fn read_model(filepath: &str) -> Option<PhiTKAdvancedAI> {
    let bytes = fs::read(Path::new(filepath)).ok()?;
    if bytes.len() <= MODEL_HEADER || bytes[..8] != MODEL_MAGIC {
        return None;
    }
    if u32::from_le_bytes(bytes[8..12].try_into().ok()?) != MODEL_VERSION {
        return None;
    }
    bincode::deserialize(&bytes[MODEL_HEADER..]).ok()
}

fn spatial_prior(position: Vector2) -> Hand {
    if position.x < 0.0 {
        Hand::Left
    } else {
        Hand::Right
    }
}

/// The policy head is one Bernoulli bit per image column: right hand or not.
fn column_actions(notes: &[PNote]) -> Vec<u8> {
    let mut right = [0u16; AI_IMAGE_W];
    let mut left = [0u16; AI_IMAGE_W];
    for note in notes {
        let Some(hand) = note.hand else { continue };
        let column = column_of(note.position.x, AI_IMAGE_W);
        match hand {
            Hand::Left => left[column] += 1,
            Hand::Right => right[column] += 1,
        }
    }
    (0..AI_IMAGE_W)
        .map(|column| (right[column] > left[column]) as u8)
        .collect()
}
