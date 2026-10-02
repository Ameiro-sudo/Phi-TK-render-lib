use crate::core::note::{Hand, Note, NoteKind};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const PATCH_SIZE: usize = 5;
pub const PATCH_COLS: usize = crate::hand::AI_IMAGE_W / PATCH_SIZE;
pub const PATCH_ROWS: usize = crate::hand::AI_IMAGE_H / PATCH_SIZE;
pub const PATCH_COUNT: usize = PATCH_COLS * PATCH_ROWS;
pub const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * crate::hand::AI_IMAGE_CHANNELS;
pub const MODEL_DIM: usize = 64;
pub const HEAD_COUNT: usize = 4;
pub const HEAD_DIM: usize = MODEL_DIM / HEAD_COUNT;
pub const BLOCK_COUNT: usize = 2;
pub const NOTE_FEATURE_DIM: usize = 16;
pub const KIND_COUNT: usize = 4;

const SPRITES: [&[u8]; KIND_COUNT] = [
    include_bytes!("../respack/click.png"),
    include_bytes!("../respack/hold.png"),
    include_bytes!("../respack/flick.png"),
    include_bytes!("../respack/drag.png"),
];

fn kind_index(kind: &NoteKind) -> usize {
    match kind {
        NoteKind::Click => 0,
        NoteKind::Hold { .. } => 1,
        NoteKind::Flick => 2,
        NoteKind::Drag => 3,
    }
}

fn image_u(x: f32) -> f32 {
    ((x + 1.0) * 0.5).clamp(0.0, 1.0)
}

fn image_v(y: f32) -> f32 {
    ((1.0 - y) * 0.5).clamp(0.0, 1.0)
}

fn patch_index(u: f32, v: f32) -> usize {
    let x = ((u * crate::hand::AI_IMAGE_W as f32) as usize).min(crate::hand::AI_IMAGE_W - 1) / PATCH_SIZE;
    let y = ((v * crate::hand::AI_IMAGE_H as f32) as usize).min(crate::hand::AI_IMAGE_H - 1) / PATCH_SIZE;
    y.min(PATCH_ROWS - 1) * PATCH_COLS + x.min(PATCH_COLS - 1)
}

fn softmax(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for score in scores.iter_mut() {
        *score = (*score - max).exp();
        sum += *score;
    }
    let inv = 1.0 / sum.max(1e-8);
    for score in scores.iter_mut() {
        *score *= inv;
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

fn positional_value(position: usize, dim: usize) -> f32 {
    let wavelength = 1.0 / 10000f32.powf(2.0 * (dim / 2) as f32 / MODEL_DIM as f32);
    let angle = position as f32 * wavelength;
    if dim % 2 == 0 {
        angle.sin()
    } else {
        angle.cos()
    }
}

fn positional_table() -> Vec<f32> {
    (0..PATCH_COUNT * MODEL_DIM)
        .map(|i| positional_value(i / MODEL_DIM, i % MODEL_DIM))
        .collect()
}

/// Mean RGB of the four embedded note sprites, compared against the patch a
/// note sits on.
#[derive(Clone, Serialize, Deserialize)]
pub struct TemplateMemory {
    pub mean_rgb: [[f32; 3]; KIND_COUNT],
    #[serde(default)]
    pub ready: bool,
}

impl TemplateMemory {
    pub fn new() -> Self {
        let mut mean_rgb = [[0.0f32; 3]; KIND_COUNT];
        for (index, raw) in SPRITES.iter().enumerate() {
            if let Some(rgb) = sprite_mean_rgb(raw) {
                mean_rgb[index] = rgb;
            }
        }
        Self { mean_rgb, ready: true }
    }

    pub fn wake(&mut self) {
        if !self.ready {
            *self = Self::new();
        }
    }

    fn similarity(&self, kind: usize, rgb: [f32; 3]) -> f32 {
        let template = self.mean_rgb[kind];
        let dot = template[0] * rgb[0] + template[1] * rgb[1] + template[2] * rgb[2];
        let norm_template = (template[0] * template[0] + template[1] * template[1] + template[2] * template[2]).sqrt();
        let norm_rgb = (rgb[0] * rgb[0] + rgb[1] * rgb[1] + rgb[2] * rgb[2]).sqrt();
        if norm_template <= 1e-6 || norm_rgb <= 1e-6 {
            0.0
        } else {
            dot / (norm_template * norm_rgb)
        }
    }
}

fn sprite_mean_rgb(raw: &[u8]) -> Option<[f32; 3]> {
    let image = image::load_from_memory(raw).ok()?.thumbnail(16, 16).to_rgb8();
    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 {
        return None;
    }
    let mut sum = [0.0f32; 3];
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            for channel in 0..3 {
                sum[channel] += pixel[channel] as f32 / 255.0;
            }
        }
    }
    let count = (width * height) as f32;
    Some([sum[0] / count, sum[1] / count, sum[2] / count])
}

fn patch_mean_rgb(frame: &[f32], u: f32, v: f32) -> [f32; 3] {
    let pixel_x = ((u * crate::hand::AI_IMAGE_W as f32) as usize).min(crate::hand::AI_IMAGE_W - 1);
    let pixel_y = ((v * crate::hand::AI_IMAGE_H as f32) as usize).min(crate::hand::AI_IMAGE_H - 1);
    let mut sum = [0.0f32; 3];
    let mut count = 0.0f32;
    for dy in 0..PATCH_SIZE {
        for dx in 0..PATCH_SIZE {
            let start = ((pixel_y / PATCH_SIZE * PATCH_SIZE + dy) * crate::hand::AI_IMAGE_W
                + pixel_x / PATCH_SIZE * PATCH_SIZE
                + dx)
                * crate::hand::AI_IMAGE_CHANNELS;
            if start + crate::hand::AI_IMAGE_CHANNELS <= frame.len() {
                for channel in 0..3 {
                    sum[channel] += frame[start + channel];
                }
                count += 1.0;
            }
        }
    }
    if count == 0.0 {
        [0.0; 3]
    } else {
        [sum[0] / count, sum[1] / count, sum[2] / count]
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Linear {
    /// Row-major, `weights.len() / input` rows of `input` weights.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub input: usize,
}

impl Linear {
    pub fn new(input: usize, out: usize) -> Self {
        let scale = (2.0 / input as f32).sqrt();
        Self {
            weights: (0..out * input).map(|_| (fastrand::f32() * 2.0 - 1.0) * scale).collect(),
            bias: vec![0.0; out],
            input,
        }
    }

    pub fn forward(&self, x: &[f32], y: &mut [f32]) {
        y.par_iter_mut().enumerate().for_each(|(row, value)| {
            let mut sum = self.bias[row];
            for (weight, input) in self.weights[row * self.input..(row + 1) * self.input].iter().zip(x) {
                sum += weight * input;
            }
            *value = if sum.is_finite() { sum.clamp(-50.0, 50.0) } else { 0.0 };
        });
    }

    fn is_finite(&self) -> bool {
        self.weights.iter().all(|w| w.is_finite()) && self.bias.iter().all(|b| b.is_finite())
    }
}

/// Root-mean-square normalisation: `y = x / sqrt(mean(x^2) + eps) * gain`.
#[derive(Clone, Serialize, Deserialize)]
pub struct RmsNorm {
    pub gain: Vec<f32>,
}

impl RmsNorm {
    pub fn new(dim: usize) -> Self {
        Self { gain: vec![1.0; dim] }
    }

    pub fn forward(&self, x: &[f32], y: &mut [f32]) {
        if x.is_empty() && x == 0 { return; }
        let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv_rms = 1.0 / (mean_square + 1e-5).sqrt();
        for ((out, input), gain) in y.iter_mut().zip(x.iter()).zip(self.gain.iter()) {
            *out = *input * inv_rms * *gain;
        }
    }

    fn is_finite(&self) -> bool { self.gain.iter().all(|g| g.is_finite()) }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EncoderBlock {
    pub norm_attention: RmsNorm,
    pub norm_ffn: RmsNorm,
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub out: Linear,
    pub ffn_in: Linear,
    pub ffn_out: Linear,
}

impl EncoderBlock {
    pub fn new() -> Self {
        Self {
            norm_attention: RmsNorm::new(MODEL_DIM),
            norm_ffn: RmsNorm::new(MODEL_DIM),
            query: Linear::new(MODEL_DIM, MODEL_DIM),
            key: Linear::new(MODEL_DIM, MODEL_DIM),
            value: Linear::new(MODEL_DIM, MODEL_DIM),
            out: Linear::new(MODEL_DIM, MODEL_DIM),
            ffn_in: Linear::new(MODEL_DIM, MODEL_DIM * 4),
            ffn_out: Linear::new(MODEL_DIM * 4, MODEL_DIM),
        }
    }

    /// Pre-norm self-attention plus FFN over `token_count` tokens of `MODEL_DIM`.
    pub fn forward(&self, tokens: &mut [f32], token_count: usize) {
        let mut normalized = vec![0.0f32; tokens.len()];
        for (input, output) in tokens.chunks(MODEL_DIM).zip(normalized.chunks_mut(MODEL_DIM)) {
            self.norm_attention.forward(input, output);
        }

        let mut queries = vec![0.0f32; token_count * MODEL_DIM];
        let mut keys = vec![0.0f32; token_count * MODEL_DIM];
        let mut values = vec![0.0f32; token_count * MODEL_DIM];
        for t in 0..token_count {
            let range = t * MODEL_DIM..(t + 1) * MODEL_DIM;
            self.query.forward(&normalized[range.clone()], &mut queries[range.clone()]);
            self.key.forward(&normalized[range.clone()], &mut keys[range.clone()]);
            self.value.forward(&normalized[range.clone()], &mut values[range]);
        }

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attended = vec![0.0f32; token_count * MODEL_DIM];
        attended.par_chunks_mut(MODEL_DIM).enumerate().for_each(|(t, out)| {
            for head in 0..HEAD_COUNT {
                let base = head * HEAD_DIM;
                let query = &queries[t * MODEL_DIM + base..t * MODEL_DIM + base + HEAD_DIM];
                let mut scores = vec![0.0f32; token_count];
                for (j, score) in scores.iter_mut().enumerate() {
                    let key = &keys[j * MODEL_DIM + base..j * MODEL_DIM + base + HEAD_DIM];
                    let mut dot = 0.0;
                    for i in 0..HEAD_DIM {
                        dot += query[i] * key[i];
                    }
                    *score = dot * scale;
                }
                softmax(&mut scores);
                for i in 0..HEAD_DIM {
                    let mut acc = 0.0;
                    for (j, score) in scores.iter().enumerate() {
                        acc += score * values[j * MODEL_DIM + base + i];
                    }
                    out[base + i] = acc;
                }
            }
        });

        let mut projected = vec![0.0f32; token_count * MODEL_DIM];
        for t in 0..token_count {
            let range = t * MODEL_DIM..(t + 1) * MODEL_DIM;
            self.out.forward(&attended[range.clone()], &mut projected[range]);
        }
        for (token, residual) in tokens.iter_mut().zip(projected) {
            *token += residual;
        }

        let mut hidden_input = vec![0.0f32; tokens.len()];
        for (input, output) in tokens.chunks(MODEL_DIM).zip(hidden_input.chunks_mut(MODEL_DIM)) {
            self.norm_ffn.forward(input, output);
        }
        let mut output = vec![0.0f32; token_count * MODEL_DIM];
        for t in 0..token_count {
            let range = t * MODEL_DIM..(t + 1) * MODEL_DIM;
            let mut hidden = vec![0.0f32; MODEL_DIM * 4];
            self.ffn_in.forward(&hidden_input[range.clone()], &mut hidden);
            for value in hidden.iter_mut() {
                *value = gelu(*value);
            }
            self.ffn_out.forward(&hidden, &mut output[range]);
        }
        for (token, residual) in tokens.iter_mut().zip(output) {
            *token += residual;
        }
    }

    /// Single query token attending over the visual tokens, with a query residual.
    pub fn cross_attention(&self, query: &[f32], tokens: &[f32]) -> Vec<f32> {
        let token_count = tokens.len() / MODEL_DIM;
        let mut normalized = vec![0.0f32; MODEL_DIM];
        self.norm_attention.forward(query, &mut normalized);
        let mut projected_query = vec![0.0f32; MODEL_DIM];
        self.query.forward(&normalized, &mut projected_query);

        let mut keys = vec![0.0f32; token_count * MODEL_DIM];
        let mut values = vec![0.0f32; token_count * MODEL_DIM];
        for j in 0..token_count {
            let range = j * MODEL_DIM..(j + 1) * MODEL_DIM;
            self.key.forward(&tokens[range.clone()], &mut keys[range.clone()]);
            self.value.forward(&tokens[range.clone()], &mut values[range]);
        }

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut head_out = vec![0.0f32; MODEL_DIM];
        for head in 0..HEAD_COUNT {
            let base = head * HEAD_DIM;
            let mut scores = vec![0.0f32; token_count];
            for (j, score) in scores.iter_mut().enumerate() {
                let key = &keys[j * MODEL_DIM + base..j * MODEL_DIM + base + HEAD_DIM];
                let mut dot = 0.0;
                for i in 0..HEAD_DIM {
                    dot += projected_query[base + i] * key[i];
                }
                *score = dot * scale;
            }
            softmax(&mut scores);
            for i in 0..HEAD_DIM {
                let mut acc = 0.0;
                for (j, score) in scores.iter().enumerate() {
                    acc += score * values[j * MODEL_DIM + base + i];
                }
                head_out[base + i] = acc;
            }
        }

        let mut projected = vec![0.0f32; MODEL_DIM];
        self.out.forward(&head_out, &mut projected);
        let mut result = query.to_vec();
        for i in 0..MODEL_DIM {
            result[i] += projected[i];
        }
        result
    }

    fn is_finite(&self) -> bool {
        self.norm_attention.is_finite()
            && self.norm_ffn.is_finite()
            && self.query.is_finite()
            && self.key.is_finite()
            && self.value.is_finite()
            && self.out.is_finite()
            && self.ffn_in.is_finite()
            && self.ffn_out.is_finite()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct HandVisionTransformer {
    pub patch_embed: Linear,
    pub note_embed: Linear,
    pub blocks: Vec<EncoderBlock>,
    pub classifier: Linear,
    pub templates: TemplateMemory,
    pub positional: Vec<f32>,
    pub eps: f32,
    #[serde(skip)]
    pub gpu: Option<crate::gpu_vit::VitGpu>,
}

impl std::fmt::Debug for HandVisionTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandVisionTransformer")
            .field("blocks", &self.blocks.len())
            .field("eps", &self.eps)
            .finish()
    }
}

impl HandVisionTransformer {
    pub fn new() -> Self {
        Self {
            patch_embed: Linear::new(PATCH_DIM, MODEL_DIM),
            note_embed: Linear::new(NOTE_FEATURE_DIM, MODEL_DIM),
            blocks: (0..BLOCK_COUNT).map(|_| EncoderBlock::new()).collect(),
            classifier: Linear::new(MODEL_DIM * 2, 1),
            templates: TemplateMemory::new(),
            positional: positional_table(),
            eps: 0.15,
            gpu: None,
        }
    }

    pub fn wake(&mut self) {
        self.templates.wake();
        if self.positional.len() != PATCH_COUNT * MODEL_DIM {
            self.positional = positional_table();
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            self.eps = 0.15;
        }
        if self.gpu.is_none() {
            self.gpu = crate::gpu_vit::VitGpu::try_new(self);
        }
    }

    pub fn validate(&self) -> bool {
        self.patch_embed.is_finite()
            && self.note_embed.is_finite()
            && self.classifier.is_finite()
            && self.positional.len() == PATCH_COUNT * MODEL_DIM
            && self.positional.iter().all(|v| v.is_finite())
            && self.blocks.iter().all(|block| block.is_finite())
            && self.eps.is_finite()
    }

    pub fn decide(&self, frame: &[f32], notes: &[Note]) -> Vec<Hand> {
        if notes.is_empty() {
            return Vec::new();
        }
        let mut logits = self.forward(frame, notes);
        if self.eps > 0.0 {
            for logit in logits.iter_mut() {
                if fastrand::f32() < self.eps {
                    *logit = -*logit;
                }
            }
        }
        self.select_hands(notes, &logits)
    }

    pub fn forward(&self, frame: &[f32], notes: &[Note]) -> Vec<f32> {
        if notes.is_empty() {
            return Vec::new();
        }
        if let Some(gpu) = self.gpu.as_ref() {
            if frame.len() >= crate::hand::AI_IMAGE_SIZE {
                let chunk = gpu.query_capacity();
                let (features, patches) = self.pack_queries(frame, notes);
                if chunk > 0 {
                    // A judge line can hold far more notes than one dispatch
                    // accepts, so the query set is chunked instead of abandoned.
                    let mut logits = Vec::with_capacity(notes.len());
                    let mut complete = true;
                    for start in (0..notes.len()).step_by(chunk) {
                        let end = (start + chunk).min(notes.len());
                        let feature_range = start * NOTE_FEATURE_DIM..end * NOTE_FEATURE_DIM;
                        match gpu.forward(frame, &features[feature_range], &patches[start..end]) {
                            Some(part) if part.len() == end - start => logits.extend_from_slice(&part),
                            _ => {
                                complete = false;
                                break;
                            }
                        }
                    }
                    if complete && logits.iter().all(|logit| logit.is_finite()) {
                        return logits;
                    }
                }
            }
        }
        self.forward_cpu(frame, notes)
    }

    fn forward_cpu(&self, frame: &[f32], notes: &[Note]) -> Vec<f32> {
        let mut tokens = self.embed_patches(frame);
        for block in &self.blocks {
            block.forward(&mut tokens, PATCH_COUNT);
        }

        notes
            .par_iter()
            .map(|note| {
                let u = image_u(note.object.translation.0.now());
                let v = image_v(note.object.translation.1.now());
                let patch = patch_index(u, v);
                let features = self.note_features(note, u, v, frame);
                let mut query = vec![0.0f32; MODEL_DIM];
                self.note_embed.forward(&features, &mut query);
                for d in 0..MODEL_DIM {
                    query[d] += self.positional[patch * MODEL_DIM + d] * 0.15;
                }

                let mut context = Vec::with_capacity(MODEL_DIM);
                for block in &self.blocks {
                    context = block.cross_attention(&query, &tokens);
                    for d in 0..MODEL_DIM {
                        query[d] = query[d] * 0.5 + context[d] * 0.5;
                    }
                }

                let mut fused = Vec::with_capacity(MODEL_DIM * 2);
                fused.extend_from_slice(&query);
                fused.extend_from_slice(&context);
                let mut logit = [0.0f32; 1];
                self.classifier.forward(&fused, &mut logit);
                logit[0]
            })
            .collect()
    }

    fn embed_patches(&self, frame: &[f32]) -> Vec<f32> {
        let width = crate::hand::AI_IMAGE_W;
        let channels = crate::hand::AI_IMAGE_CHANNELS;
        let mut tokens = vec![0.0f32; PATCH_COUNT * MODEL_DIM];
        tokens.par_chunks_mut(MODEL_DIM).enumerate().for_each(|(patch, token)| {
            let row = patch / PATCH_COLS;
            let col = patch % PATCH_COLS;
            let mut raw = vec![0.0f32; PATCH_DIM];
            let mut i = 0;
            for dy in 0..PATCH_SIZE {
                for dx in 0..PATCH_SIZE {
                    let start = ((row * PATCH_SIZE + dy) * width + col * PATCH_SIZE + dx) * channels;
                    for channel in 0..channels {
                        if start + channel < frame.len() {
                            raw[i] = frame[start + channel];
                        }
                        i += 1;
                    }
                }
            }
            self.patch_embed.forward(&raw, token);
            for d in 0..MODEL_DIM {
                token[d] += self.positional[patch * MODEL_DIM + d];
                if !token[d].is_finite() {
                    token[d] = 0.0;
                }
            }
        });
        tokens
    }

    fn note_features(&self, note: &Note, u: f32, v: f32, frame: &[f32]) -> [f32; NOTE_FEATURE_DIM] {
        let kind = kind_index(&note.kind);
        let mut f = [0.0f32; NOTE_FEATURE_DIM];
        f[0] = u;
        f[1] = v;
        f[2] = note.object.translation.0.now();
        f[3] = note.object.translation.1.now();
        f[4] = note.time;
        f[5] = note.height;
        f[6] = note.speed;
        f[7] = note.above as u8 as f32;
        f[8 + kind] = 1.0;
        f[12] = note.fake as u8 as f32;
        f[13] = match note.kind {
            NoteKind::Hold { end_time, .. } => (end_time - note.time).max(0.0),
            _ => 0.0,
        };
        f[14] = self.templates.similarity(kind, patch_mean_rgb(frame, u, v));
        f[15] = (note.time * 7.0).sin();
        f
    }

    pub(crate) fn pack_queries(&self, frame: &[f32], notes: &[Note]) -> (Vec<f32>, Vec<u32>) {
        let mut features = vec![0.0f32; notes.len() * NOTE_FEATURE_DIM];
        let mut patches = vec![0u32; notes.len()];
        for (i, note) in notes.iter().enumerate() {
            let u = image_u(note.object.translation.0.now());
            let v = image_v(note.object.translation.1.now());
            patches[i] = patch_index(u, v) as u32;
            let feature = self.note_features(note, u, v, frame);
            features[i * NOTE_FEATURE_DIM..(i + 1) * NOTE_FEATURE_DIM].copy_from_slice(&feature);
        }
        (features, patches)
    }

    /// Vision logit, spatial prior and left/right balance decide each hand;
    /// simultaneous notes are spread across both hands.
    fn select_hands(&self, notes: &[Note], logits: &[f32]) -> Vec<Hand> {
        let mut hands = vec![Hand::Left; notes.len()];
        let mut left_run = 0usize;
        let mut right_run = 0usize;
        let mut left_total = 0usize;
        let mut right_total = 0usize;
        let mut previous_time = f32::NEG_INFINITY;
        let mut previous_hand: Option<Hand> = None;

        let mut order: Vec<usize> = (0..notes.len()).collect();
        order.sort_by(|&a, &b| {
            notes[a]
                .time
                .partial_cmp(&notes[b].time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &i in &order {
            let x = notes[i].object.translation.0.now();
            let logit = logits.get(i).copied().unwrap_or(0.0);
            let spatial = if x < -0.05 {
                -1.0
            } else if x > 0.05 {
                1.0
            } else {
                0.0
            };
            let total = (left_total + right_total).max(1) as f32;
            let balance = (right_total as f32 - left_total as f32) / total;
            let score = 0.45 * logit + 0.40 * spatial + 0.15 * balance * 2.0;

            let same_time = (notes[i].time - previous_time).abs() < 1e-3;
            let mut hand = if score < 0.0 { Hand::Left } else { Hand::Right };

            match hand {
                Hand::Left if left_run >= crate::hand::ai::CONSECUTIVE_LIMIT => hand = Hand::Right,
                Hand::Right if right_run >= crate::hand::ai::CONSECUTIVE_LIMIT => hand = Hand::Left,
                _ => {}
            }
            if same_time && previous_hand == Some(hand) {
                hand = match hand {
                    Hand::Left => Hand::Right,
                    Hand::Right => Hand::Left,
                };
            }

            hands[i] = hand;
            previous_hand = Some(hand);
            match hand {
                Hand::Left => {
                    left_total += 1;
                    left_run += 1;
                    right_run = 0;
                }
                Hand::Right => {
                    right_total += 1;
                    right_run += 1;
                    left_run = 0;
                }
            }
            previous_time = notes[i].time;
        }
        hands
    }
}
