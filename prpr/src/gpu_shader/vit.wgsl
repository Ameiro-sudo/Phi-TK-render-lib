// ViT forward for hand assignment: patch embed -> encoder blocks
// (self-attention + FFN) -> per-note cross-attention + classifier.
// Geometry and packed offsets mirror hand/vit.rs and gpu_vit.rs::offs.

const PATCH_SIZE: u32 = 5u;
const PATCH_COLS: u32 = 16u;
const PATCH_ROWS: u32 = 9u;
const PATCH_COUNT: u32 = 144u;
const PATCH_DIM: u32 = 75u;
const MODEL_DIM: u32 = 64u;
const HEAD_COUNT: u32 = 4u;
const HEAD_DIM: u32 = 16u;
const BLOCK_COUNT: u32 = 2u;
const NOTE_FEATURE_DIM: u32 = 16u;
const IMAGE_W: u32 = 80u;
const IMAGE_H: u32 = 45u;
const IMAGE_CHANNELS: u32 = 3u;
const FFN_DIM: u32 = 256u;
const FUSED_DIM: u32 = 128u;

const OFF_PATCH_WEIGHTS: u32 = 0u;
const OFF_PATCH_BIAS: u32 = OFF_PATCH_WEIGHTS + (MODEL_DIM * PATCH_DIM);
const OFF_POSITIONAL: u32 = OFF_PATCH_BIAS + MODEL_DIM;
const OFF_NOTE_WEIGHTS: u32 = OFF_POSITIONAL + (PATCH_COUNT * MODEL_DIM);
const OFF_NOTE_BIAS: u32 = OFF_NOTE_WEIGHTS + (MODEL_DIM * NOTE_FEATURE_DIM);
const BLOCK_BASE: u32 = OFF_NOTE_BIAS + MODEL_DIM;

const B_ATTN_GAIN: u32 = 0u;
const B_QUERY_WEIGHTS: u32 = B_ATTN_GAIN + MODEL_DIM;
const B_QUERY_BIAS: u32 = B_QUERY_WEIGHTS + (MODEL_DIM * MODEL_DIM);
const B_KEY_WEIGHTS: u32 = B_QUERY_BIAS + MODEL_DIM;
const B_KEY_BIAS: u32 = B_KEY_WEIGHTS + (MODEL_DIM * MODEL_DIM);
const B_VALUE_WEIGHTS: u32 = B_KEY_BIAS + MODEL_DIM;
const B_VALUE_BIAS: u32 = B_VALUE_WEIGHTS + (MODEL_DIM * MODEL_DIM);
const B_OUT_WEIGHTS: u32 = B_VALUE_BIAS + MODEL_DIM;
const B_OUT_BIAS: u32 = B_OUT_WEIGHTS + (MODEL_DIM * MODEL_DIM);
const B_FFN_GAIN: u32 = B_OUT_BIAS + MODEL_DIM;
const B_FFN_IN_WEIGHTS: u32 = B_FFN_GAIN + MODEL_DIM;
const B_FFN_IN_BIAS: u32 = B_FFN_IN_WEIGHTS + (FFN_DIM * MODEL_DIM);
const B_FFN_OUT_WEIGHTS: u32 = B_FFN_IN_BIAS + FFN_DIM;
const B_FFN_OUT_BIAS: u32 = B_FFN_OUT_WEIGHTS + (MODEL_DIM * FFN_DIM);
const BLOCK_SIZE: u32 = B_FFN_OUT_BIAS + MODEL_DIM;

const OFF_CLASSIFIER: u32 = BLOCK_BASE + (BLOCK_COUNT * BLOCK_SIZE);

struct VitParams {
    query_count: u32,
    layer: u32,
    final_in_second: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> layer_params: VitParams;
@group(0) @binding(1) var<storage, read> frame: array<f32>;
@group(0) @binding(2) var<storage, read_write> tokens_a: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read> query_features: array<f32>;
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;
@group(0) @binding(6) var<storage, read> query_patches: array<u32>;
@group(0) @binding(7) var<storage, read_write> tokens_b: array<f32>;

fn sanitize(value: f32) -> f32 {
    if (value != value) {
        return 0.0;
    }
    return clamp(value, -50.0, 50.0);
}

// Root-mean-square normalisation: x / sqrt(mean(x^2) + eps) * gain.
fn rmsnorm(x: ptr<function, array<f32, MODEL_DIM>>, gain_offset: u32) {
    var sum_squares = 0.0;
    for (var i = 0u; i < MODEL_DIM; i++) {
        sum_squares += (*x)[i] * (*x)[i];
    }
    let inv_rms = inverseSqrt(sum_squares / f32(MODEL_DIM) + 1e-5);
    for (var i = 0u; i < MODEL_DIM; i++) {
        (*x)[i] = (*x)[i] * inv_rms * weights[gain_offset + i];
    }
}

fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

fn load_token(index: u32, from_tokens_b: bool, out: ptr<function, array<f32, MODEL_DIM>>) {
    for (var i = 0u; i < MODEL_DIM; i++) {
        (*out)[i] = select(tokens_a[index * MODEL_DIM + i], tokens_b[index * MODEL_DIM + i], from_tokens_b);
    }
}

// One patch -> one token: patch embed + learned positional table.
@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
    let patch_id = id.x;
    if (patch_id >= PATCH_COUNT) {
        return;
    }
    let col = patch_id % PATCH_COLS;
    let row = patch_id / PATCH_COLS;
    var raw: array<f32, PATCH_DIM>;
    var i = 0u;
    for (var dy = 0u; dy < PATCH_SIZE; dy++) {
        for (var dx = 0u; dx < PATCH_SIZE; dx++) {
            let x = col * PATCH_SIZE + dx;
            let y = row * PATCH_SIZE + dy;
            let start = (y * IMAGE_W + x) * IMAGE_CHANNELS;
            for (var channel = 0u; channel < IMAGE_CHANNELS; channel++) {
                let index = start + channel;
                if (index < arrayLength(&frame)) {
                    raw[i] = frame[index];
                }
                i++;
            }
        }
    }
    for (var r = 0u; r < MODEL_DIM; r++) {
        var sum = weights[OFF_PATCH_BIAS + r];
        for (var c = 0u; c < PATCH_DIM; c++) {
            sum += weights[OFF_PATCH_WEIGHTS + r * PATCH_DIM + c] * raw[c];
        }
        tokens_a[patch_id * MODEL_DIM + r] = sanitize(sum) + weights[OFF_POSITIONAL + patch_id * MODEL_DIM + r];
    }
}

// Even layer reads tokens_a and writes tokens_b, odd layer the reverse; the
// ping-pong keeps every thread reading its own source buffer.
@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
    let token_index = id.x;
    if (token_index >= PATCH_COUNT) {
        return;
    }
    let base = BLOCK_BASE + layer_params.layer * BLOCK_SIZE;
    let to_tokens_b = (layer_params.layer & 1u) == 0u;
    var x: array<f32, MODEL_DIM>;
    load_token(token_index, to_tokens_b, &x);

    var normalized: array<f32, MODEL_DIM>;
    for (var i = 0u; i < MODEL_DIM; i++) {
        normalized[i] = x[i];
    }
    rmsnorm(&normalized, base + B_ATTN_GAIN);

    var query_token: array<f32, MODEL_DIM>;
    for (var r = 0u; r < MODEL_DIM; r++) {
        var sum = weights[base + B_QUERY_BIAS + r];
        for (var c = 0u; c < MODEL_DIM; c++) {
            sum += weights[base + B_QUERY_WEIGHTS + r * MODEL_DIM + c] * normalized[c];
        }
        query_token[r] = sanitize(sum);
    }

    let scale = inverseSqrt(f32(HEAD_DIM));
    var attended: array<f32, MODEL_DIM>;
    for (var head = 0u; head < HEAD_COUNT; head++) {
        let head_base = head * HEAD_DIM;
        var scores: array<f32, PATCH_COUNT>;
        var max_score = -3.4028235e38;
        for (var j = 0u; j < PATCH_COUNT; j++) {
            var key_source: array<f32, MODEL_DIM>;
            load_token(j, to_tokens_b, &key_source);
            rmsnorm(&key_source, base + B_ATTN_GAIN);
            var key: array<f32, MODEL_DIM>;
            for (var r = 0u; r < MODEL_DIM; r++) {
                var sum = weights[base + B_KEY_BIAS + r];
                for (var c = 0u; c < MODEL_DIM; c++) {
                    sum += weights[base + B_KEY_WEIGHTS + r * MODEL_DIM + c] * key_source[c];
                }
                key[r] = sanitize(sum);
            }
            var dot = 0.0;
            for (var t = 0u; t < HEAD_DIM; t++) {
                dot += query_token[head_base + t] * key[head_base + t];
            }
            scores[j] = dot * scale;
            if (scores[j] > max_score) {
                max_score = scores[j];
            }
        }
        var total = 0.0;
        for (var j = 0u; j < PATCH_COUNT; j++) {
            scores[j] = exp(scores[j] - max_score);
            total += scores[j];
        }
        let inv_total = 1.0 / max(total, 1e-8);
        var sum_head: array<f32, HEAD_DIM>;
        for (var t = 0u; t < HEAD_DIM; t++) {
            sum_head[t] = 0.0;
        }
        for (var j = 0u; j < PATCH_COUNT; j++) {
            var value_source: array<f32, MODEL_DIM>;
            load_token(j, to_tokens_b, &value_source);
            rmsnorm(&value_source, base + B_ATTN_GAIN);
            var value: array<f32, MODEL_DIM>;
            for (var r = 0u; r < MODEL_DIM; r++) {
                var sum = weights[base + B_VALUE_BIAS + r];
                for (var c = 0u; c < MODEL_DIM; c++) {
                    sum += weights[base + B_VALUE_WEIGHTS + r * MODEL_DIM + c] * value_source[c];
                }
                value[r] = sanitize(sum);
            }
            let score = scores[j] * inv_total;
            for (var t = 0u; t < HEAD_DIM; t++) {
                sum_head[t] += score * value[head_base + t];
            }
        }
        for (var t = 0u; t < HEAD_DIM; t++) {
            attended[head_base + t] = sum_head[t];
        }
    }

    for (var r = 0u; r < MODEL_DIM; r++) {
        var sum = weights[base + B_OUT_BIAS + r];
        for (var c = 0u; c < MODEL_DIM; c++) {
            sum += weights[base + B_OUT_WEIGHTS + r * MODEL_DIM + c] * attended[c];
        }
        x[r] += sanitize(sum);
    }

    var ffn_input: array<f32, MODEL_DIM>;
    for (var i = 0u; i < MODEL_DIM; i++) {
        ffn_input[i] = x[i];
    }
    rmsnorm(&ffn_input, base + B_FFN_GAIN);

    var hidden: array<f32, FFN_DIM>;
    for (var r = 0u; r < FFN_DIM; r++) {
        var sum = weights[base + B_FFN_IN_BIAS + r];
        for (var c = 0u; c < MODEL_DIM; c++) {
            sum += weights[base + B_FFN_IN_WEIGHTS + r * MODEL_DIM + c] * ffn_input[c];
        }
        hidden[r] = gelu(sanitize(sum));
    }
    for (var r = 0u; r < MODEL_DIM; r++) {
        var sum = weights[base + B_FFN_OUT_BIAS + r];
        for (var c = 0u; c < FFN_DIM; c++) {
            sum += weights[base + B_FFN_OUT_WEIGHTS + r * FFN_DIM + c] * hidden[c];
        }
        x[r] += sanitize(sum);
    }

    for (var i = 0u; i < MODEL_DIM; i++) {
        if (x[i] != x[i]) {
            x[i] = 0.0;
        }
        if (to_tokens_b) {
            tokens_b[token_index * MODEL_DIM + i] = x[i];
        } else {
            tokens_a[token_index * MODEL_DIM + i] = x[i];
        }
    }
}

// Per-note query: cross-attend over the visual tokens, then classify the fused
// query/context pair. `layer_params.final_in_second` selects the final buffer.
@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) id: vec3<u32>) {
    let query_index = id.x;
    if (query_index >= layer_params.query_count
        || query_index * NOTE_FEATURE_DIM >= arrayLength(&query_features)) {
        return;
    }
    let patch_id = query_patches[query_index];
    let feature_base = query_index * NOTE_FEATURE_DIM;
    let in_second = layer_params.final_in_second == 1u;

    var features: array<f32, NOTE_FEATURE_DIM>;
    for (var i = 0u; i < NOTE_FEATURE_DIM; i++) {
        features[i] = query_features[feature_base + i];
    }

    var query_token: array<f32, MODEL_DIM>;
    for (var r = 0u; r < MODEL_DIM; r++) {
        var sum = weights[OFF_NOTE_BIAS + r];
        for (var c = 0u; c < NOTE_FEATURE_DIM; c++) {
            sum += weights[OFF_NOTE_WEIGHTS + r * NOTE_FEATURE_DIM + c] * features[c];
        }
        query_token[r] = sanitize(sum) + weights[OFF_POSITIONAL + patch_id * MODEL_DIM + r] * 0.15;
    }

    var context: array<f32, MODEL_DIM>;
    for (var i = 0u; i < MODEL_DIM; i++) {
        context[i] = query_token[i];
    }

    for (var layer = 0u; layer < BLOCK_COUNT; layer++) {
        let base = BLOCK_BASE + layer * BLOCK_SIZE;
        var normalized: array<f32, MODEL_DIM>;
        for (var i = 0u; i < MODEL_DIM; i++) {
            normalized[i] = query_token[i];
        }
        rmsnorm(&normalized, base + B_ATTN_GAIN);

        var projected_query: array<f32, MODEL_DIM>;
        for (var r = 0u; r < MODEL_DIM; r++) {
            var sum = weights[base + B_QUERY_BIAS + r];
            for (var c = 0u; c < MODEL_DIM; c++) {
                sum += weights[base + B_QUERY_WEIGHTS + r * MODEL_DIM + c] * normalized[c];
            }
            projected_query[r] = sanitize(sum);
        }

        let scale = inverseSqrt(f32(HEAD_DIM));
        var attention_out: array<f32, MODEL_DIM>;
        for (var head = 0u; head < HEAD_COUNT; head++) {
            let head_base = head * HEAD_DIM;
            var scores: array<f32, PATCH_COUNT>;
            var max_score = -3.4028235e38;
            for (var j = 0u; j < PATCH_COUNT; j++) {
                var key: array<f32, MODEL_DIM>;
                for (var r = 0u; r < MODEL_DIM; r++) {
                    var sum = weights[base + B_KEY_BIAS + r];
                    for (var c = 0u; c < MODEL_DIM; c++) {
                        let token = select(tokens_a[j * MODEL_DIM + c], tokens_b[j * MODEL_DIM + c], in_second);
                        sum += weights[base + B_KEY_WEIGHTS + r * MODEL_DIM + c] * token;
                    }
                    key[r] = sanitize(sum);
                }
                var dot = 0.0;
                for (var t = 0u; t < HEAD_DIM; t++) {
                    dot += projected_query[head_base + t] * key[head_base + t];
                }
                scores[j] = dot * scale;
                if (scores[j] > max_score) {
                    max_score = scores[j];
                }
            }
            var total = 0.0;
            for (var j = 0u; j < PATCH_COUNT; j++) {
                scores[j] = exp(scores[j] - max_score);
                total += scores[j];
            }
            let inv_total = 1.0 / max(total, 1e-8);
            var sum_head: array<f32, HEAD_DIM>;
            for (var t = 0u; t < HEAD_DIM; t++) {
                sum_head[t] = 0.0;
            }
            for (var j = 0u; j < PATCH_COUNT; j++) {
                var value: array<f32, MODEL_DIM>;
                for (var r = 0u; r < MODEL_DIM; r++) {
                    var sum = weights[base + B_VALUE_BIAS + r];
                    for (var c = 0u; c < MODEL_DIM; c++) {
                        let token = select(tokens_a[j * MODEL_DIM + c], tokens_b[j * MODEL_DIM + c], in_second);
                        sum += weights[base + B_VALUE_WEIGHTS + r * MODEL_DIM + c] * token;
                    }
                    value[r] = sanitize(sum);
                }
                let score = scores[j] * inv_total;
                for (var t = 0u; t < HEAD_DIM; t++) {
                    sum_head[t] += score * value[head_base + t];
                }
            }
            for (var t = 0u; t < HEAD_DIM; t++) {
                attention_out[head_base + t] = sum_head[t];
            }
        }

        var projected: array<f32, MODEL_DIM>;
        for (var r = 0u; r < MODEL_DIM; r++) {
            var sum = weights[base + B_OUT_BIAS + r];
            for (var c = 0u; c < MODEL_DIM; c++) {
                sum += weights[base + B_OUT_WEIGHTS + r * MODEL_DIM + c] * attention_out[c];
            }
            projected[r] = sanitize(sum);
        }

        for (var i = 0u; i < MODEL_DIM; i++) {
            query_token[i] = query_token[i] * 0.5 + (query_token[i] + projected[i]) * 0.5;
            context[i] = query_token[i];
        }
    }

    var fused: array<f32, FUSED_DIM>;
    for (var i = 0u; i < MODEL_DIM; i++) {
        fused[i] = query_token[i];
        fused[MODEL_DIM + i] = context[i];
    }
    var logit = weights[OFF_CLASSIFIER + FUSED_DIM];
    for (var c = 0u; c < FUSED_DIM; c++) {
        logit += weights[OFF_CLASSIFIER + c] * fused[c];
    }
    if (logit != logit) {
        logit = 0.0;
    }
    logits[query_index] = logit;
}
