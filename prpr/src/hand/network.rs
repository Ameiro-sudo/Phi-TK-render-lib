use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(crate) enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Swish,
    GELU,
    Linear,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct NetworkLayer {
    #[serde(default)]
    pub weights: Vec<Vec<f32>>,
    #[serde(default)]
    pub biases: Vec<f32>,
    #[serde(default)]
    pub activations: Vec<f32>,
    #[serde(default)]
    pub pre_activations: Vec<f32>,
    #[serde(default)]
    pub mom_weights: Vec<Vec<f32>>,
    #[serde(default)]
    pub mom_biases: Vec<f32>,
    pub activation_func: ActivationFunction,
}

/// Column-wise policy/value head: one left/right bit per image column plus a
/// scalar value estimate, updated by the background PPO trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DeepNeuralNetwork {
    pub layers: Vec<NetworkLayer>,
    #[serde(default)]
    pub value_weights: Vec<f32>,
    #[serde(default)]
    pub value_bias: f32,
    #[serde(default)]
    pub lr: f32,
    #[serde(default)]
    pub mom: f32,
    pub epoch: u64,
    #[serde(default)]
    pub last_loss: f32,
    #[serde(default)]
    pub bad_epochs: usize,
}

impl DeepNeuralNetwork {
    pub fn new() -> Self {
        let mut network = Self {
            layers: Vec::new(),
            value_weights: Vec::new(),
            value_bias: 0.0,
            lr: 0.0003,
            mom: 0.9,
            epoch: 0,
            last_loss: f32::INFINITY,
            bad_epochs: 0,
        };
        network.build_architecture();
        network
    }

    pub fn validate(&self) -> bool {
        self.layers.iter().all(|layer| {
            layer.weights.iter().flatten().all(|w| w.is_finite())
                && layer.biases.iter().all(|b| b.is_finite())
        }) && self.lr.is_finite()
            && self.mom.is_finite()
            && self.value_weights.iter().all(|w| w.is_finite())
            && self.value_bias.is_finite()
    }

    pub fn activate(x: f32, func: &ActivationFunction) -> f32 {
        let x = x.clamp(-50.0, 50.0);
        let result = match func {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Sigmoid => Self::sigmoid(x),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Swish => x * Self::sigmoid(x),
            ActivationFunction::GELU => {
                if x.abs() < 3.0 {
                    let a = 0.7978845608 * (x + 0.044715 * x * x * x);
                    0.5 * x * (1.0 + a.tanh())
                } else {
                    x.max(0.0)
                }
            }
            ActivationFunction::Linear => x,
        };
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }

    pub fn activate_derivative_from_z(x: f32, func: &ActivationFunction) -> f32 {
        let x = x.clamp(-100.0, 100.0);
        match func {
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::Sigmoid => {
                let s = Self::sigmoid(x);
                (s * (1.0 - s)).clamp(0.0, 1.0)
            }
            ActivationFunction::Tanh => {
                let t = x.tanh();
                (1.0 - t * t).clamp(0.0, 1.0)
            }
            ActivationFunction::Swish => {
                let s = if x > 10.0 { 1.0 } else if x < -10.0 { 0.0 } else { Self::sigmoid(x) };
                (s + x * s * (1.0 - s)).clamp(-10.0, 10.0)
            }
            ActivationFunction::GELU => {
                let a = 0.7978845608 * (x + 0.044715 * x * x * x);
                let ta = a.tanh();
                let sech2 = 1.0 - ta * ta;
                let ap = 0.7978845608 * (1.0 + 0.134145 * x * x);
                (0.5 * (1.0 + ta) + 0.5 * x * sech2 * ap).clamp(-10.0, 10.0)
            }
            ActivationFunction::Linear => 1.0,
        }
    }

    fn sigmoid(x: f32) -> f32 {
        if x > 20.0 {
            1.0
        } else if x < -20.0 {
            0.0
        } else {
            1.0 / (1.0 + (-x).exp())
        }
    }

    fn build_architecture(&mut self) {
        let input_dim = super::AI_IMAGE_SIZE;
        let out_dim = super::AI_IMAGE_W;
        self.add_dense(input_dim, 512, ActivationFunction::ReLU);
        self.add_dense(512, 256, ActivationFunction::ReLU);
        self.add_dense(256, 128, ActivationFunction::ReLU);
        self.add_dense(128, out_dim, ActivationFunction::Linear);
        let scale = (1.0 / out_dim as f32).sqrt();
        self.value_weights = (0..out_dim).map(|_| (fastrand::f32() * 2.0 - 1.0) * scale).collect();
        self.value_bias = 0.0;
    }

    fn add_dense(&mut self, input_size: usize, output_size: usize, activation: ActivationFunction) {
        let scale = match activation {
            ActivationFunction::ReLU | ActivationFunction::GELU | ActivationFunction::Swish => {
                (2.0 / input_size as f32).sqrt()
            }
            _ => (1.0 / input_size as f32).sqrt(),
        };
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| (fastrand::f32() * 2.0 - 1.0) * scale).collect())
            .collect();
        self.layers.push(NetworkLayer {
            weights,
            biases: vec![0.0; output_size],
            activations: vec![0.0; output_size],
            pre_activations: vec![0.0; output_size],
            mom_weights: vec![vec![0.0; input_size]; output_size],
            mom_biases: vec![0.0; output_size],
            activation_func: activation,
        });
    }

    fn dense_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; layer.weights.len()];
        let mut pre_activations = vec![0.0; layer.weights.len()];
        layer
            .weights
            .par_iter()
            .zip(layer.biases.par_iter())
            .zip(pre_activations.par_iter_mut())
            .zip(output.par_iter_mut())
            .for_each(|(((weights, bias), pre_activation), value)| {
                let mut sum = *bias;
                for (weight, input) in weights.iter().zip(input.iter()) {
                    if weight.is_finite() && input.is_finite() {
                        sum += weight * input;
                    }
                }
                if !sum.is_finite() {
                    sum = 0.0;
                }
                sum = sum.clamp(-100.0, 100.0);
                *pre_activation = sum;
                *value = Self::activate(sum, &layer.activation_func);
                if !value.is_finite() {
                    *value = 0.0;
                }
            });
        layer.pre_activations = pre_activations;
        layer.activations = output.clone();
        output
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let sanitized: Vec<f32> = input.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect();
        let mut activations = sanitized;
        for layer in self.layers.iter_mut() {
            activations = Self::dense_forward(layer, &activations);
        }
        activations
    }

    pub fn value_from(&self, output: &[f32]) -> f32 {
        if self.value_weights.len() != output.len() {
            return output.iter().sum::<f32>() / output.len().max(1) as f32;
        }
        let mut value = self.value_bias;
        for (weight, out) in self.value_weights.iter().zip(output.iter()) {
            value += weight * out;
        }
        value
    }

    pub fn backward(
        &mut self,
        input: &[f32],
        output_gradients: &[f32],
        weight_gradients: &mut [Vec<Vec<f32>>],
        bias_gradients: &mut [Vec<f32>],
    ) {
        let layer_count = self.layers.len();
        if layer_count == 0 {
            return;
        }
        let mut errors = vec![vec![0.0; 0]; layer_count];

        let last = layer_count - 1;
        let mut output_error: Vec<f32> = output_gradients.to_vec();
        for (i, error) in output_error.iter_mut().enumerate() {
            let pre_activation = self.layers[last].pre_activations.get(i).copied().unwrap_or(0.0);
            *error *= Self::activate_derivative_from_z(pre_activation, &self.layers[last].activation_func);
        }
        if !output_error.iter().all(|e| e.is_finite()) {
            return;
        }
        errors[last] = output_error;

        for layer_index in (0..layer_count).rev() {
            if layer_index < layer_count - 1 {
                let next = &self.layers[layer_index + 1];
                let mut error = vec![0.0; self.layers[layer_index].activations.len()];
                for (i, value) in error.iter_mut().enumerate() {
                    for (j, &next_error) in errors[layer_index + 1].iter().enumerate() {
                        if j < next.weights.len() && i < next.weights[j].len() {
                            *value += next_error * next.weights[j][i];
                        }
                    }
                }
                let func = &self.layers[layer_index].activation_func;
                let has_pre_activation = self.layers[layer_index].pre_activations.len() >= error.len();
                for (i, value) in error.iter_mut().enumerate() {
                    let derivative = if has_pre_activation {
                        Self::activate_derivative_from_z(self.layers[layer_index].pre_activations[i], func)
                    } else {
                        1.0
                    };
                    *value *= derivative;
                }
                if !error.iter().all(|e| e.is_finite()) {
                    return;
                }
                errors[layer_index] = error;
            }

            let previous: &[f32] = if layer_index > 0 {
                &self.layers[layer_index - 1].activations
            } else {
                input
            };
            for (i, &error) in errors[layer_index].iter().enumerate() {
                if i >= weight_gradients[layer_index].len() {
                    continue;
                }
                let common = previous.len().min(weight_gradients[layer_index][i].len());
                for j in 0..common {
                    weight_gradients[layer_index][i][j] += error * previous[j];
                }
                if i < bias_gradients[layer_index].len() {
                    bias_gradients[layer_index][i] += error;
                }
            }
        }
    }

    fn clip_gradients(weight_gradients: &mut [Vec<Vec<f32>>], bias_gradients: &mut [Vec<f32>], max_norm: f32) {
        let mut total = 0.0f32;
        for gradient in weight_gradients.iter() {
            for row in gradient {
                for value in row {
                    total += value * value;
                }
            }
        }
        for gradient in bias_gradients.iter() {
            for value in gradient {
                total += value * value;
            }
        }
        let norm = total.sqrt();
        if norm > max_norm {
            let scale = max_norm / (norm + 1e-7);
            for gradient in weight_gradients.iter_mut() {
                for row in gradient.iter_mut() {
                    for value in row.iter_mut() {
                        *value *= scale;
                    }
                }
            }
            for gradient in bias_gradients.iter_mut() {
                for value in gradient.iter_mut() {
                    *value *= scale;
                }
            }
        }
    }

    fn update_weights_with_momentum(
        &mut self,
        weight_gradients: &[Vec<Vec<f32>>],
        bias_gradients: &[Vec<f32>],
    ) {
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            if layer_index >= weight_gradients.len() {
                continue;
            }
            for i in 0..layer.weights.len() {
                if i >= weight_gradients[layer_index].len() {
                    continue;
                }
                for j in 0..layer.weights[i].len() {
                    if j >= weight_gradients[layer_index][i].len() {
                        continue;
                    }
                    layer.mom_weights[i][j] = self.mom * layer.mom_weights[i][j] + weight_gradients[layer_index][i][j];
                    layer.weights[i][j] -= self.lr * layer.mom_weights[i][j];
                }
            }
            for i in 0..layer.biases.len() {
                if i >= bias_gradients[layer_index].len() {
                    continue;
                }
                layer.mom_biases[i] = self.mom * layer.mom_biases[i] + bias_gradients[layer_index][i];
                layer.biases[i] -= self.lr * layer.mom_biases[i];
            }
        }
    }

    pub(crate) fn train_with_ppo(
        &mut self,
        experiences: &[crate::hand::Experience],
        epochs: usize,
        budget: Duration,
    ) {
        if experiences.is_empty() || epochs == 0 {
            return;
        }
        const MINI_BATCH: usize = 32;
        const GRAD_CLIP: f32 = 0.5;

        let deadline = if budget.is_zero() {
            None
        } else {
            Some(std::time::Instant::now() + budget)
        };
        let expired = || deadline.map_or(false, |deadline| std::time::Instant::now() >= deadline);

        for layer in &mut self.layers {
            if layer.mom_weights.len() != layer.weights.len() {
                layer.mom_weights = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
            }
            if layer.mom_biases.len() != layer.biases.len() {
                layer.mom_biases = vec![0.0; layer.biases.len()];
            }
        }

        let output_dim = super::AI_IMAGE_W;
        let mut total_loss = 0.0f32;
        let mut total_samples = 0usize;

        'epochs: for _ in 0..epochs {
            if expired() {
                break;
            }
            let mut order: Vec<usize> = (0..experiences.len()).collect();
            fastrand::shuffle(&mut order);

            for chunk in order.chunks(MINI_BATCH) {
                if expired() {
                    break 'epochs;
                }
                let mut weight_gradients: Vec<Vec<Vec<f32>>> = self
                    .layers
                    .iter()
                    .map(|layer| vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()])
                    .collect();
                let mut bias_gradients: Vec<Vec<f32>> =
                    self.layers.iter().map(|layer| vec![0.0; layer.biases.len()]).collect();
                let mut value_weight_gradients = vec![0.0f32; output_dim];
                let mut value_bias_gradient = 0.0f32;

                let mut policy_loss = 0.0f32;
                let mut value_loss = 0.0f32;
                let mut batch_samples = 0usize;

                for &i in chunk {
                    if expired() {
                        break 'epochs;
                    }
                    let experience = &experiences[i];
                    let output = self.forward(&experience.state);
                    let value_prediction = self.value_from(&output);
                    let value_error = value_prediction - experience.return_;
                    let mut output_gradient = vec![0.0f32; output.len()];

                    for (j, &logit) in output.iter().enumerate() {
                        let action = if experience.actions.get(j).copied().unwrap_or(0) > 0 {
                            1.0f32
                        } else {
                            0.0f32
                        };
                        let new_prob = Self::sigmoid(logit).clamp(1e-6, 1.0 - 1e-6);
                        let old_logit = experience.old_out.get(j).copied().unwrap_or(logit);
                        let old_prob = Self::sigmoid(old_logit).clamp(1e-6, 1.0 - 1e-6);

                        let log_ratio = (action * (new_prob / old_prob).ln()
                            + (1.0 - action) * ((1.0 - new_prob) / (1.0 - old_prob)).ln())
                        .clamp(-20.0, 20.0);
                        let ratio = log_ratio.exp();
                        let unclipped = ratio * experience.advantage;
                        let clipped = ratio.clamp(1.0 - super::ai::PPO_EPS, 1.0 + super::ai::PPO_EPS)
                            * experience.advantage;
                        policy_loss -= unclipped.min(clipped);

                        // Zero the gradient when the clipped branch wins, as PPO requires.
                        if unclipped <= clipped {
                            output_gradient[j] = -experience.advantage * ratio * (action - new_prob);
                        }
                    }

                    value_loss += value_error * value_error;
                    if self.value_weights.len() == output.len() {
                        let factor = 2.0 * super::ai::PPO_VF_COEF * value_error;
                        for (gradient, out) in value_weight_gradients.iter_mut().zip(output.iter()) {
                            *gradient += factor * out;
                        }
                        value_bias_gradient += factor;
                    }

                    self.backward(&experience.state, &output_gradient, &mut weight_gradients, &mut bias_gradients);
                    batch_samples += 1;
                    total_samples += 1;
                }

                if batch_samples == 0 {
                    break 'epochs;
                }
                let scale = 1.0 / batch_samples as f32;
                for gradient in weight_gradients.iter_mut() {
                    for row in gradient.iter_mut() {
                        for value in row.iter_mut() {
                            *value *= scale;
                        }
                    }
                }
                for gradient in bias_gradients.iter_mut() {
                    for value in gradient.iter_mut() {
                        *value *= scale;
                    }
                }
                for value in value_weight_gradients.iter_mut() {
                    *value *= scale;
                }
                value_bias_gradient *= scale;

                Self::clip_gradients(&mut weight_gradients, &mut bias_gradients, GRAD_CLIP);
                self.update_weights_with_momentum(&weight_gradients, &bias_gradients);

                if self.value_weights.len() == output_dim {
                    let value_lr = self.lr * 0.5;
                    for j in 0..output_dim {
                        self.value_weights[j] -= value_lr * value_weight_gradients[j];
                    }
                    self.value_bias -= value_lr * value_bias_gradient;
                }

                total_loss += (policy_loss + value_loss) * scale;
            }
        }

        if total_samples > 0 {
            self.adapt_lr(total_loss / total_samples as f32);
            self.epoch += 1;
        }
    }

    pub(crate) fn adapt_lr(&mut self, loss: f32) {
        let loss_change = if self.last_loss.is_finite() && self.last_loss > 1e-8 {
            (loss / self.last_loss).clamp(0.1, 10.0)
        } else {
            1.0
        };
        let epoch_factor = if self.epoch < 50 {
            0.9
        } else if self.epoch < 200 {
            1.0
        } else {
            0.7
        };

        if loss_change < 0.98 {
            self.lr = (self.lr * 1.015 * epoch_factor).min(0.02);
            self.bad_epochs = 0;
        } else if loss_change > 1.02 {
            self.lr = (self.lr * 0.90).max(0.0001);
            self.bad_epochs += 1;
            if self.bad_epochs >= 3 {
                self.lr = (self.lr * 0.7).max(0.00005);
            }
            if self.bad_epochs >= 8 {
                self.bad_epochs = 0;
                self.lr = 0.002;
            }
        } else {
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * (self.epoch % 200) as f32 / 200.0).cos());
            self.lr = 0.0008 * cosine * epoch_factor;
            self.bad_epochs = 0;
        }
        self.lr = self.lr.clamp(0.0001, 0.02);
        self.last_loss = if loss.is_finite() { loss } else { self.last_loss };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn experiences(count: usize) -> Vec<crate::hand::Experience> {
        (0..count)
            .map(|i| crate::hand::Experience {
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
    fn ppo_epochs_zero_is_noop() {
        let mut network = DeepNeuralNetwork::new();
        let epoch = network.epoch;
        let lr = network.lr;
        network.train_with_ppo(&experiences(64), 0, Duration::ZERO);
        assert_eq!(network.epoch, epoch, "epochs=0 must not count an update");
        assert_eq!(network.lr, lr, "epochs=0 must not touch the optimizer");
    }

    #[test]
    fn ppo_budget_stops_early() {
        let mut network = DeepNeuralNetwork::new();
        let start = Instant::now();
        network.train_with_ppo(&experiences(64), 100_000, Duration::from_millis(50));
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_secs(5), "budget was ignored, ran for {elapsed:?}");
        assert!(network.epoch >= 1, "at least one partial update should land");
        assert!(network.validate());
    }

    #[test]
    fn ppo_updates_weights() {
        let mut network = DeepNeuralNetwork::new();
        let before = network.layers[0].weights[0][0];
        network.train_with_ppo(&experiences(64), 1, Duration::ZERO);
        let after = network.layers[0].weights[0][0];
        assert_ne!(before, after, "a full epoch must change the parameters");
        assert!(network.validate());
    }
}
