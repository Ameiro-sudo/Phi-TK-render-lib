use crate::gpu_utils;

use fastrand;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use wgpu;
use wgpu::util::DeviceExt;

pub(crate) fn default_gnorm() -> f32 { 5.0_f32 }

impl crate::hand_model::Vector2 {
    pub fn clean(&mut self) {
        if !self.x.is_finite() { self.x = 0.0; }
        if !self.y.is_finite() { self.y = 0.0; }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) enum LayerType {
    Dense,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
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
    pub layer_type: LayerType,
    pub activation_func: ActivationFunction,
    #[serde(skip)]
    pub weights_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    pub biases_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    pub activations_buffer: Option<wgpu::Buffer>,
}

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
    #[serde(default)]
    pub drop: f32,
    pub bs: usize,
    #[serde(default = "default_gnorm")]
    pub gnorm: f32,
    #[serde(default)]
    pub last_loss: f32,
    #[serde(default)]
    pub bad_ep: usize,
    pub ep: u64,
    #[serde(skip)]
    pub device: Option<wgpu::Device>,
    #[serde(skip)]
    pub queue: Option<wgpu::Queue>,
    #[serde(skip)]
    pub gpu_exec: Option<Arc<gpu_utils::GpuNetworkExecutor>>,
    #[serde(skip)]
    pub gpu_ok: bool,
    #[serde(skip)]
    pub bs_buf: Option<wgpu::Buffer>,
    #[serde(skip)]
    pub init_try: bool,
    #[serde(skip)]
    pub init_fail: bool,
}

impl DeepNeuralNetwork {
    pub fn new() -> Self {
        let mut net = Self {
            layers: Vec::new(),
            value_weights: Vec::new(),
            value_bias: 0.0,
            lr: 0.0003,
            mom: 0.9,
            drop: 0.0,
            bs: 64,
            gnorm: 0.5,
            last_loss: f32::INFINITY,
            bad_ep: 0,
            ep: 0,
            device: None,
            queue: None,
            gpu_exec: None,
            gpu_ok: false,
            bs_buf: None,
            init_try: false,
            init_fail: false,
        };
        net.build_architecture();
        net
    }

    #[allow(dead_code)]
    pub(crate) fn clean(&mut self) {
        for layer in &mut self.layers {
            for w in layer.weights.iter_mut().flatten() {
                if !w.is_finite() { *w = 0.0; }
            }
            for b in &mut layer.biases {
                if !b.is_finite() { *b = 0.0; }
            }
        }
        if !self.lr.is_finite() { self.lr = 0.0003; }
        if !self.mom.is_finite() { self.mom = 0.9; }
    }

    pub fn validate(&self) -> bool {
        for layer in &self.layers {
            for w in layer.weights.iter().flatten() {
                if !w.is_finite() { return false; }
            }
            for b in &layer.biases {
                if !b.is_finite() { return false; }
            }
        }
        self.lr.is_finite() && self.mom.is_finite()
    }

    pub fn activate(x: f32, func: &ActivationFunction) -> f32 {
        let x = x.clamp(-50.0, 50.0);
        let r = match func {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Sigmoid => {
                if x > 20.0 { 1.0 } else if x < -20.0 { 0.0 } else { 1.0 / (1.0 + (-x).exp()) }
            }
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Swish => {
                let s = if x > 20.0 { 1.0 } else if x < -20.0 { 0.0 } else { 1.0 / (1.0 + (-x).exp()) };
                x * s
            }
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
        if r.is_finite() { r } else { 0.0 }
    }

    pub fn activate_derivative_from_z(x: f32, func: &ActivationFunction) -> f32 {
        let x = x.clamp(-100.0, 100.0);
        match func {
            ActivationFunction::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFunction::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                (s * (1.0 - s)).clamp(0.0, 1.0)
            }
            ActivationFunction::Tanh => {
                let t = x.tanh();
                (1.0 - t * t).clamp(0.0, 1.0)
            }
            ActivationFunction::Swish => {
                let s = if x > 10.0 { 1.0 } else if x < -10.0 { 0.0 } else { 1.0 / (1.0 + (-x).exp()) };
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

    fn build_architecture(&mut self) {
        let input_dim = super::AI_IMAGE_SIZE;
        let out_dim = super::AI_IMAGE_W;
        self.add_dense(input_dim, 512, ActivationFunction::ReLU);
        self.add_dense(512, 256, ActivationFunction::ReLU);
        self.add_dense(256, 128, ActivationFunction::ReLU);
        self.add_dense(128, out_dim, ActivationFunction::Linear);
        let std_dev = (1.0 / out_dim as f32).sqrt();
        self.value_weights = (0..out_dim).map(|_| (fastrand::f32() * 2.0 - 1.0) * std_dev).collect();
        self.value_bias = 0.0;
    }

    fn add_dense(&mut self, in_sz: usize, out_sz: usize, act: ActivationFunction) {
        let std_dev = match act {
            ActivationFunction::ReLU | ActivationFunction::GELU | ActivationFunction::Swish => (2.0 / in_sz as f32).sqrt(),
            _ => (1.0 / in_sz as f32).sqrt(),
        };
        let mut weights = Vec::with_capacity(out_sz);
        let mut mom_w = Vec::with_capacity(out_sz);
        for _ in 0..out_sz {
            let mut row = Vec::with_capacity(in_sz);
            let mut mrow = Vec::with_capacity(in_sz);
            for _ in 0..in_sz {
                row.push((fastrand::f32() * 2.0 - 1.0) * std_dev);
                mrow.push(0.0);
            }
            weights.push(row);
            mom_w.push(mrow);
        }
        self.layers.push(NetworkLayer {
            weights,
            biases: vec![0.0; out_sz],
            activations: vec![0.0; out_sz],
            pre_activations: vec![0.0; out_sz],
            mom_weights: mom_w,
            mom_biases: vec![0.0; out_sz],
            layer_type: LayerType::Dense,
            activation_func: act,
            weights_buffer: None,
            biases_buffer: None,
            activations_buffer: None,
        });
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn dense_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; layer.weights.len()];
        let mut zs = vec![0.0; layer.weights.len()];
        layer.weights.par_iter()
            .zip(layer.biases.par_iter())
            .zip(zs.par_iter_mut())
            .zip(output.par_iter_mut())
            .for_each(|(((w, b), z), o)| {
                let mut sum = *b;
                for (wi, xi) in w.iter().zip(input.iter()) {
                    if wi.is_finite() && xi.is_finite() { sum += wi * xi; }
                }
                if !sum.is_finite() { sum = 0.0; }
                sum = sum.clamp(-100.0, 100.0);
                *z = sum;
                *o = DeepNeuralNetwork::activate(sum, &layer.activation_func);
                if !o.is_finite() { *o = 0.0; }
            });
        layer.pre_activations = zs;
        layer.activations = output.clone();
        output
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let ci: Vec<f32> = input.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect();
        if self.gpu_ok && self.device.is_some() && self.gpu_exec.is_some() {
            return self.gpu_forward(&ci);
        }
        self.cpu_forward(&ci)
    }

    pub fn value_head(&mut self, input: &[f32]) -> f32 {
        let out = self.forward(input);
        if self.value_weights.len() != out.len() {
            return out.iter().sum::<f32>() / out.len().max(1) as f32;
        }
        let mut v = self.value_bias;
        for (w, o) in self.value_weights.iter().zip(out.iter()) {
            v += w * o;
        }
        v
    }

    pub(crate) fn cpu_forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut ci = input.to_vec();
        for layer in self.layers.iter_mut() {
            for x in &mut ci { if !x.is_finite() { *x = 0.0; } }
            ci = Self::dense_forward(layer, &ci);
        }
        ci
    }

    pub(crate) fn gpu_forward(&mut self, input: &[f32]) -> Vec<f32> {
        if !self.gpu_ok { self.init_gpu_sync(); }
        if !self.gpu_ok || self.device.is_none() || self.queue.is_none() || self.gpu_exec.is_none() {
            return self.cpu_forward(input);
        }
        if let Some(exec) = self.gpu_exec.clone() {
            let gpu_layers: Vec<gpu_utils::NetworkLayerGPU> = self.layers.iter().map(|l| {
                gpu_utils::NetworkLayerGPU {
                    weights_flattened: l.weights.iter().flatten().cloned().collect(),
                    biases: l.biases.clone(),
                    output_size: l.weights.len(),
                    seq_len: 1,
                    layer_type: gpu_utils::LayerTypeGPU::Dense,
                    num_inputs: 0,
                }
            }).collect();
            if gpu_layers.is_empty() { return self.cpu_forward(input); }
            let rs = self.layers.last().unwrap().activations.len();
            let r = exec.execute_network_forward(input, &gpu_layers, rs);
            if let Some(last) = self.layers.last_mut() { last.activations = r.clone(); }
            r
        } else {
            self.cpu_forward(input)
        }
    }

    pub fn backward(&mut self, input: &[f32], output_gradients: &[f32], total_gradients: &mut [Vec<Vec<f32>>], total_bias_gradients: &mut [Vec<f32>]) {
        let nl = self.layers.len();
        let mut errors = vec![vec![0.0; 0]; nl];

        let last = nl - 1;
        let mut oe: Vec<f32> = output_gradients.to_vec();
        for i in 0..oe.len() {
            let z = if i < self.layers[last].pre_activations.len() { self.layers[last].pre_activations[i] } else { 0.0 };
            oe[i] *= Self::activate_derivative_from_z(z, &self.layers[last].activation_func);
        }
        if !oe.iter().all(|e| e.is_finite()) { return; }
        errors[last] = oe;

        for li in (0..nl).rev() {
            if li < nl - 1 {
                let next = &self.layers[li + 1];
                let mut ce = vec![0.0; self.layers[li].activations.len()];
                for i in 0..ce.len() {
                    for (j, &e) in errors[li + 1].iter().enumerate() {
                        if j < next.weights.len() && i < next.weights[j].len() {
                            ce[i] += e * next.weights[j][i];
                        }
                    }
                }
                let func = &self.layers[li].activation_func;
                let has_z = self.layers[li].pre_activations.len() >= ce.len();
                for i in 0..ce.len() {
                    let d = if has_z { Self::activate_derivative_from_z(self.layers[li].pre_activations[i], func) } else { 1.0 };
                    ce[i] *= d;
                }
                if !ce.iter().all(|e| e.is_finite()) { return; }
                errors[li] = ce;
            }

            let prev: &[f32] = if li > 0 { &self.layers[li - 1].activations } else { input };
            for (i, &e) in errors[li].iter().enumerate() {
                if i >= total_gradients[li].len() { continue; }
                let common = prev.len().min(total_gradients[li][i].len());
                for j in 0..common { total_gradients[li][i][j] += e * prev[j]; }
                if i < total_bias_gradients[li].len() { total_bias_gradients[li][i] += e; }
            }
        }
    }

    pub(crate) fn clip_gradients(&self, gradients: &mut [Vec<Vec<f32>>], bias_gradients: &mut [Vec<f32>], max_norm: f32) {
        let mut total_sq: f32 = 0.0;
        for g in gradients.iter() { for row in g { for &v in row { total_sq += v * v; } } }
        for g in bias_gradients.iter() { for &v in g { total_sq += v * v; } }
        let norm = total_sq.sqrt();
        if norm > max_norm {
            let s = max_norm / (norm + 1e-7);
            for g in gradients.iter_mut() { for row in g.iter_mut() { for v in row.iter_mut() { *v *= s; } } }
            for g in bias_gradients.iter_mut() { for v in g.iter_mut() { *v *= s; } }
        }
    }

    pub(crate) fn update_weights_with_mom(&mut self, gradients: &[Vec<Vec<f32>>], bias_gradients: &[Vec<f32>]) {
        for (li, layer) in self.layers.iter_mut().enumerate() {
            if li >= gradients.len() { continue; }
            for i in 0..layer.weights.len() {
                if i >= gradients[li].len() { continue; }
                for j in 0..layer.weights[i].len() {
                    if j >= gradients[li][i].len() { continue; }
                    layer.mom_weights[i][j] = self.mom * layer.mom_weights[i][j] + gradients[li][i][j];
                    layer.weights[i][j] -= self.lr * layer.mom_weights[i][j];
                }
            }
            for i in 0..layer.biases.len() {
                if i >= bias_gradients[li].len() { continue; }
                layer.mom_biases[i] = self.mom * layer.mom_biases[i] + bias_gradients[li][i];
                layer.biases[i] -= self.lr * layer.mom_biases[i];
            }
        }
    }

    pub(crate) fn train_with_ppo(&mut self, exps: &[crate::hand::Experience], epochs: usize, budget: std::time::Duration) {
        if exps.is_empty() || epochs == 0 { return; }
        const MINI_BS: usize = 32;
        const GRAD_CLIP: f32 = 0.5;

        let deadline = if budget.is_zero() { None } else { Some(std::time::Instant::now() + budget) };
        let expired = || deadline.map_or(false, |d| std::time::Instant::now() >= d);

        for layer in &mut self.layers {
            if layer.mom_weights.len() != layer.weights.len() {
                layer.mom_weights = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
            }
            if layer.mom_biases.len() != layer.biases.len() {
                layer.mom_biases = vec![0.0; layer.biases.len()];
            }
        }

        let out_dim = super::AI_IMAGE_W;
        let mut total_loss = 0.0f32;
        let mut total_samples = 0usize;

        'ep: for _ in 0..epochs {
            if expired() { break 'ep; }
            let mut idx: Vec<usize> = (0..exps.len()).collect();
            fastrand::shuffle(&mut idx);
            for chunk in idx.chunks(MINI_BS) {
                if expired() { break 'ep; }
                let mut tgrads: Vec<Vec<Vec<f32>>> = self.layers.iter().map(|l| {
                    vec![vec![0.0; l.weights[0].len()]; l.weights.len()]
                }).collect();
                let mut tbgrads: Vec<Vec<f32>> = self.layers.iter().map(|l| vec![0.0; l.biases.len()]).collect();
                let mut batch_vw_grad = vec![0.0f32; out_dim];
                let mut batch_vb_grad = 0.0f32;

                let mut batch_policy_loss = 0.0f32;
                let mut batch_value_loss = 0.0f32;
                let mut n_done = 0usize;

                for &i in chunk {
                    if expired() { break 'ep; }
                    let e = &exps[i];
                    let state = &e.state;
                    let old_out_raw = &e.old_out;
                    let advantage = e.advantage;
                    let return_ = e.return_;

                    let out = self.cpu_forward(state);

                    let mut old_probs = Vec::with_capacity(out_dim);
                    let mut new_probs = Vec::with_capacity(out_dim);
                    for j in 0..out.len().min(out_dim) {
                        let oj = if j < old_out_raw.len() { old_out_raw[j] } else { out[j] };
                        old_probs.push(Self::sigmoid(oj));
                        new_probs.push(Self::sigmoid(out[j]));
                    }

                    let mut action_bits = Vec::with_capacity(out_dim);
                    let mut a = e.action;
                    for _ in 0..out_dim {
                        action_bits.push(a & 1);
                        a >>= 1;
                    }

                    let mut log_ratio = 0.0f32;
                    for j in 0..out_dim.min(old_probs.len()) {
                        let np_f32: f32 = new_probs[j].clamp(1e-6, 1.0 - 1e-6);
                        let op_f32: f32 = old_probs[j].clamp(1e-6, 1.0 - 1e-6);
                        let act_f32: f32 = action_bits[j] as f32;
                        log_ratio += act_f32 * (np_f32 / op_f32).ln()
                            + (1.0 - act_f32) * ((1.0 - np_f32) / (1.0 - op_f32)).ln();
                    }
                    let ratio: f32 = log_ratio.exp().clamp(1.0 - super::ai::PPO_EPS, 1.0 + super::ai::PPO_EPS);
                    let surr1 = ratio * advantage;
                    let surr2 = ratio.clamp(1.0 - super::ai::PPO_EPS, 1.0 + super::ai::PPO_EPS) * advantage;
                    batch_policy_loss -= surr1.min(surr2);

                    let v_pred = if self.value_weights.len() == out.len() {
                        let mut v = self.value_bias;
                        for (w, o) in self.value_weights.iter().zip(out.iter()) { v += w * o; }
                        v
                    } else {
                        out.iter().sum::<f32>() / out.len().max(1) as f32
                    };
                    batch_value_loss += (v_pred - return_).powi(2);

                    let mut og = vec![0.0f32; out.len()];
                    for j in 0..out.len().min(out_dim) {
                        let p_f32: f32 = new_probs[j].clamp(1e-6, 1.0 - 1e-6);
                        let act_f32: f32 = action_bits[j] as f32;
                        let sign = if act_f32 > 0.5 { 1.0f32 } else { -1.0f32 };
                        og[j] = sign * advantage * p_f32 * (1.0 - p_f32);
                    }
                    self.backward(state, &og, &mut tgrads, &mut tbgrads);

                    if self.value_weights.len() == out.len() {
                        let vf_err: f32 = 2.0 * super::ai::PPO_VF_COEF * (v_pred - return_);
                        for j in 0..out.len().min(out_dim) {
                            batch_vw_grad[j] += vf_err * out[j];
                        }
                    }
                    n_done += 1;
                    total_samples += 1;
                }

                if n_done == 0 { break 'ep; }
                let bs = n_done as f32;
                let inv = 1.0 / bs;
                for g in tgrads.iter_mut() {
                    for row in g.iter_mut() {
                        for v in row.iter_mut() { *v *= inv; }
                    }
                }
                for g in tbgrads.iter_mut() {
                    for v in g.iter_mut() { *v *= inv; }
                }
                for v in batch_vw_grad.iter_mut() { *v *= inv; }
                batch_vb_grad *= inv;

                self.clip_gradients(&mut tgrads, &mut tbgrads, GRAD_CLIP);
                self.update_weights_with_mom(&tgrads, &tbgrads);

                if self.value_weights.len() == out_dim {
                    let vlr: f32 = self.lr * 0.5;
                    for j in 0..out_dim {
                        self.value_weights[j] -= vlr * batch_vw_grad[j];
                    }
                    self.value_bias -= vlr * batch_vb_grad;
                }

                total_loss += (batch_policy_loss + batch_value_loss) * inv;
            }
        }

        if total_samples > 0 {
            self.adapt_lr(total_loss / total_samples as f32);
            self.ep += 1;
        }
    }

    pub(crate) fn adapt_lr(&mut self, loss: f32) {
        let lcr = if self.last_loss.is_finite() && self.last_loss > 1e-8 {
            (loss / self.last_loss).clamp(0.1, 10.0)
        } else { 1.0 };

        let ef = if self.ep < 50 { 0.9 } else if self.ep < 200 { 1.0 } else { 0.7 };

        if lcr < 0.98 {
            self.lr = (self.lr * 1.015 * ef).min(0.02);
            self.bad_ep = 0;
        } else if lcr > 1.02 {
            self.lr = (self.lr * 0.90).max(0.0001);
            self.bad_ep += 1;
            if self.bad_ep >= 3 { self.lr = (self.lr * 0.7).max(0.00005); }
            if self.bad_ep >= 8 { self.bad_ep = 0; self.lr = 0.002; }
        } else {
            let cf = 0.5 * (1.0 + (std::f32::consts::PI * (self.ep % 200) as f32 / 200.0).cos());
            self.lr = 0.0008 * cf * ef;
            self.bad_ep = 0;
        }
        self.lr = self.lr.clamp(0.0001, 0.02);
        self.last_loss = if loss.is_finite() { loss } else { self.last_loss };
    }

    pub fn init_gpu_sync(&mut self) {
        if self.gpu_ok || (self.init_try && self.init_fail) { return; }
        if !self.init_try {
            self.init_try = true;
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(_) => { self.init_fail = true; return; }
            };
            let ok = rt.block_on(async { self.init_gpu().await });
            if ok && self.device.is_some() && self.queue.is_some() && self.gpu_exec.is_some() {
                self.gpu_ok = true;
                println!("[GPU] OK");
            } else {
                self.init_fail = true;
                self.device = None; self.queue = None; self.bs_buf = None; self.gpu_exec = None;
                for l in &mut self.layers { l.weights_buffer = None; l.biases_buffer = None; l.activations_buffer = None; }
                println!("[GPU] FAIL, using CPU");
            }
        }
    }

    pub async fn init_gpu(&mut self) -> bool {
        if self.gpu_ok { return true; }
        let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = match inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }).await { Ok(a) => a, Err(e) => { println!("[GPU] Adapter: {}", e); return false } };

        let info = adapter.get_info();
        let sw = {
            let n = info.name.to_lowercase();
            n.contains("llvmpipe") || n.contains("swiftshader") || n.contains("software") || n.contains("virtual")
                || info.vendor == 0x10005 || (info.vendor == 0x8086 && n.contains("haswell"))
        };
        if sw { return false; }

        let (dev, queue) = match adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("GPU"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }).await { Ok(dq) => dq, Err(e) => { println!("[GPU] Device: {}", e); return false } };

        let exec = Arc::new(gpu_utils::GpuNetworkExecutor::new(Arc::new(dev.clone()), Arc::new(queue.clone())));
        for (i, l) in self.layers.iter_mut().enumerate() {
            let wf: Vec<f32> = l.weights.iter().flatten().cloned().collect();
            l.weights_buffer = Some(dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("L{} w", i)), contents: bytemuck::cast_slice(&wf),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            l.biases_buffer = Some(dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("L{} b", i)), contents: bytemuck::cast_slice(&l.biases),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            l.activations_buffer = Some(dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("L{} a", i)), size: (l.activations.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
            }));
        }
        self.bs_buf = Some(dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bs"), contents: bytemuck::cast_slice(&[self.bs as u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));
        self.device = Some(dev); self.queue = Some(queue); self.gpu_exec = Some(exec); self.gpu_ok = true;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn exps(n: usize) -> Vec<crate::hand::Experience> {
        (0..n)
            .map(|i| crate::hand::Experience {
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
    fn ppo_epochs_zero_is_noop() {
        let mut net = DeepNeuralNetwork::new();
        let ep = net.ep;
        let lr = net.lr;
        net.train_with_ppo(&exps(64), 0, Duration::ZERO);
        assert_eq!(net.ep, ep, "epochs=0 must not count an update");
        assert_eq!(net.lr, lr, "epochs=0 must not touch the optimizer");
    }

    #[test]
    fn ppo_budget_stops_early() {
        let mut net = DeepNeuralNetwork::new();
        let start = Instant::now();
        net.train_with_ppo(&exps(64), 100_000, Duration::from_millis(50));
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "budget was ignored, ran for {elapsed:?}"
        );
        assert!(net.ep >= 1, "at least one partial update should land");
        assert!(net.validate());
    }

    #[test]
    fn ppo_updates_weights() {
        let mut net = DeepNeuralNetwork::new();
        let before = net.layers[0].weights[0][0];
        net.train_with_ppo(&exps(64), 1, Duration::ZERO);
        let after = net.layers[0].weights[0][0];
        assert_ne!(before, after, "a full epoch must change the parameters");
        assert!(net.validate());
    }
}

