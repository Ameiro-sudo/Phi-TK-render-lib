/*
 * vit.rs - Vision Transformer for note hand (L/R) assignment
 *
 * Patch-encodes the rendered frame (from MSRenderTarget::read_pixels_resized),
 * remembers respack note patterns as template tokens, scores each note via
 * cross-attention against the visual field, then applies chart-distribution
 * policy to emit a Left/Right decision.
 *
 * Naming: single-letter locals/fns where readable (kernel style).
 * Coupling: traits Emb/Att/Clf/Dst — ViT is the sole implementor.
 */
use crate::core::note::{Hand, Note, NoteKind};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/* ── geometry ─────────────────────────────────────────────────────────── */
pub const PS: usize = 5; // patch side (pixels)
pub const PG: usize = crate::hand::AI_IMAGE_W / PS; // 16
pub const PH: usize = crate::hand::AI_IMAGE_H / PS; // 9
pub const NP: usize = PG * PH; // 144 visual tokens
pub const PD: usize = PS * PS * crate::hand::AI_IMAGE_CHANNELS; // 75 raw patch dim
pub const D: usize = 64; // model width
pub const NH: usize = 4; // heads
pub const HD: usize = D / NH; // head dim
pub const NL: usize = 2; // transformer layers
pub const NT: usize = 4; // note-kind templates (click/hold/flick/drag)
pub const NF: usize = 16; // note feature dim before projection

/* ── embedded respack patterns (compile-time "memory") ───────────────── */
const RB: [&[u8]; NT] = [
    include_bytes!("../respack/click.png"),
    include_bytes!("../respack/hold.png"),
    include_bytes!("../respack/flick.png"),
    include_bytes!("../respack/drag.png"),
];

fn k2i(k: &NoteKind) -> usize {
    match k {
        NoteKind::Click => 0,
        NoteKind::Hold { .. } => 1,
        NoteKind::Flick => 2,
        NoteKind::Drag => 3,
    }
}

/* ── traits: embedding / attention / classifier / distribution ────────── */
pub trait Emb {
    /// frame -> [NP, D] patch tokens
    fn eb(&self, x: &[f32]) -> Vec<f32>;
    /// note + patch context -> [D] query token
    fn en(&self, n: &Note, pk: usize, m: &M) -> Vec<f32>;
}

pub trait Att {
    /// multi-head self-attn residual block over [n, D]
    fn fw(&self, x: &mut [f32], n: usize);
    /// cross-attn: query [1,D] over keys/values [NP,D] -> [D]
    fn xa(&self, q: &[f32], kv: &[f32]) -> Vec<f32>;
}

pub trait Clf {
    /// fused token -> logit ( >0 => Right )
    fn cg(&self, t: &[f32]) -> f32;
}

pub trait Dst {
    /// logits + note layout -> final hands
    fn ap(&self, ns: &[Note], lg: &[f32]) -> Vec<Hand>;
}

/* ── linear layer ────────────────────────────────────────────────────── */
#[derive(Clone, Serialize, Deserialize)]
pub struct L {
    pub w: Vec<f32>, // [o, i] row-major
    pub b: Vec<f32>,
    pub o: usize,
    pub i: usize,
}

impl L {
    pub fn n(i: usize, o: usize) -> Self {
        let s = (2.0 / i as f32).sqrt();
        Self {
            w: (0..o * i).map(|_| (fastrand::f32() * 2.0 - 1.0) * s).collect(),
            b: vec![0.0; o],
            o,
            i,
        }
    }
    #[inline]
    pub fn f(&self, x: &[f32], y: &mut [f32]) {
        // y must be len o; x len i
        y.par_iter_mut().enumerate().for_each(|(r, v)| {
            let mut s = self.b[r];
            let rw = &self.w[r * self.i..(r + 1) * self.i];
            for (w, xi) in rw.iter().zip(x.iter()) {
                s += w * xi;
            }
            *v = if s.is_finite() { s.clamp(-50., 50.) } else { 0. };
        });
    }
}

/* ── LayerNorm ───────────────────────────────────────────────────────── */
#[derive(Clone, Serialize, Deserialize)]
pub struct N {
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

impl N {
    pub fn n(d: usize) -> Self {
        Self {
            g: vec![1.0; d],
            b: vec![0.0; d],
        }
    }
    #[inline]
    pub fn f(&self, x: &mut [f32]) {
        let m = x.iter().sum::<f32>() / x.len() as f32;
        let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / x.len() as f32;
        let r = 1.0 / (v + 1e-5).sqrt();
        for (i, a) in x.iter_mut().enumerate() {
            *a = (*a - m) * r * self.g[i] + self.b[i];
        }
    }
}

/* ── one transformer encoder block ───────────────────────────────────── */
#[derive(Clone, Serialize, Deserialize)]
pub struct B {
    pub ln0: N,
    pub ln1: N,
    pub q: L,  // D -> D
    pub k: L,  // D -> D
    pub v: L,  // D -> D
    pub o: L,  // D -> D
    pub f0: L, // D -> 4D
    pub f1: L, // 4D -> D
}

impl B {
    pub fn n() -> Self {
        Self {
            ln0: N::n(D),
            ln1: N::n(D),
            q: L::n(D, D),
            k: L::n(D, D),
            v: L::n(D, D),
            o: L::n(D, D),
            f0: L::n(D, D * 4),
            f1: L::n(D * 4, D),
        }
    }
}

/* ── template memory: respack patterns the model "remembers" ─────────── */
#[derive(Clone, Serialize, Deserialize)]
pub struct M {
    pub t: Vec<f32>, // [NT, D] pooled template embeddings
    #[serde(skip)]
    pub ready: bool,
}

impl M {
    pub fn n() -> Self {
        let mut t = vec![0.0; NT * D];
        for (j, raw) in RB.iter().enumerate() {
            let e = t0(raw);
            t[j * D..(j + 1) * D].copy_from_slice(&e);
        }
        Self { t, ready: true }
    }
    pub fn wake(&mut self) {
        if self.ready {
            return;
        }
        *self = Self::n();
    }
    /// cosine-like similarity of feature f against kind template
    pub fn s(&self, k: usize, f: &[f32]) -> f32 {
        let t = &self.t[k * D..(k + 1) * D];
        let mut d = 0.0;
        let mut n0 = 0.0;
        let mut n1 = 0.0;
        let m = f.len().min(D);
        for i in 0..m {
            d += f[i] * t[i];
            n0 += f[i] * f[i];
            n1 += t[i] * t[i];
        }
        if n0 <= 1e-8 || n1 <= 1e-8 {
            return 0.0;
        }
        d / (n0.sqrt() * n1.sqrt())
    }
}

/// decode one respack png -> pooled [D] embedding (fixed pattern memory)
fn t0(raw: &[u8]) -> Vec<f32> {
    let mut e = vec![0.0f32; D];
    let Ok(im) = image::load_from_memory(raw) else {
        return e;
    };
    let im = im.thumbnail(32, 32).to_rgb8();
    let (w, h) = (im.width() as usize, im.height() as usize);
    if w == 0 || h == 0 {
        return e;
    }
    // coarse 4x4 spatial grid of mean RGB + global stats -> fill D
    let gx = 4.min(w);
    let gy = 4.min(h);
    let mut i = 0;
    for yy in 0..gy {
        for xx in 0..gx {
            let x0 = xx * w / gx;
            let x1 = ((xx + 1) * w / gx).max(x0 + 1);
            let y0 = yy * h / gy;
            let y1 = ((yy + 1) * h / gy).max(y0 + 1);
            let mut r = 0u32;
            let mut g = 0u32;
            let mut b = 0u32;
            let mut c = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = im.get_pixel(x as u32, y as u32);
                    r += p[0] as u32;
                    g += p[1] as u32;
                    b += p[2] as u32;
                    c += 1;
                }
            }
            let c = c.max(1) as f32;
            if i + 3 <= D {
                e[i] = r as f32 / c / 255.0;
                e[i + 1] = g as f32 / c / 255.0;
                e[i + 2] = b as f32 / c / 255.0;
                i += 3;
            }
        }
    }
    // aspect + edge energy fill remaining dims
    if i + 2 <= D {
        e[i] = w as f32 / h.max(1) as f32;
        e[i + 1] = 1.0; // present flag
    }
    e
}

/* ── ViT core ────────────────────────────────────────────────────────── */
/// positional encoding (sin/cos, cached)
fn pe0(i: usize, d: usize) -> f32 {
    let p = i as f32;
    let w = 1.0 / 10000f32.powf(2.0 * (d / 2) as f32 / D as f32);
    if d % 2 == 0 {
        (p * w).sin()
    } else {
        (p * w).cos()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct V {
    pub pe: L,           // PD -> D patch embed
    pub ne: L,           // NF -> D note query embed
    pub bl: Vec<B>,      // encoder blocks
    pub fc: L,           // 2D -> 1 classifier (query ⊕ attended)
    pub m: M,            // respack template memory
    pub pos: Vec<f32>,   // [NP, D] positional table
    pub npos: Vec<f32>,  // [NF] note-pos bias table (x,y slots reuse)
    pub eps: f32,        // explore rate
    pub tot: u64,        // decision count
    pub ok: u64,         // spatial-agreement count
    /// GPU compute handle (vit.wgsl); None => CPU rayon path
    #[serde(skip)]
    pub g: Option<crate::gpu_vit::VitGpu>,
}

impl std::fmt::Debug for V {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V")
            .field("bl", &self.bl.len())
            .field("eps", &self.eps)
            .field("tot", &self.tot)
            .field("ok", &self.ok)
            .finish()
    }
}

impl V {
    pub fn n() -> Self {
        Self {
            pe: L::n(PD, D),
            ne: L::n(NF, D),
            bl: (0..NL).map(|_| B::n()).collect(),
            fc: L::n(D * 2, 1),
            m: M::n(),
            pos: (0..NP * D).map(|i| pe0(i / D, i % D)).collect(),
            npos: vec![0.0; NF],
            eps: 0.15,
            tot: 0,
            ok: 0,
            g: None,
        }
    }

    pub fn wake(&mut self) {
        self.m.wake();
        if self.pos.len() != NP * D {
            self.pos = (0..NP * D).map(|i| pe0(i / D, i % D)).collect();
        }
        if self.eps <= 0.0 || !self.eps.is_finite() {
            self.eps = 0.15;
        }
        if self.g.is_none() {
            self.g = crate::gpu_vit::VitGpu::try_new(self);
        }
    }

    /// features + patch indices for GPU qry kernel
    pub(crate) fn qpack(&self, ns: &[Note]) -> (Vec<f32>, Vec<u32>) {
        let mut f = vec![0.0f32; ns.len() * NF];
        let mut p = vec![0u32; ns.len()];
        for (i, n) in ns.iter().enumerate() {
            let u = Self::ux(n.object.translation.0.now());
            let v = ((1.0 - n.object.translation.1.now()) * 0.5).clamp(0.0, 1.0);
            let pk = Self::pk(u, v);
            f[i * NF..(i + 1) * NF].copy_from_slice(&Self::nf(n, u, v, &self.m));
            p[i] = pk as u32;
        }
        (f, p)
    }

    pub fn val(&self) -> bool {
        self.bl.iter().all(|b| {
            b.q.w.iter().all(|w| w.is_finite()) && b.ln0.g.iter().all(|g| g.is_finite())
        }) && self.eps.is_finite()
    }

    /// pixel(x,y) -> patch index
    #[inline]
    pub fn pk(x: f32, y: f32) -> usize {
        // x,y in image-normalized [0,1]; AI image is top-left origin (row-major)
        let cx = ((x * crate::hand::AI_IMAGE_W as f32) as isize).clamp(0, crate::hand::AI_IMAGE_W as isize - 1) as usize;
        let cy = ((y * crate::hand::AI_IMAGE_H as f32) as isize).clamp(0, crate::hand::AI_IMAGE_H as isize - 1) as usize;
        let px = (cx / PS).min(PG - 1);
        let py = (cy / PS).min(PH - 1);
        py * PG + px
    }

    /// note world x (approx -1..1) -> image u
    #[inline]
    pub fn ux(x: f32) -> f32 {
        ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    }

    /// build [NF] raw features for a note + its patch slot
    pub(crate) fn nf(n: &Note, u: f32, v: f32, m: &M) -> Vec<f32> {
        let k = k2i(&n.kind);
        // local patch stats would be filled by caller via m similarity
        let mut f = [0.0f32; NF];
        f[0] = u;
        f[1] = v;
        f[2] = n.object.translation.0.now();
        f[3] = n.object.translation.1.now();
        f[4] = n.time;
        f[5] = n.height;
        f[6] = n.speed;
        f[7] = n.above as u8 as f32;
        // one-hot kind
        f[8 + k] = 1.0;
        // template similarities (all 4 kinds — AI "remembers" patterns)
        for j in 0..NT {
            // use patch-agnostic priors + kind boost
            f[12] += if j == k { 1.0 } else { 0.0 };
        }
        f[13] = m.s(k, &m.t[k * D..(k + 1) * D]); // self-sim sanity ~1
        f[14] = u * 2.0 - 1.0; // centered x
        f[15] = (n.time * 7.0).sin(); // temporal phase
        f.to_vec()
    }

    /// full forward: frame + notes -> per-note right-logits
    pub fn fw(&self, img: &[f32], ns: &[Note]) -> Vec<f32> {
        // GPU: vit.wgsl embed -> enc (NL) -> qry; fall back to CPU on any miss
        if let Some(g) = self.g.as_ref() {
            if img.len() >= crate::hand::AI_IMAGE_SIZE && !ns.is_empty() {
                let (qf, qp) = self.qpack(ns);
                if let Some(lg) = g.forward(img, &qf, &qp) {
                    if lg.len() == ns.len() && lg.iter().all(|x| x.is_finite()) {
                        return lg;
                    }
                }
            }
        }

        let mut tk = self.eb(img); // [NP, D]
        // self-attention over visual tokens
        for b in &self.bl {
            b.fw(&mut tk, NP);
        }

        // per-note query: cross-attend then classify
        ns.par_iter()
            .map(|n| {
                let u = Self::ux(n.object.translation.0.now());
                // v: map world-y into image rows; render uses y-up, image y-down
                let vy = n.object.translation.1.now();
                let v = ((1.0 - vy) * 0.5).clamp(0.0, 1.0);
                let pk = Self::pk(u, v);
                let raw = Self::nf(n, u, v, &self.m);
                let mut qt = vec![0.0; D];
                self.ne.f(&raw, &mut qt);
                // add positional
                for d in 0..D {
                    qt[d] += self.pos[pk * D + d] * 0.15;
                }
                // cross-attend to visual field
                let mut ctx = Vec::with_capacity(D);
                for b in &self.bl {
                    ctx = b.xa(&qt, &tk);
                    for d in 0..D {
                        qt[d] = qt[d] * 0.5 + ctx[d] * 0.5;
                    }
                }
                // fuse [query, context] -> logit
                let mut fus = Vec::with_capacity(D * 2);
                fus.extend_from_slice(&qt);
                fus.extend_from_slice(&ctx);
                let mut o = [0.0f32; 1];
                self.fc.f(&fus, &mut o);
                o[0]
            })
            .collect()
    }
}

/* ── Emb impl ────────────────────────────────────────────────────────── */
impl Emb for V {
    fn eb(&self, x: &[f32]) -> Vec<f32> {
        let w = crate::hand::AI_IMAGE_W;
        let c = crate::hand::AI_IMAGE_CHANNELS;
        let mut out = vec![0.0f32; NP * D];
        // parallel over patches
        out.par_chunks_mut(D).enumerate().for_each(|(p, tok)| {
            let py = p / PG;
            let px = p % PG;
            let mut raw = vec![0.0f32; PD];
            let mut i = 0;
            for dy in 0..PS {
                for dx in 0..PS {
                    let sx = px * PS + dx;
                    let sy = py * PS + dy;
                    let si = (sy * w + sx) * c;
                    for ch in 0..c {
                        if si + ch < x.len() {
                            raw[i] = x[si + ch];
                        }
                        i += 1;
                    }
                }
            }
            self.pe.f(&raw, tok);
            for d in 0..D {
                tok[d] += self.pos[p * D + d];
                if !tok[d].is_finite() {
                    tok[d] = 0.0;
                }
            }
        });
        out
    }

    fn en(&self, n: &Note, pk: usize, m: &M) -> Vec<f32> {
        let u = Self::ux(n.object.translation.0.now());
        let v = ((1.0 - n.object.translation.1.now()) * 0.5).clamp(0.0, 1.0);
        let raw = Self::nf(n, u, v, m);
        let mut t = vec![0.0; D];
        self.ne.f(&raw, &mut t);
        for d in 0..D {
            t[d] += self.pos[pk * D + d] * 0.15;
        }
        t
    }
}

/* ── Att impl ────────────────────────────────────────────────────────── */
impl Att for B {
    fn fw(&self, x: &mut [f32], n: usize) {
        // pre-LN self-attention residual over n tokens of width D
        let mut a = x.to_vec();
        for t in a.chunks_mut(D) {
            self.ln0.f(t);
        }
        // Q,K,V projections for all tokens
        let mut q = vec![0.0; n * D];
        let mut k = vec![0.0; n * D];
        let mut v = vec![0.0; n * D];
        // sequential matmul (n small ~144, D=64)
        for t in 0..n {
            self.q.f(&a[t * D..(t + 1) * D], &mut q[t * D..(t + 1) * D]);
            self.k.f(&a[t * D..(t + 1) * D], &mut k[t * D..(t + 1) * D]);
            self.v.f(&a[t * D..(t + 1) * D], &mut v[t * D..(t + 1) * D]);
        }
        // multi-head attention
        let mut out = vec![0.0; n * D];
        let scale = 1.0 / (HD as f32).sqrt();
        out.par_chunks_mut(D).enumerate().for_each(|(ti, o)| {
            for h in 0..NH {
                let base = h * HD;
                // scores vs all keys
                let mut s = vec![0.0f32; n];
                let qi = &q[ti * D + base..ti * D + base + HD];
                for (j, sj) in s.iter_mut().enumerate() {
                    let kj = &k[j * D + base..j * D + base + HD];
                    let mut d = 0.0;
                    for t in 0..HD {
                        d += qi[t] * kj[t];
                    }
                    *sj = d * scale;
                }
                // softmax
                let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut z = 0.0;
                for sj in s.iter_mut() {
                    *sj = (*sj - mx).exp();
                    z += *sj;
                }
                let inv = 1.0 / z.max(1e-8);
                for sj in s.iter_mut() {
                    *sj *= inv;
                }
                // weighted V
                for t in 0..HD {
                    let mut acc = 0.0;
                    for (j, sj) in s.iter().enumerate() {
                        acc += *sj * v[j * D + base + t];
                    }
                    o[base + t] = acc;
                }
            }
        });
        // output proj + residual
        let mut p = vec![0.0; n * D];
        for t in 0..n {
            self.o.f(&out[t * D..(t + 1) * D], &mut p[t * D..(t + 1) * D]);
        }
        for i in 0..x.len() {
            x[i] += p[i];
        }
        // FFN
        let mut h = x.to_vec();
        for t in h.chunks_mut(D) {
            self.ln1.f(t);
        }
        let mut f = vec![0.0; n * D * 4];
        let mut g = vec![0.0; n * D];
        for t in 0..n {
            self.f0.f(&h[t * D..(t + 1) * D], &mut f[t * D * 4..(t + 1) * D * 4]);
            for a in f[t * D * 4..(t + 1) * D * 4].iter_mut() {
                // gelu
                let xv = *a;
                *a = 0.5 * xv * (1.0 + (0.7978845608_f32 * (xv + 0.044715 * xv * xv * xv)).tanh());
            }
            self.f1.f(&f[t * D * 4..(t + 1) * D * 4], &mut g[t * D..(t + 1) * D]);
        }
        for i in 0..x.len() {
            x[i] += g[i];
        }
    }

    fn xa(&self, q: &[f32], kv: &[f32]) -> Vec<f32> {
        // q: [D], kv: [NP*D]
        let n = kv.len() / D;
        let mut ql = vec![0.0; D];
        let qn = {
            let mut t = q.to_vec();
            self.ln0.f(&mut t);
            t
        };
        self.q.f(&qn, &mut ql);
        // K,V over all patches
        let mut ks = vec![0.0; n * D];
        let mut vs = vec![0.0; n * D];
        for j in 0..n {
            self.k.f(&kv[j * D..(j + 1) * D], &mut ks[j * D..(j + 1) * D]);
            self.v.f(&kv[j * D..(j + 1) * D], &mut vs[j * D..(j + 1) * D]);
        }
        let mut o = vec![0.0; D];
        let scale = 1.0 / (HD as f32).sqrt();
        for h in 0..NH {
            let base = h * HD;
            let mut s = vec![0.0f32; n];
            for (j, sj) in s.iter_mut().enumerate() {
                let mut d = 0.0;
                for t in 0..HD {
                    d += ql[base + t] * ks[j * D + base + t];
                }
                *sj = d * scale;
            }
            let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0.0;
            for sj in s.iter_mut() {
                *sj = (*sj - mx).exp();
                z += *sj;
            }
            let inv = 1.0 / z.max(1e-8);
            for t in 0..HD {
                let mut acc = 0.0;
                for (j, sj) in s.iter().enumerate() {
                    acc += *sj * vs[j * D + base + t];
                }
                o[base + t] = acc * inv;
            }
        }
        let mut p = vec![0.0; D];
        self.o.f(&o, &mut p);
        // residual-ish
        let mut r = q.to_vec();
        for i in 0..D {
            r[i] += p[i];
        }
        r
    }
}

/* ── Clf impl ────────────────────────────────────────────────────────── */
impl Clf for V {
    fn cg(&self, t: &[f32]) -> f32 {
        let mut o = [0.0f32; 1];
        self.fc.f(t, &mut o);
        o[0]
    }
}

/* ── Dst: chart-distribution policy ──────────────────────────────────── */
impl Dst for V {
    fn ap(&self, ns: &[Note], lg: &[f32]) -> Vec<Hand> {
        let n = ns.len();
        if n == 0 {
            return vec![];
        }
        // 1. prior from spatial position (left half -> Left)
        // 2. ViT logit
        // 3. distribution balance: running L/R ratio
        // 4. consecutive-run clamp
        // 5. simultaneous (same time) must not overload one hand when split possible
        let mut h = vec![Hand::Left; n];
        let mut cl = 0usize; // consecutive-left run
        let mut cr = 0usize; // consecutive-right run
        let mut nl = 0usize; // total left
        let mut nr = 0usize; // total right
        let mut prev_t = f32::NEG_INFINITY;
        let mut prev_h: Option<Hand> = None;

        // sort indices by time for sequential policy
        let mut ord: Vec<usize> = (0..n).collect();
        ord.sort_by(|&a, &b| {
            ns[a]
                .time
                .partial_cmp(&ns[b].time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &i in &ord {
            let x = ns[i].object.translation.0.now();
            let l = lg.get(i).copied().unwrap_or(0.0);
            // spatial prior: note x<0 => left
            let sp = if x < -0.05 {
                -1.0
            } else if x > 0.05 {
                1.0
            } else {
                0.0
            };
            // balance pull: if one hand overloaded, nudge
            let tot = (nl + nr).max(1) as f32;
            let bal = (nr as f32 - nl as f32) / tot; // >0 means right heavy -> push left
            // score: vision + spatial + balance
            let sc = 0.45 * l + 0.40 * sp + 0.15 * bal * 2.0;

            // same-time chord: force diversity if both near center
            let sim = (ns[i].time - prev_t).abs() < 1e-3;

            let mut cand = if sc < 0.0 { Hand::Left } else { Hand::Right };

            // consecutive limit (kernel: CONSECUTIVE_LIMIT from parent)
            const CL: usize = crate::hand::ai::CONSECUTIVE_LIMIT;
            match cand {
                Hand::Left => {
                    if cl >= CL {
                        cand = Hand::Right;
                    }
                }
                Hand::Right => {
                    if cr >= CL {
                        cand = Hand::Left;
                    }
                }
            }
            // if still tied on a simultaneous pair, alternate
            if sim && prev_h == Some(cand) {
                cand = match cand {
                    Hand::Left => Hand::Right,
                    Hand::Right => Hand::Left,
                };
            }

            h[i] = cand;
            prev_h = Some(cand);
            match cand {
                Hand::Left => {
                    nl += 1;
                    cl += 1;
                    cr = 0;
                }
                Hand::Right => {
                    nr += 1;
                    cr += 1;
                    cl = 0;
                }
            }
            prev_t = ns[i].time;
        }
        h
    }
}

/* ── public entry: score + decide ────────────────────────────────────── */
/// Run ViT on (frame, notes) and return Left/Right per note.
/// `t` is wall/game time used for epsilon-decay bookkeeping (in-place via interior).
pub fn go(v: &V, img: &[f32], ns: &[Note]) -> Vec<Hand> {
    if ns.is_empty() {
        return vec![];
    }
    let mut lg = v.fw(img, ns);
    // epsilon explore (only disturbs logit sign; policy re-applies distribution)
    if v.eps > 0.0 {
        for l in lg.iter_mut() {
            if fastrand::f32() < v.eps {
                *l = -*l;
            }
        }
    }
    v.ap(ns, &lg)
}
