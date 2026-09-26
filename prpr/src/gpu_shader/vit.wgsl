// ViT forward: patch embed -> encoder blocks (self-attn + FFN) -> query cross-attn + classify
// Geometry must match hand/vit.rs (PS/PG/PH/NP/PD/D/NH/HD/NL/NF).

const PS: u32 = 5u;
const PG: u32 = 16u;
const PH: u32 = 9u;
const NP: u32 = 144u;
const PD: u32 = 75u;
const D: u32 = 64u;
const NH: u32 = 4u;
const HD: u32 = 16u;
const NL: u32 = 2u;
const NF: u32 = 16u;
const IW: u32 = 80u;
const IH: u32 = 45u;
const IC: u32 = 3u;
const FF: u32 = 256u; // 4 * D
const QD: u32 = 128u; // 2 * D fused

// packed weight offsets (f32 indices) — keep in sync with gpu_vit.rs::offs
const OFF_PE: u32 = 0u;
const OFF_PEB: u32 = OFF_PE + (D * PD);
const OFF_POS: u32 = OFF_PEB + D;
const OFF_NE: u32 = OFF_POS + (NP * D);
const OFF_NEB: u32 = OFF_NE + (D * NF);
const LAY0: u32 = OFF_NEB + D;
// per layer: ln0g ln0b | q q | k k | v v | o o | ln1g ln1b | f0 f0 | f1 f1
const L0_LN0G: u32 = 0u;
const L0_LN0B: u32 = L0_LN0G + D;
const L0_QW: u32 = L0_LN0B + D;
const L0_QB: u32 = L0_QW + (D * D);
const L0_KW: u32 = L0_QB + D;
const L0_KB: u32 = L0_KW + (D * D);
const L0_VW: u32 = L0_KB + D;
const L0_VB: u32 = L0_VW + (D * D);
const L0_OW: u32 = L0_VB + D;
const L0_OB: u32 = L0_OW + (D * D);
const L0_LN1G: u32 = L0_OB + D;
const L0_LN1B: u32 = L0_LN1G + D;
const L0_F0W: u32 = L0_LN1B + D;
const L0_F0B: u32 = L0_F0W + (FF * D);
const L0_F1W: u32 = L0_F0B + FF;
const L0_F1B: u32 = L0_F1W + (D * FF);
const LAY_SZ: u32 = L0_F1B + D;
const OFF_FC: u32 = LAY0 + (NL * LAY_SZ);
const W_LEN: u32 = OFF_FC + QD + 1u;

struct VitP {
    nq: u32,
    layer: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> p: VitP;
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read_write> tk: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> qin: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read> qp: array<u32>;
@group(0) @binding(7) var<storage, read_write> tkb: array<f32>;

fn ok_f(s: f32) -> f32 {
    if (s != s) {return 0.0;}
    return clamp(s, -50.0, 50.0);
}

fn layernorm(x: ptr<function, array<f32, 64>>, goff: u32, boff: u32) {
    var m = 0.0;
    for (var i = 0u; i < D; i++) {
        m += (*x)[i];
    }
    m /= f32(D);
    var v = 0.0;
    for (var i = 0u; i < D; i++) {
        let d = (*x)[i] - m;
        v += d * d;
    }
    v /= f32(D);
    let r = inverseSqrt(v + 1e-5);
    for (var i = 0u; i < D; i++) {
        (*x)[i] = ((*x)[i] - m) * r * w[goff + i] + w[boff + i];
    }
}

fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

fn load_tok(j: u32, src_to_b: bool, dst: ptr<function, array<f32, D>>) {
    for (var i = 0u; i < D; i++) {
        (*dst)[i] = select(tkb[j * D + i], tk[j * D + i], src_to_b);
    }
}

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
    let pe_i = id.x;
    if (pe_i >= NP) {
        return;
    }
    let px = pe_i % PG;
    let py = pe_i / PG;
    var raw: array<f32, PD>;
    var i = 0u;
    for (var dy = 0u; dy < PS; dy++) {
        for (var dx = 0u; dx < PS; dx++) {
            let sx = px * PS + dx;
            let sy = py * PS + dy;
            let si = (sy * IW + sx) * IC;
            for (var ch = 0u; ch < IC; ch++) {
                let idx = si + ch;
                if (idx < arrayLength(&img)) {
                    raw[i] = img[idx];
                }
                i++;
            }
        }
    }
    for (var r = 0u; r < D; r++) {
        var s = w[OFF_PEB + r];
        for (var c = 0u; c < PD; c++) {
            s += w[OFF_PE + r * PD + c] * raw[c];
        }
        tk[pe_i * D + r] = ok_f(s) + w[OFF_POS + pe_i * D + r];
    }
}

// p.layer even: tk -> tkb; odd: tkb -> tk (ping-pong avoids cross-thread races)
@compute @workgroup_size(64)
fn enc(@builtin(global_invocation_id) id: vec3<u32>) {
    let ti = id.x;
    if (ti >= NP) {return;}
    let lo = LAY0 + p.layer * LAY_SZ;
    let src_to_b = (p.layer & 1u) == 0u;
    var x: array<f32, D>;
    for (var i = 0u; i < D; i++) {
        x[i] = select(tkb[ti * D + i], tk[ti * D + i], src_to_b);
    }
    // pre-LN + Q for self
    var a: array<f32, D>;
    for (var i = 0u; i < D; i++) {a[i] = x[i];}
    layernorm(&a, lo + L0_LN0G, lo + L0_LN0B);
    var q: array<f32, D>;
    for (var r = 0u; r < D; r++) {
        var s = w[lo + L0_QB + r];

        for (var c = 0u; c < D; c++) {s += w[lo + L0_QW + r * D + c] * a[c];}
        q[r] = ok_f(s);
    }

    let scale = inverseSqrt(f32(HD));
    var att: array<f32, D>;
    for (var h = 0u; h < NH; h++) {
        let base = h * HD;
        var s: array<f32, NP>;
        var mx = -3.4028235e38;
        for (var j = 0u; j < NP; j++) {
            var aj: array<f32, D>;
            load_tok(j, src_to_b, &aj);
            layernorm(&aj, lo + L0_LN0G, lo + L0_LN0B);
            var kj: array<f32, D>;
            for (var r = 0u; r < D; r++) {
                var sum = w[lo + L0_KB + r];
                for (var c = 0u; c < D; c++) {
                    sum += w[lo + L0_KW + r * D + c] * aj[c];
                }
                kj[r] = ok_f(sum);
            }
            var d = 0.0;
            for (var t = 0u; t < HD; t++) {d += q[base + t] * kj[base + t];}
            s[j] = d * scale;
            if (s[j] > mx) {mx = s[j];}
        }
        var z = 0.0;
        for (var j = 0u; j < NP; j++) {
            s[j] = exp(s[j] - mx);
            z += s[j];
        }
        let inv = 1.0 / max(z, 1e-8);
        var acc: array<f32, HD>;
        for (var t = 0u; t < HD; t++) {
            acc[t] = 0.0;
        }
        for (var j = 0u; j < NP; j++) {
            var aj: array<f32, D>;
            load_tok(j, src_to_b, &aj);
            layernorm(&aj, lo + L0_LN0G, lo + L0_LN0B);
            var vj: array<f32, D>;
            for (var r = 0u; r < D; r++) {
                var sum = w[lo + L0_VB + r];
                for (var c = 0u; c < D; c++) {
                    sum += w[lo + L0_VW + r * D + c] * aj[c];
                }
                vj[r] = ok_f(sum);
            }
            let wj = s[j] * inv;
            for (var t = 0u; t < HD; t++) {
                acc[t] += wj * vj[base + t];
            }
        }
        for (var t = 0u; t < HD; t++) {
            att[base + t] = acc[t];
        }
    }

    for (var r = 0u; r < D; r++) {
        var s = w[lo + L0_OB + r];
        for (var c = 0u; c < D; c++) {
            s += w[lo + L0_OW + r * D + c] * att[c];
        }
        x[r] += ok_f(s);
    }

    var h: array<f32, D>;
    for (var i = 0u; i < D; i++) {
        h[i] = x[i];
    }
    layernorm(&h, lo + L0_LN1G, lo + L0_LN1B);

    var hid: array<f32, FF>;
    for (var r = 0u; r < FF; r++) {
        var s = w[lo + L0_F0B + r];
        for (var c = 0u; c < D; c++) {
            s += w[lo + L0_F0W + r * D + c] * h[c];
        }
        hid[r] = gelu(ok_f(s));
    }
    for (var r = 0u; r < D; r++) {
        var s = w[lo + L0_F1B + r];
        for (var c = 0u; c < FF; c++) {
            s += w[lo + L0_F1W + r * FF + c] * hid[c];
        }
        x[r] += ok_f(s);
    }

    for (var i = 0u; i < D; i++) {
        if (x[i] != x[i]) {
            x[i] = 0.0;
        }
        if (src_to_b) {
            tkb[ti * D + i] = x[i];
        } else {
            tk[ti * D + i] = x[i];
        }
    }
}

// p._p0 == 1 => final tokens live in tkb (NL odd); else tk (NL even)
@compute @workgroup_size(64)
fn qry(@builtin(global_invocation_id) id: vec3<u32>) {
    let qi = id.x;
    if (qi >= p.nq || qi * NF >= arrayLength(&qin)) {return;}
    let pk = qp[qi];
    let fbase = qi * NF;
    let src_b = p._p0 == 1u;

    var raw: array<f32, NF>;
    for (var i = 0u; i < NF; i++) {raw[i] = qin[fbase + i];}

    var qt: array<f32, D>;
    for (var r = 0u; r < D; r++) {
        var s = w[OFF_NEB + r];
        for (var c = 0u; c < NF; c++) {
            s += w[OFF_NE + r * NF + c] * raw[c];
        }
        let pos = w[OFF_POS + pk * D + r] * 0.15;
        qt[r] = ok_f(s) + pos;
    }

    var ctx: array<f32, D>;
    for (var i = 0u; i < D; i++) {ctx[i] = qt[i];}

    for (var li = 0u; li < NL; li++) {
        let lo = LAY0 + li * LAY_SZ;
        var qn: array<f32, D>;
        for (var i = 0u; i < D; i++) {qn[i] = qt[i];}
        layernorm(&qn, lo + L0_LN0G, lo + L0_LN0B);

        var ql: array<f32, D>;
        for (var r = 0u; r < D; r++) {
            var s = w[lo + L0_QB + r];
            for (var c = 0u; c < D; c++) {
                s += w[lo + L0_QW + r * D + c] * qn[c];
            }
            ql[r] = ok_f(s);
        }

        let scale = inverseSqrt(f32(HD));
        var o: array<f32, D>;
        for (var h = 0u; h < NH; h++) {
            let base = h * HD;
            var sh: array<f32, NP>;
            var mxh = -3.4028235e38;
            for (var j = 0u; j < NP; j++) {
                var kj: array<f32, D>;
                for (var r = 0u; r < D; r++) {
                    var sk = w[lo + L0_KB + r];
                    for (var c = 0u; c < D; c++) {
                        let tv = select(tk[j * D + c], tkb[j * D + c], src_b);
                        sk += w[lo + L0_KW + r * D + c] * tv;
                    }
                    kj[r] = ok_f(sk);
                }
                var d = 0.0;
                for (var t = 0u; t < HD; t++) {
                    d += ql[base + t] * kj[base + t];
                }
                sh[j] = d * scale;
                if (sh[j] > mxh) {
                    mxh = sh[j];
                }
            }
            var zh = 0.0;
            for (var j = 0u; j < NP; j++) {
                sh[j] = exp(sh[j] - mxh);
                zh += sh[j];
            }
            let invh = 1.0 / max(zh, 1e-8);
            var acch: array<f32, HD>;
            for (var t = 0u; t < HD; t++) {
                acch[t] = 0.0;
            }
            for (var j = 0u; j < NP; j++) {
                var vj: array<f32, D>;
                for (var r = 0u; r < D; r++) {
                    var sv = w[lo + L0_VB + r];
                    for (var c = 0u; c < D; c++) {
                        let tv = select(tk[j * D + c], tkb[j * D + c], src_b);
                        sv += w[lo + L0_VW + r * D + c] * tv;
                    }
                    vj[r] = ok_f(sv);
                }
                let wj = sh[j] * invh;
                for (var t = 0u; t < HD; t++) {
                    acch[t] += wj * vj[base + t];
                }
            }
            for (var t = 0u; t < HD; t++) {
                o[base + t] = acch[t];
            }
        }

        var pr: array<f32, D>;
        for (var r = 0u; r < D; r++) {
            var s = w[lo + L0_OB + r];
            for (var c = 0u; c < D; c++) {
                s += w[lo + L0_OW + r * D + c] * o[c];
            }
            pr[r] = ok_f(s);
        }

        // CPU: ctx = xa(qt) = qt + proj; qt = 0.5*qt + 0.5*ctx
        var nxt: array<f32, D>;
        for (var i = 0u; i < D; i++) {
            nxt[i] = qt[i] * 0.5 + (qt[i] + pr[i]) * 0.5;
        }
        for (var i = 0u; i < D; i++) {
            qt[i] = nxt[i];
            ctx[i] = qt[i];
        }
    }

    var fus: array<f32, QD>;
    for (var i = 0u; i < D; i++) {
        fus[i] = qt[i];
        fus[D + i] = ctx[i];
    }
    var lg = w[OFF_FC + QD];
    for (var c = 0u; c < QD; c++) {
        lg += w[OFF_FC + c] * fus[c];
    }
    if (lg != lg) {
        lg = 0.0;
    }
    out[qi] = lg;
}
