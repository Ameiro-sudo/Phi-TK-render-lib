/*
 * gpu_vit.rs — wgpu compute executor for hand/vit.rs
 *
 * Uploads packed V weights once, runs embed / enc / qry kernels from
 * gpu_shader/vit.wgsl. CPU path in vit.rs remains the fallback.
 */
use crate::hand::vit::{V, D, NF, NL, NP, PD};
use wgpu::util::DeviceExt;

const FF: usize = 4 * D; // 256
const QD: usize = 2 * D; // 128

/// Packed weight layout offsets — mirrors gpu_shader/vit.wgsl consts.
pub mod offs {
    use super::*;

    pub const OFF_PE: usize = 0;
    pub const OFF_PEB: usize = OFF_PE + D * PD;
    pub const OFF_POS: usize = OFF_PEB + D;
    pub const OFF_NE: usize = OFF_POS + NP * D;
    pub const OFF_NEB: usize = OFF_NE + D * NF;
    pub const LAY0: usize = OFF_NEB + D;

    pub const L0_LN0G: usize = 0;
    pub const L0_LN0B: usize = L0_LN0G + D;
    pub const L0_QW: usize = L0_LN0B + D;
    pub const L0_QB: usize = L0_QW + D * D;
    pub const L0_KW: usize = L0_QB + D;
    pub const L0_KB: usize = L0_KW + D * D;
    pub const L0_VW: usize = L0_KB + D;
    pub const L0_VB: usize = L0_VW + D * D;
    pub const L0_OW: usize = L0_VB + D;
    pub const L0_OB: usize = L0_OW + D * D;
    pub const L0_LN1G: usize = L0_OB + D;
    pub const L0_LN1B: usize = L0_LN1G + D;
    pub const L0_F0W: usize = L0_LN1B + D;
    pub const L0_F0B: usize = L0_F0W + FF * D;
    pub const L0_F1W: usize = L0_F0B + FF;
    pub const L0_F1B: usize = L0_F1W + D * FF;
    pub const LAY_SZ: usize = L0_F1B + D;

    pub const OFF_FC: usize = LAY0 + NL * LAY_SZ;
    pub const W_LEN: usize = OFF_FC + QD + 1;
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VitP {
    nq: u32,
    layer: u32,
    _p0: u32,
    _p1: u32,
}

/// Encode V linear/LN weights into one flat buffer (layout == vit.wgsl).
pub fn pack(v: &V) -> Vec<f32> {
    use offs::*;
    let mut b = vec![0.0f32; W_LEN];
    b[OFF_PE..OFF_PE + D * PD].copy_from_slice(&v.pe.w);
    b[OFF_PEB..OFF_PEB + D].copy_from_slice(&v.pe.b);
    b[OFF_POS..OFF_POS + NP * D].copy_from_slice(&v.pos);
    b[OFF_NE..OFF_NE + D * NF].copy_from_slice(&v.ne.w);
    b[OFF_NEB..OFF_NEB + D].copy_from_slice(&v.ne.b);
    for (li, blk) in v.bl.iter().enumerate() {
        let lo = LAY0 + li * LAY_SZ;
        b[lo + L0_LN0G..lo + L0_LN0G + D].copy_from_slice(&blk.ln0.g);
        b[lo + L0_LN0B..lo + L0_LN0B + D].copy_from_slice(&blk.ln0.b);
        b[lo + L0_QW..lo + L0_QW + D * D].copy_from_slice(&blk.q.w);
        b[lo + L0_QB..lo + L0_QB + D].copy_from_slice(&blk.q.b);
        b[lo + L0_KW..lo + L0_KW + D * D].copy_from_slice(&blk.k.w);
        b[lo + L0_KB..lo + L0_KB + D].copy_from_slice(&blk.k.b);
        b[lo + L0_VW..lo + L0_VW + D * D].copy_from_slice(&blk.v.w);
        b[lo + L0_VB..lo + L0_VB + D].copy_from_slice(&blk.v.b);
        b[lo + L0_OW..lo + L0_OW + D * D].copy_from_slice(&blk.o.w);
        b[lo + L0_OB..lo + L0_OB + D].copy_from_slice(&blk.o.b);
        b[lo + L0_LN1G..lo + L0_LN1G + D].copy_from_slice(&blk.ln1.g);
        b[lo + L0_LN1B..lo + L0_LN1B + D].copy_from_slice(&blk.ln1.b);
        b[lo + L0_F0W..lo + L0_F0W + FF * D].copy_from_slice(&blk.f0.w);
        b[lo + L0_F0B..lo + L0_F0B + FF].copy_from_slice(&blk.f0.b);
        b[lo + L0_F1W..lo + L0_F1W + D * FF].copy_from_slice(&blk.f1.w);
        b[lo + L0_F1B..lo + L0_F1B + D].copy_from_slice(&blk.f1.b);
    }
    b[OFF_FC..OFF_FC + QD].copy_from_slice(&v.fc.w);
    b[OFF_FC + QD] = v.fc.b[0];
    b
}

fn storage_ent(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_ent(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn load_shader(dev: &wgpu::Device) -> wgpu::ShaderModule {
    dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vit"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_shader/vit.wgsl").into()),
    })
}

fn make_pipeline_with_shader(
    dev: &wgpu::Device,
    lay: &wgpu::BindGroupLayout,
    ep: &str,
    sh: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vit"),
        bind_group_layouts: &[lay],
        push_constant_ranges: &[],
    });
    dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(ep),
        layout: Some(&pl),
        module: sh,
        entry_point: Some(ep),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}

fn storage_buf(dev: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    p_embed: wgpu::ComputePipeline,
    p_enc: wgpu::ComputePipeline,
    p_qry: wgpu::ComputePipeline,
    uni: wgpu::Buffer,
    img: wgpu::Buffer,
    tk: wgpu::Buffer,
    tkb: wgpu::Buffer,
    wt: wgpu::Buffer,
    qin: wgpu::Buffer,
    qp: wgpu::Buffer,
    out: wgpu::Buffer,
    q_cap: usize,
}

/// GPU handle for ViT forward. Cheap to clone (shared inner).
#[derive(Clone)]
pub struct VitGpu {
    inner: std::sync::Arc<Inner>,
}

impl std::fmt::Debug for VitGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VitGpu").finish_non_exhaustive()
    }
}

async fn try_init(v: &V) -> Option<Inner> {
    let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL,
        ..Default::default()
    });
    let adapter = inst
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok()?;
    let info = adapter.get_info();
    let n = info.name.to_lowercase();
    if n.contains("llvmpipe")
        || n.contains("swiftshader")
        || n.contains("software")
        || n.contains("virtual")
        || n.contains("warp")
        || n.contains("basic render")
    {
        return None;
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("vit"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .ok()?;

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vit"),
        entries: &[
            uniform_ent(0),
            storage_ent(1, true),
            storage_ent(2, false),
            storage_ent(3, true),
            storage_ent(4, true),
            storage_ent(5, false),
            storage_ent(6, true),
            storage_ent(7, false),
        ],
    });

    let shader = load_shader(&device);
    let p_embed = make_pipeline_with_shader(&device, &layout, "embed", &shader);
    let p_enc = make_pipeline_with_shader(&device, &layout, "enc", &shader);
    let p_qry = make_pipeline_with_shader(&device, &layout, "qry", &shader);

    let uni = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vit.uni"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let wt_data = pack(v);
    let wt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vit.w"),
        contents: bytemuck::cast_slice(&wt_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let img_bytes = (crate::hand::AI_IMAGE_SIZE * 4) as u64;
    let tk_bytes = ((NP * D) * 4) as u64;
    let q_cap = 512usize;
    let img = storage_buf(&device, "vit.img", img_bytes);
    let tk = storage_buf(&device, "vit.tk", tk_bytes);
    let tkb = storage_buf(&device, "vit.tkb", tk_bytes);
    let qin = storage_buf(&device, "vit.qin", (q_cap * NF * 4) as u64);
    let qp = storage_buf(&device, "vit.qp", (q_cap * 4) as u64);
    let out = storage_buf(&device, "vit.out", (q_cap * 4) as u64);

    Some(Inner {
        device,
        queue,
        layout,
        p_embed,
        p_enc,
        p_qry,
        uni,
        img,
        tk,
        tkb,
        wt,
        qin,
        qp,
        out,
        q_cap,
    })
}

impl VitGpu {
    /// One-shot init (blocking). `None` => caller must use CPU.
    pub fn try_new(v: &V) -> Option<Self> {
        if !v.val() {
            return None;
        }
        let rt = tokio::runtime::Runtime::new().ok()?;
        let inner = rt.block_on(try_init(v))?;
        Some(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Forward: frame + per-note features [nq, NF] + patch idx -> right-logits.
    pub fn forward(&self, img: &[f32], qfeat: &[f32], qpk: &[u32]) -> Option<Vec<f32>> {
        let nq = qpk.len();
        if nq == 0 {
            return Some(vec![]);
        }
        let r = &self.inner;
        if nq > r.q_cap || qfeat.len() < nq * NF {
            return None;
        }

        let mut p = VitP {
            nq: nq as u32,
            layer: 0,
            _p0: 0,
            _p1: 0,
        };
        r.queue.write_buffer(&r.uni, 0, bytemuck::bytes_of(&p));
        r.queue
            .write_buffer(&r.img, 0, bytemuck::cast_slice(&img[..img.len().min(crate::hand::AI_IMAGE_SIZE)]));
        r.queue.write_buffer(&r.qin, 0, bytemuck::cast_slice(qfeat));
        r.queue.write_buffer(&r.qp, 0, bytemuck::cast_slice(qpk));

        // One bind group for all passes: qry reads tk/tkb via bindings 2/7 and
        // never touches img, so binding 1 keeps the read-only img buffer.
        let bg = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &r.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: r.uni.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: r.img.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: r.tk.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: r.wt.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: r.qin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: r.out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: r.qp.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: r.tkb.as_entire_binding(),
                    },
                ],
                label: Some("vit.bg"),
            });

        let wg = |n: u32| -> u32 { n.div_ceil(64) };

        // Separate submits so layer uniform updates are visible per pass.
        {
            let mut e = r.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vit.embed"),
            });
            {
                let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("embed"),
                    timestamp_writes: None,
                });
                c.set_pipeline(&r.p_embed);
                c.set_bind_group(0, &bg, &[]);
                c.dispatch_workgroups(wg(NP as u32), 1, 1);
            }
            r.queue.submit(Some(e.finish()));
        }

        for li in 0..NL {
            p.layer = li as u32;
            r.queue.write_buffer(&r.uni, 0, bytemuck::bytes_of(&p));
            let mut e = r.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vit.enc"),
            });
            {
                let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("enc"),
                    timestamp_writes: None,
                });
                c.set_pipeline(&r.p_enc);
                c.set_bind_group(0, &bg, &[]);
                c.dispatch_workgroups(wg(NP as u32), 1, 1);
            }
            r.queue.submit(Some(e.finish()));
        }

        p.layer = 0;
        // After NL ping-pong enc layers: tokens in tkb iff NL is odd
        p._p0 = (NL as u32) & 1;
        r.queue.write_buffer(&r.uni, 0, bytemuck::bytes_of(&p));
        {
            let mut e = r.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vit.qry"),
            });
            {
                let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("qry"),
                    timestamp_writes: None,
                });
                c.set_pipeline(&r.p_qry);
                c.set_bind_group(0, &bg, &[]);
                c.dispatch_workgroups(wg(nq as u32), 1, 1);
            }
            r.queue.submit(Some(e.finish()));
        }

        let dst = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vit.dl"),
            size: (nq * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut e2 = r.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vit.dl"),
        });
        e2.copy_buffer_to_buffer(&r.out, 0, &dst, 0, (nq * 4) as u64);
        r.queue.submit(Some(e2.finish()));

        let sl = dst.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = r.device.poll(wgpu::PollType::Wait);
        if rx.recv().ok().and_then(|x| x.ok()).is_none() {
            return None;
        }
        let mut logits = {
            let data = sl.get_mapped_range();
            let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            v
        };
        dst.unmap();
        if logits.len() < nq {
            logits.resize(nq, 0.0);
        }
        if logits.iter().any(|x| x.is_nan()) {
            return None;
        }
        Some(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_len_matches_layout() {
        let v = V::n();
        let b = pack(&v);
        assert_eq!(b.len(), offs::W_LEN);
        // OFF_FC(115136) + QD(128) + 1 bias
        assert_eq!(b.len(), 115265);
    }

    #[test]
    fn vit_wgsl_parses() {
        // Same front-end as wgpu create_shader_module — fails fast on syntax.
        match naga::front::wgsl::parse_str(include_str!("gpu_shader/vit.wgsl")) {
            Ok(m) => {
                let mut validator = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                );
                if let Err(e) = validator.validate(&m) {
                    panic!("vit.wgsl validation failed: {e}");
                }
            }
            Err(e) => {
                panic!(
                    "vit.wgsl parse failed:\n{}",
                    e.emit_to_string(include_str!("gpu_shader/vit.wgsl"))
                );
            }
        }
    }

    /// End-to-end: creates pipelines (catches layout mismatches) and runs
    /// embed/enc/qry. Skips when no GPU adapter is available.
    #[test]
    fn vit_gpu_forward_runs() {
        let v = V::n();
        let Some(g) = VitGpu::try_new(&v) else {
            eprintln!("skip: no GPU adapter");
            return;
        };
        let img = vec![0.5f32; crate::hand::AI_IMAGE_SIZE];
        let nq = 3usize;
        let qfeat = vec![0.1f32; nq * NF];
        let qpk = vec![0u32, 70, 143];
        let out = g
            .forward(&img, &qfeat, &qpk)
            .expect("gpu forward failed (validation error or NaN)");
        assert_eq!(out.len(), nq);
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
