use macroquad::{
    texture::{RenderTarget, Texture2D},
    window::get_internal_gl,
    miniquad::{gl::GLuint, RenderPass, Texture, TextureFormat},
};
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Bumped once at the start of every chart update cycle. `read_pixels_resized`
/// memoizes its result within a cycle, so a single full-frame GPU readback is
/// shared by *all* judge lines instead of performing one per line per request.
static READBACK_CYCLE: AtomicU64 = AtomicU64::new(0);

pub fn begin_readback_cycle() {
    READBACK_CYCLE.fetch_add(1, Ordering::Relaxed);
}

struct ReadbackState {
    /// (readback cycle, fbo id, result length, rgb data in 0..=1)
    cache: Option<(u64, GLuint, usize, Vec<f32>)>,
    /// ~6 MB staging buffer, reused across readbacks
    staging: Vec<u8>,
}

static READBACK_STATE: Lazy<Mutex<ReadbackState>> = Lazy::new(|| {
    Mutex::new(ReadbackState { cache: None, staging: Vec::new() })
});

pub struct MSRenderTarget {
    dim: (u32, u32),
    fbo: GLuint,
    rbo: GLuint,
    dummy: RenderTarget,
    output: [RenderTarget; 2],
}

pub fn copy_fbo(src: GLuint, dst: GLuint, dim: (u32, u32)) -> bool {
    unsafe {
                use macroquad::miniquad::gl::*;
                glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst);
                let (w, h) = (dim.0 as i32, dim.1 as i32);
                glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
                glGetError() == GL_NO_ERROR
            }}

pub fn internal_id(target: &RenderTarget) -> GLuint {
    target.render_pass.gl_internal_id(unsafe { get_internal_gl() }.quad_context)
}

impl MSRenderTarget {
    pub fn new(dim: (u32, u32), samples: u32) -> Self {
        let mut fbo = 0;
        let mut rbo = 0;
        unsafe {
            use macroquad::miniquad::gl::*;
            glGenRenderbuffers(1, &mut rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, rbo);
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples as i32, GL_RGB8, dim.0 as i32, dim.1 as i32);
            glGenFramebuffers(1, &mut fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
        }

        let gl = unsafe { get_internal_gl() };

        // 创建输出目标
        let mut create_target = || {
            let texture = Texture::new_render_texture(
                gl.quad_context,
                macroquad::miniquad::TextureParams {
                    width: dim.0,
                    height: dim.1,
                    format: TextureFormat::RGB8,
                    ..Default::default()
                },
            );
            RenderTarget {
                texture: Texture2D::from_miniquad_texture(texture),
                render_pass: RenderPass::new(gl.quad_context, texture, None),
            }
        };

        let output1 = create_target();
        let output2 = create_target();

        // 创建dummy纹理
        let dummy_texture = Texture::new_render_texture(
            gl.quad_context,
            macroquad::miniquad::TextureParams {
                width: dim.0,
                height: dim.1,
                format: TextureFormat::RGB8,
                ..Default::default()
            },
        );

        Self {
            dim,
            fbo,
            rbo,
            dummy: RenderTarget {
                texture: Texture2D::from_miniquad_texture(dummy_texture),
                render_pass: RenderPass::from_raw(gl.quad_context, fbo, dummy_texture),
            },
            output: [output1, output2],
        }
    }

    pub fn blit(&self) {
        copy_fbo(self.fbo, internal_id(&self.output[0]), self.dim);
    }

    pub fn swap(&mut self) {
        self.output.swap(0, 1);
    }

    pub fn input(&self) -> RenderTarget {
        self.dummy
    }

    pub fn output(&self) -> RenderTarget {
        self.output[0]
    }

    pub fn old(&self) -> RenderTarget {
        self.output[1]
    }

    /// Frame readback for the AI. Memoized per readback cycle (see
    /// `begin_readback_cycle`): the chart is read at most once per update no
    /// matter how many judge lines ask for it.
    pub fn read_pixels_resized(&self, target_w: usize, target_h: usize) -> Vec<f32> {
        let cycle = READBACK_CYCLE.load(Ordering::Relaxed);
        let fbo = internal_id(&self.output[0]);
        let len = target_w * target_h * 3;

        let mut state = READBACK_STATE.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((cached_cycle, cached_fbo, cached_len, data)) = state.cache.as_ref() {
            if *cached_cycle == cycle && *cached_fbo == fbo && *cached_len == len {
                return data.clone();
            }
        }
        let fresh = self.read_pixels_resized_uncached(&mut state.staging, target_w, target_h);
        state.cache = Some((cycle, fbo, len, fresh.clone()));
        fresh
    }

    fn read_pixels_resized_uncached(&self, staging: &mut Vec<u8>, target_w: usize, target_h: usize) -> Vec<f32> {
        let (src_w, src_h) = self.dim;
        let total_pixels = (src_w * src_h * 3) as usize;
        let fbo = internal_id(&self.output[0]);

        // Keep the ~6 MB staging buffer alive across calls instead of
        // allocating a fresh one for every readback.
        staging.clear();
        staging.resize(total_pixels, 0);

        unsafe {
            use macroquad::miniquad::gl::*;
            glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
            glReadPixels(0, 0, src_w as i32, src_h as i32, GL_RGB, GL_UNSIGNED_BYTE, staging.as_mut_ptr() as *mut _);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        }

        let mut result = Vec::with_capacity(target_w * target_h * 3);
        for ty in 0..target_h {
            for tx in 0..target_w {
                let sx = ((tx * src_w as usize) / target_w).min(src_w as usize - 1);
                let sy = ((ty * src_h as usize) / target_h).min(src_h as usize - 1);
                let src_idx = (sy * src_w as usize + sx) * 3;
                result.push(staging[src_idx] as f32 / 255.0);
                result.push(staging[src_idx + 1] as f32 / 255.0);
                result.push(staging[src_idx + 2] as f32 / 255.0);
            }
        }
        result
    }
}

impl Drop for MSRenderTarget {
    fn drop(&mut self) {
        unsafe {
            use miniquad::gl::*;
            glDeleteRenderbuffers(1, &self.rbo);
            glDeleteFramebuffers(1, &self.fbo);
        }
    }
}