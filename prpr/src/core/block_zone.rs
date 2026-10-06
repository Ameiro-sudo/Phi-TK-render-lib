//! Resolved block-area ("noise field") rectangle.
//!
//! Extracted from Phira Pro's `block_shader.rs` (GPL-3.0) so that the flat
//! CPU renderer can be used without the GPU effect path: the same `Zone` is
//! produced either way, only the draw call differs.

use super::block::{BlockArea, BlockPhase};
use super::Vector;

/// A resolved rectangle (axis-aligned in its own space).
#[derive(Clone, PartialEq)]
pub struct Zone {
    pub center: Vector,
    pub half: Vector,
    pub angle: f32,
    pub invert: bool,
    pub active: bool,
    /// Ready layer is selected in the last 0.5 seconds before enableTime.
    pub ready: bool,
    /// Initial DisabledBlockShow fades the sprite mask in over 0.5 seconds.
    pub opacity: f32,
}

impl Zone {
    pub fn from_area(area: &BlockArea, time: f64, aspect: f32) -> Option<Self> {
        let phase = area.phase(time);
        if phase == BlockPhase::Hidden {
            return None;
        }
        let tr = area.transform(time, aspect);
        let half = tr.size.map(|v| v.abs() * 0.5);
        if half.x == 0. || half.y == 0. {
            return None;
        }
        let active = phase == BlockPhase::Active;
        let fades_in = !area.is_active(area.appear_time);
        Some(Self {
            center: tr.center,
            half,
            angle: tr.rotation.to_radians(),
            invert: area.is_subtract,
            active,
            ready: !active && time < area.enable_time && time >= area.enable_time - 0.5,
            opacity: if !active && fades_in {
                ((time - area.appear_time) / 0.5).clamp(0., 1.) as f32
            } else {
                1.
            },
        })
    }
}
