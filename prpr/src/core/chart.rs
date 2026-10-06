use super::{
    draw_block_zones_simple, BlockArea, BpmList, Effect, JudgeLine, JudgeLineKind, Matrix, Resource, UIElement, Vector, Video, Zone,
};
use crate::{fs::FileSystem, judge::JudgeStatus, ui::Ui};
use anyhow::{Context, Result};
use macroquad::prelude::*;
use std::cell::RefCell;
use tracing::warn;

/// Per-frame cache of the resolved 噪域 rectangles.
#[derive(Default)]
struct BlockFrame {
    timeline: super::block_timeline::BlockTimeline,
    // `Resource::time` is an f32 in this fork, so the cache key is f32 too.
    key: Option<(f32, f32, usize)>,
    zones: Vec<Zone>,
}

//use rayon::prelude::*;

#[derive(Default)]
pub struct ChartExtra {
    pub effects: Vec<Effect>,
    pub global_effects: Vec<Effect>,
    pub videos: Vec<Video>,
}

#[derive(Default)]
pub struct ChartSettings {
    pub pe_alpha_extension: bool,
    pub hold_partial_cover: bool,
}

pub struct Chart {
    pub offset: f32,
    pub lines: Vec<JudgeLine>,
    pub bpm_list: RefCell<BpmList>,
    pub settings: ChartSettings,
    pub extra: ChartExtra,
    pub order: Vec<usize>,
    pub attach_ui: [Option<usize>; 7],
    /// Phigros 9th-chapter 噪域 (`blockAreaList`). Empty for every older chart.
    pub block_areas: Vec<BlockArea>,
    block_frame: RefCell<BlockFrame>,
    world_positions: Vec<Vector>,
    trs: Vec<Matrix>,
}

impl Chart {
    pub fn new(offset: f32, lines: Vec<JudgeLine>, bpm_list: BpmList, settings: ChartSettings, extra: ChartExtra) -> Self {
        let mut attach_ui = [None; 7];
        let mut order = (0..lines.len())
            .filter(|it| {
                if let Some(element) = lines[*it].attach_ui {
                    attach_ui[element as usize - 1] = Some(*it);
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        order.sort_by_key(|it| (lines[*it].z_index, *it));
        let capacity = lines.len();
        Self {
            offset,
            lines,
            bpm_list: RefCell::new(bpm_list),
            settings,
            extra,

            order,
            attach_ui,
            block_areas: Vec::new(),
            block_frame: RefCell::default(),
            world_positions: Vec::with_capacity(capacity),
            trs: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn with_element<R>(&self, ui: &mut Ui, res: &Resource, element: UIElement, ct: Option<(f32, f32)>, pt: Option<(f32, f32)>, f: impl FnOnce(&mut Ui, Color) -> R) -> R {
        if let Some(id) = self.attach_ui[element as usize - 1] {
            let lines = &self.lines;
            let line = &lines[id];
            let obj = &line.object;
            let mut tr = JudgeLine::fetch_pos(line, res, lines);
            tr.y = -tr.y;
            let mut color = self.lines[id].color.now_opt().unwrap_or(WHITE);
            color.a *= obj.now_alpha().max(0.);
            let scale = obj.now_scale_fix(ct.map_or_else(|| Vector::default(), |(x, y)| Vector::new(x, y)));
            let ro = obj.new_rotation_wrt_point(-obj.rotation.now().to_radians(), pt.map_or_else(|| Vector::default(), |(x, y)| Vector::new(x, y)));
            ui.with(Matrix::new_translation(&tr) * ro * scale, |ui| f(ui, color))
        } else {
            f(ui, WHITE)
        }
    }

    pub fn with_element_noscale<R>(&self, ui: &mut Ui, res: &Resource, element: UIElement, ct: Option<(f32, f32)>, f: impl FnOnce(&mut Ui, Color) -> R) -> R {
        if let Some(id) = self.attach_ui[element as usize - 1] {
            let obj = &self.lines[id].object;
            let mut tr = obj.now_translation(res);
            tr.y = -tr.y;
            let mut color = self.lines[id].color.now_opt().unwrap_or(WHITE);
            color.a *= obj.now_alpha().max(0.);
            let mut scale = obj.now_scale_fix(ct.map_or_else(|| Vector::default(), |(x, y)| Vector::new(x , y)));
            scale.m11 = 1.0;
            ui.with(obj.now_rotation().append_translation(&tr) * scale, |ui| f(ui, color))
        } else {
            f(ui, WHITE)
        }
    }

    pub async fn load_textures(&mut self, fs: &mut dyn FileSystem) -> Result<()> {
        for line in &mut self.lines {
            if let JudgeLineKind::Texture(tex, path) = &mut line.kind {
                *tex = image::load_from_memory(&fs.load_file(path).await.with_context(|| format!("failed to load illustration {path}"))?)?.into();
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.lines
            .iter_mut()
            .flat_map(|it| it.notes.iter_mut())
            .for_each(|note| note.judge = JudgeStatus::NotJudged);
        for line in &mut self.lines {
            line.cache.reset(&mut line.notes);
        }
        for video in &mut self.extra.videos {
            video.next_frame = 0;
        }
    }

    pub fn update(&mut self, res: &mut Resource) {
        // All judge lines share a single full-frame GPU readback per update.
        crate::core::begin_readback_cycle();

        // `Anim::set_time` 内部已按二分定位关键帧，父线的世界坐标也由下面的
        // `fetch_pos` 按帧记忆化，这一段的每帧开销已是 O(判定线数)。
        for line in &mut self.lines {
            line.object.set_time(res.time);
            // Drop the previous frame's position *before* recomputing: a parent whose
            // index is greater than its child's would otherwise be resolved from a
            // stale value. `fetch_pos` then memoizes parents already computed this frame.
            line.cached_world_pos = None;
        }

        let count = self.lines.len();
        self.world_positions.clear();
        self.trs.clear();
        for i in 0..count {
            let pos = JudgeLine::fetch_pos(&self.lines[i], res, &self.lines);
            self.world_positions.push(pos);
            self.lines[i].cached_world_pos = Some(pos);
            self.trs.push(self.lines[i].now_transform_with_pos(pos));
        }

        let guard = self.bpm_list.borrow();
        for (index, line) in self.lines.iter_mut().enumerate() {
            line.update(res, self.trs[index], &guard, index);

            if res.config.hand_split {
                line.update_hand_assign_with_world_pos(res, self.world_positions[index], index);
            }
        }
        drop(guard);

        for effect in &mut self.extra.effects {
            effect.update(res);
        }
        for video in &mut self.extra.videos {
            if let Err(err) = video.update(res.time) {
                warn!("video error: {err:?}");
            }
        }
    }

    pub fn render(&self, ui: &mut Ui, res: &mut Resource) {
        res.apply_model_of(&Matrix::identity().append_nonuniform_scaling(&Vector::new(if res.config.flip_x() { -1. } else { 1. }, 1.)), |res| {
            for video in &self.extra.videos {
                video.render(res);
            }
        });
        res.apply_model_of(&Matrix::identity().append_nonuniform_scaling(&Vector::new(if res.config.flip_x() { -1. } else { 1. }, -1.)), |res| {
            let guard = self.bpm_list.borrow();
            for id in &self.order {
                self.lines[*id].render(ui, res, &self.lines, &guard, &self.settings, *id);
            }
            drop(guard);
            res.note_buffer.borrow_mut().draw_all();
            if res.config.sample_count > 1 {
                unsafe { get_internal_gl() }.flush();
                if let Some(target) = &res.chart_target {
                    target.blit();
                }
            }
        });
    }

    /// Draw the 噪域 (`blockAreaList`) rectangles of the current frame.
    ///
    /// Uses the flat CPU tessellator rather than Phira Pro's GPU mask/shader
    /// path: identical geometry, no render-target copies, deterministic output,
    /// which is what an offline video render needs.
    pub fn render_block_overlay(&self, res: &mut Resource) {
        if self.block_areas.is_empty() {
            return;
        }
        let aspect = res.aspect_ratio;
        {
            let zones = self.block_zones(res);
            draw_block_zones_simple(aspect, &zones, false);
        }
    }

    fn block_zones(&self, res: &Resource) -> std::cell::Ref<'_, [Zone]> {
        // The block module works in seconds (f64); this fork's `Resource::time`
        // is an f32, so widen once here.
        let now = res.time as f64;
        let key = (res.time, res.aspect_ratio, self.block_areas.len());
        {
            let mut cache = self.block_frame.borrow_mut();
            if cache.key != Some(key) {
                let BlockFrame { timeline, zones, .. } = &mut *cache;
                zones.clear();
                zones.extend(
                    timeline
                        .at(&self.block_areas, now)
                        .iter()
                        .filter_map(|&id| Zone::from_area(&self.block_areas[id], now, res.aspect_ratio)),
                );
                cache.key = Some(key);
            }
        }
        std::cell::Ref::map(self.block_frame.borrow(), |cache| cache.zones.as_slice())
    }
}