mod extra;
pub use extra::parse_extra;

mod pec;
pub use pec::parse_pec;

mod pgr;
pub use pgr::parse_phigros;

mod rpe;
pub use rpe::{parse_rpe, RPE_HEIGHT, RPE_WIDTH};

pub(crate) fn process_lines(v: &mut [crate::core::JudgeLine]) {
    use crate::ext::NotNanExt;
    use ordered_float::NotNan;
    use std::collections::HashMap;
    let line_count = v.len();
    for line in v.iter_mut() {
        if let Some(parent_index) = line.parent {
            if parent_index >= line_count {
                line.parent = None;
            }
        }
    }

    // A note gets `multiple_hint` (the wider "double note" glyph) when at least
    // one other note *anywhere in the chart* shares its exact time.
    //
    // This used to build a per-line index array (O(n log n)), concatenate every
    // note time — duplicating the ones belonging to a same-time group — sort the
    // whole list (O(n log n)) and finally walk it with a merge cursor. Counting
    // occurrences directly is a single O(n) pass plus one O(n) lookup pass, and
    // it never touches the note order.
    let mut time_counts: HashMap<NotNan<f32>, u32> = HashMap::new();
    for line in v.iter() {
        for note in &line.notes {
            *time_counts.entry(note.time.not_nan()).or_default() += 1;
        }
    }
    if time_counts.values().any(|&c| c >= 2) {
        for line in v.iter_mut() {
            for note in line.notes.iter_mut() {
                if time_counts.get(&note.time.not_nan()).copied().unwrap_or(0) >= 2 {
                    note.multiple_hint = true;
                }
            }
        }
    }
}

#[rustfmt::skip]
pub const RPE_TWEEN_MAP: [crate::core::TweenId; 30] = {
    use crate::core::{easing_from as e, TweenMajor::*, TweenMinor::*};
    [
        2, 2, // linear
        e(Sine, Out), e(Sine, In),
        e(Quad, Out), e(Quad, In),
        e(Sine, InOut), e(Quad, InOut),
        e(Cubic, Out), e(Cubic, In),
        e(Quart, Out), e(Quart, In),
        e(Cubic, InOut), e(Quart, InOut),
        e(Quint, Out), e(Quint, In),
        e(Expo, Out), e(Expo, In),
        e(Circ, Out), e(Circ, In),
        e(Back, Out), e(Back, In),
        e(Circ, InOut), e(Back, InOut),
        e(Elastic, Out), e(Elastic, In),
        e(Bounce, Out), e(Bounce, In),
        e(Bounce, InOut), e(Elastic, InOut),
    ]
};
