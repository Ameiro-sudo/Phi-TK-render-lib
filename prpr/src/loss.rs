use crate::core::note::{Hand, Note, NoteKind};
use crate::core::Chart;
use crate::hand_model::{ErgonomicHandSystem, Vector2};
use crate::judge::{Judgement, PlayResult, LIMIT_BAD, LIMIT_GOOD, LIMIT_PERFECT};

pub fn judgement_penalty(judgement: Judgement) -> f32 {
    match judgement {
        Judgement::Perfect => 0.0,
        Judgement::Good => 0.35,
        Judgement::Bad => 1.0,
        Judgement::Miss => 2.0,
    }
}

pub fn note_loss(judgement: Judgement, time_error: f32, hand_feasible: bool) -> f32 {
    let timing = 0.5 * time_error * time_error;
    let physical = if hand_feasible { 0.0 } else { 5.0 };
    judgement_penalty(judgement) + timing + physical
}

/// Turns physical-model errors into a judgement. The timing limits are shared
/// with `judge.rs`, so the model is graded exactly like a player.
pub fn judgement_from_errors(
    success: bool,
    position_error: f32,
    time_error: f32,
    kind: &NoteKind,
) -> Judgement {
    let judgement = if !success {
        Judgement::Miss
    } else if time_error <= LIMIT_PERFECT && position_error <= 0.1 {
        Judgement::Perfect
    } else if time_error <= LIMIT_GOOD {
        Judgement::Good
    } else if time_error <= LIMIT_BAD {
        Judgement::Bad
    } else {
        Judgement::Miss
    };
    match kind {
        NoteKind::Flick | NoteKind::Drag if matches!(judgement, Judgement::Bad) => Judgement::Good,
        _ => judgement,
    }
}

pub fn chart_loss(result: &PlayResult, feasible_flags: &[bool]) -> f32 {
    let notes = result.num_of_notes.max(1) as f32;
    let accuracy_loss = (1.0 - result.accuracy as f32).max(0.0);
    let miss_rate = result.counts[Judgement::Miss as usize] as f32 / notes;

    let timing_variance = if result.num_of_notes > 0 {
        let imbalance = (result.early as f32 - result.late as f32) / notes;
        imbalance * imbalance
    } else {
        0.0
    };

    let combo_loss = if result.num_of_notes > 0 {
        1.0 - (result.max_combo as f32 / notes).min(1.0)
    } else {
        0.0
    };

    let infeasible_rate = if feasible_flags.is_empty() {
        0.0
    } else {
        let infeasible = feasible_flags.iter().filter(|&&ok| !ok).count() as f32;
        infeasible / feasible_flags.len() as f32
    };

    const MISS_WEIGHT: f32 = 1.5;
    const TIMING_WEIGHT: f32 = 0.3;
    const COMBO_WEIGHT: f32 = 0.4;
    const INFEASIBLE_WEIGHT: f32 = 2.0;

    accuracy_loss
        + MISS_WEIGHT * miss_rate
        + TIMING_WEIGHT * timing_variance
        + COMBO_WEIGHT * combo_loss
        + INFEASIBLE_WEIGHT * infeasible_rate
}

pub fn chart_loss_from_counts(counts: [u32; 4], max_combo: u32, total: u32) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let notes = total as f32;
    let perfect = counts[Judgement::Perfect as usize] as f32;
    let good = counts[Judgement::Good as usize] as f32;
    let bad = counts[Judgement::Bad as usize] as f32;
    let miss = counts[Judgement::Miss as usize] as f32;

    let accuracy = (perfect + 0.65 * good) / notes;
    let miss_rate = miss / notes;
    let bad_rate = bad / notes;
    let combo_loss = 1.0 - (max_combo as f32 / notes).min(1.0);

    (1.0 - accuracy) + 1.5 * miss_rate + 0.4 * combo_loss + 0.8 * bad_rate
}

pub fn simulate_note_outcome(
    note: &Note,
    hand: Hand,
    hand_system: &ErgonomicHandSystem,
    note_world_x: f32,
    current_time: f32,
) -> (Judgement, f32, bool) {
    let position = Vector2::new(note_world_x, 0.0);
    let (success, position_error, time_error, _) =
        hand_system.evaluate_note_success(hand, &position, note.time, current_time, &note.kind);
    let judgement = judgement_from_errors(success, position_error, time_error, &note.kind);
    let feasible = success || time_error <= LIMIT_BAD;
    (judgement, time_error, feasible)
}

pub fn evaluate_assignments(
    chart: &Chart,
    assignments: &[Hand],
    hand_system: &ErgonomicHandSystem,
) -> (Vec<f32>, f32) {
    let mut per_note = Vec::with_capacity(assignments.len());
    let mut counts = [0u32; 4];
    let mut max_combo = 0u32;
    let mut combo = 0u32;
    let mut total = 0u32;
    let mut feasible_flags = Vec::with_capacity(assignments.len());

    let mut index = 0;
    for line in &chart.lines {
        for note in &line.notes {
            if note.fake {
                continue;
            }
            if index >= assignments.len() {
                break;
            }

            let world_x = note.object.translation.0.now_opt().unwrap_or(0.0);
            let (judgement, time_error, feasible) =
                simulate_note_outcome(note, assignments[index], hand_system, world_x, note.time);

            per_note.push(note_loss(judgement, time_error, feasible));
            feasible_flags.push(feasible);
            counts[judgement as usize] += 1;
            total += 1;
            match judgement {
                Judgement::Perfect | Judgement::Good => {
                    combo += 1;
                    if combo > max_combo {
                        max_combo = combo;
                    }
                }
                _ => combo = 0,
            }
            index += 1;
        }
    }

    let infeasible = if feasible_flags.is_empty() {
        0.0
    } else {
        2.0 * feasible_flags.iter().filter(|&&ok| !ok).count() as f32 / feasible_flags.len() as f32
    };
    (per_note, chart_loss_from_counts(counts, max_combo, total) + infeasible)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_is_zero_loss() {
        assert_eq!(note_loss(Judgement::Perfect, 0.0, true), 0.0);
    }

    #[test]
    fn miss_is_heaviest() {
        let perfect = note_loss(Judgement::Perfect, 0.0, true);
        let good = note_loss(Judgement::Good, 0.0, true);
        let bad = note_loss(Judgement::Bad, 0.0, true);
        let miss = note_loss(Judgement::Miss, 0.0, true);
        assert!(perfect < good && good < bad && bad < miss);
    }

    #[test]
    fn infeasible_hand_heavily_penalized() {
        let reachable = note_loss(Judgement::Perfect, 0.0, true);
        let unreachable = note_loss(Judgement::Perfect, 0.0, false);
        assert!(unreachable - reachable >= 4.9);
    }

    #[test]
    fn chart_loss_full_perfect_is_zero() {
        let result = PlayResult {
            score: 1_000_000,
            accuracy: 1.0,
            max_combo: 100,
            num_of_notes: 100,
            counts: [100, 0, 0, 0],
            early: 50,
            late: 50,
            std: 0.0,
        };
        let flags = vec![true; 100];
        let loss = chart_loss(&result, &flags);
        assert!(loss.abs() < 1e-6, "full perfect should be 0 loss, got {loss}");
    }

    #[test]
    fn chart_loss_counts_api() {
        assert_eq!(chart_loss_from_counts([100, 0, 0, 0], 100, 100), 0.0);
        assert!(chart_loss_from_counts([0, 0, 0, 100], 0, 100) > 2.0);
    }
}
