use crate::board::HexGameState;
use crate::core::Hex;
use crate::encoder::BOARD_AREA;
use crate::mcts::MCTSEngine;

#[cfg(test)]
mod tests {
    use super::*;

    /// Run MCTS twice with the same deterministic setup; verify visit distributions match.
    #[test]
    fn mcts_deterministic_replay() {
        let game = HexGameState::new();
        let mut engine1 =
            MCTSEngine::with_arena_sim_hint(game.clone(), 50, 200, 1.5, 2, false, 19652.0, 0);
        // init_root returns (tensor, offset_q, offset_r, legal_moves)
        let (_tensor1, oq, or_, legal1) = engine1.init_root().expect("init_root");
        // Expand with uniform policy
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        engine1.expand_root(&uniform, 0.0, oq, or_, &legal1);

        while !engine1.done() {
            let (_, count) = engine1.select_leaves(8);
            let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
            let values = vec![0.0f32; count as usize];
            engine1.expand_and_backprop(&policies, &values);
        }
        let (_, _, visits1, _) = engine1.get_results();

        // Second identical run
        let game2 = HexGameState::new();
        let mut engine2 =
            MCTSEngine::with_arena_sim_hint(game2, 50, 200, 1.5, 2, false, 19652.0, 0);
        let (_tensor2, oq2, or2, legal2) = engine2.init_root().expect("init_root");
        let uniform2 = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        engine2.expand_root(&uniform2, 0.0, oq2, or2, &legal2);

        while !engine2.done() {
            let (_, count) = engine2.select_leaves(8);
            let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
            let values = vec![0.0f32; count as usize];
            engine2.expand_and_backprop(&policies, &values);
        }
        let (_, _, visits2, _) = engine2.get_results();
        assert_eq!(
            visits1, visits2,
            "MCTS visit distributions must be deterministic"
        );
    }

    /// After re-rooting to a child, the new root's visit count must equal
    /// the child's visit count before re-rooting.
    #[test]
    fn mcts_reroot_visit_counts_preserved() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 100, 300, 1.5, 2, false, 19652.0, 0);
        let (_tensor, oq, or_, legal) = engine.init_root().expect("init_root");
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        engine.expand_root(&uniform, 0.0, oq, or_, &legal);

        while !engine.done() {
            let (_, count) = engine.select_leaves(8);
            let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
            let values = vec![0.0f32; count as usize];
            engine.expand_and_backprop(&policies, &values);
        }
        let (moves_q, moves_r, visits, _) = engine.get_results();
        assert!(!visits.is_empty(), "no children after search");

        // Find the child with the most visits
        let mut best_idx = 0;
        for i in 1..visits.len() {
            if visits[i] > visits[best_idx] {
                best_idx = i;
            }
        }
        let visits_before = visits[best_idx];
        let best_q = moves_q[best_idx];
        let best_r = moves_r[best_idx];

        engine
            .re_root(best_q as i16, best_r as i16, 50)
            .expect("re_root should find child");
        let root_visits = engine.arena[engine.root_idx as usize].visit_count;
        assert_eq!(root_visits, visits_before, "re_root: visit count mismatch");
    }

    /// After any number of simulations with values in [-1, 1], root Q must stay in [-1, 1].
    #[test]
    fn mcts_root_value_bounded() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 80, 300, 1.5, 2, false, 19652.0, 0);
        let (_tensor, oq, or_, legal) = engine.init_root().expect("init_root");
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        engine.expand_root(&uniform, 0.0, oq, or_, &legal);

        // Use random values in [-1, 1] for each batch
        let mut seed = 42u64;
        while !engine.done() {
            let (_, count) = engine.select_leaves(8);
            let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
            let values: Vec<f32> = (0..count as usize)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (seed as f32 / u64::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            engine.expand_and_backprop(&policies, &values);
        }
        let (_, _, _, root_q) = engine.get_results();
        assert!(
            (-1.0..=1.0).contains(&root_q),
            "root Q {} out of range [-1, 1]",
            root_q
        );
    }

    /// Verify that T1-4's assertion fires on wrong-length batch.
    #[test]
    #[should_panic(expected = "policies length")]
    fn mcts_expand_and_backprop_wrong_length_panics() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 100, 300, 1.5, 2, false, 19652.0, 0);
        let (_tensor, oq, or_, legal) = engine.init_root().expect("init_root");
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        engine.expand_root(&uniform, 0.0, oq, or_, &legal);

        // Run a few selections to create pending leaves
        let (_, count) = engine.select_leaves(8);

        // Give wrong-length policies (should be count * BOARD_AREA)
        let wrong_policies = vec![0.0f32; BOARD_AREA]; // should be count * BOARD_AREA
        let values = vec![0.0f32; count as usize];
        engine.expand_and_backprop(&wrong_policies, &values);
    }

    /// After select_leaves but before expand_and_backprop, done() must be false
    /// when num_simulations > batch_size.
    #[test]
    fn mcts_done_not_true_before_backprop() {
        let game = HexGameState::new();
        let mut engine =
            MCTSEngine::with_arena_sim_hint(game.clone(), 100, 300, 1.5, 2, false, 0.0, 0);
        let (_tensor, oq, or_, legal) = engine.init_root().expect("init_root");
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        engine.expand_root(&uniform, 0.0, oq, or_, &legal);

        let (_, count) = engine.select_leaves(8);
        assert!(count > 0, "expected non-zero leaves");
        assert!(
            !engine.done(),
            "done() must be false after select_leaves but before backprop"
        );

        let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
        let values = vec![0.0f32; count as usize];
        engine.expand_and_backprop(&policies, &values);
        assert!(
            !engine.done(),
            "done() must be false after only 8 of 100 sims"
        );
    }

    #[test]
    fn mcts_backprop_does_not_flip_between_same_player_placements() {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening move");
        assert_eq!(game.current_player(), 1);
        assert_eq!(game.placements_remaining(), 2);

        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let (_tensor, oq, or_, legal) = engine.init_root().expect("init_root");
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        engine.expand_root(&uniform, 0.0, oq, or_, &legal);

        let (_, count) = engine.select_leaves(1);
        assert_eq!(count, 1);
        let policies = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        let values = vec![1.0f32];
        engine.expand_and_backprop(&policies, &values);

        let (_, _, _, root_q) = engine.get_results();
        assert!(
            root_q > 0.5,
            "same-player placement edge should preserve value sign, got {root_q}"
        );
    }

    #[test]
    fn mcts_sparse_stage2_prior_prefers_outside_sparse_action() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let legal = vec![Hex::new(50, 50), Hex::new(0, 0)];
        let mut dense = vec![-10.0f32; BOARD_AREA];
        dense[16 * 33 + 16] = 10.0;
        let sparse_actions = vec![(50, 50)];
        let sparse_logits = vec![20.0f32];

        engine.expand_root_with_sparse_priors(
            &dense,
            0.0,
            -16,
            -16,
            &legal,
            &sparse_actions,
            &sparse_logits,
            2,
            0.25,
        );
        let priors = engine.root_child_priors();

        assert!(priors[0] > priors[1], "outside sparse action should win stage2 prior");
    }

    #[test]
    fn mcts_sparse_stage1_no_overlap_falls_back_to_dense() {
        let game = HexGameState::new();
        let mut dense_engine =
            MCTSEngine::with_arena_sim_hint(game.clone(), 1, 50, 1.5, 2, false, 0.0, 0);
        let mut sparse_engine =
            MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let legal = vec![Hex::new(0, 0), Hex::new(1, 0)];
        let mut dense = vec![-10.0f32; BOARD_AREA];
        dense[16 * 33 + 16] = 3.0;
        dense[17 * 33 + 16] = 1.0;
        dense_engine.expand_root(&dense, 0.0, -16, -16, &legal);
        sparse_engine.expand_root_with_sparse_priors(
            &dense,
            0.0,
            -16,
            -16,
            &legal,
            &[(50, 50)],
            &[20.0],
            1,
            1.0,
        );

        assert_eq!(sparse_engine.root_child_priors(), dense_engine.root_child_priors());
    }

    #[test]
    fn mcts_pending_leaf_metadata_matches_selected_leaves() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 8, 50, 1.5, 2, false, 0.0, 0);
        let (_tensor, oq, or_, legal) = engine.init_root().expect("init_root");
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        engine.expand_root(&uniform, 0.0, oq, or_, &legal);

        let (_, count) = engine.select_leaves(4);
        let meta = engine.pending_leaf_metadata();

        assert_eq!(meta.len(), count as usize);
        assert!(meta.iter().all(|(_, _, legal)| !legal.is_empty()));
    }
}
