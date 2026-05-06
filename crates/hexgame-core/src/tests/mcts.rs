use crate::board::HexGameState;
use crate::core::Hex;
use crate::encoder::BOARD_AREA;
use crate::mcts::{MCTSEngine, MCTSError};

#[cfg(test)]
mod tests {
    use super::*;

    fn init_root_parts(engine: &mut MCTSEngine) -> (i32, i32, Vec<Hex>, u64) {
        let init = engine
            .init_root()
            .expect("init_root should not fail")
            .expect("root should be non-terminal");
        (
            init.offset_q,
            init.offset_r,
            init.legal_moves,
            init.root_generation,
        )
    }

    fn expand_root(
        engine: &mut MCTSEngine,
        policy: &[f32],
        oq: i32,
        or_: i32,
        legal: &[Hex],
        root_generation: u64,
    ) {
        engine
            .expand_root(root_generation, policy, 0.0, oq, or_, legal)
            .expect("root expansion should succeed");
    }

    fn select_leaves(engine: &mut MCTSEngine, batch_size: u32) -> (u64, u32) {
        let batch = engine
            .select_leaves(batch_size)
            .expect("leaf selection should succeed");
        (batch.batch_generation, batch.non_terminal_count)
    }

    fn expand_and_backprop(
        engine: &mut MCTSEngine,
        batch_generation: u64,
        policies: &[f32],
        values: &[f32],
    ) {
        engine
            .expand_and_backprop(batch_generation, policies, values)
            .expect("backpropagation should succeed");
    }

    /// Run MCTS twice with the same deterministic setup; verify visit distributions match.
    #[test]
    fn mcts_deterministic_replay() {
        let game = HexGameState::new();
        let mut engine1 =
            MCTSEngine::with_arena_sim_hint(game.clone(), 50, 200, 1.5, 2, false, 19652.0, 0);
        // init_root returns (tensor, offset_q, offset_r, legal_moves)
        let (oq, or_, legal1, root_generation) = init_root_parts(&mut engine1);
        // Expand with uniform policy
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine1, &uniform, oq, or_, &legal1, root_generation);

        while !engine1.done() {
            let (batch_generation, count) = select_leaves(&mut engine1, 8);
            let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
            let values = vec![0.0f32; count as usize];
            expand_and_backprop(&mut engine1, batch_generation, &policies, &values);
        }
        let (_, _, visits1, _) = engine1.get_results();

        // Second identical run
        let game2 = HexGameState::new();
        let mut engine2 =
            MCTSEngine::with_arena_sim_hint(game2, 50, 200, 1.5, 2, false, 19652.0, 0);
        let (oq2, or2, legal2, root_generation) = init_root_parts(&mut engine2);
        let uniform2 = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine2, &uniform2, oq2, or2, &legal2, root_generation);

        while !engine2.done() {
            let (batch_generation, count) = select_leaves(&mut engine2, 8);
            let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
            let values = vec![0.0f32; count as usize];
            expand_and_backprop(&mut engine2, batch_generation, &policies, &values);
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
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine, &uniform, oq, or_, &legal, root_generation);

        while !engine.done() {
            let (batch_generation, count) = select_leaves(&mut engine, 8);
            let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
            let values = vec![0.0f32; count as usize];
            expand_and_backprop(&mut engine, batch_generation, &policies, &values);
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
            .re_root(best_q, best_r, 50)
            .expect("re_root should find child");
        assert_eq!(
            engine.root_idx, 0,
            "re_root should compact selected child to arena root"
        );
        let root_visits = engine.arena[engine.root_idx as usize].visit_count;
        assert_eq!(root_visits, visits_before, "re_root: visit count mismatch");
    }

    #[test]
    fn mcts_re_root_discards_unreachable_sibling_subtrees() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 100, 300, 1.5, 2, false, 19652.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine, &uniform, oq, or_, &legal, root_generation);

        while !engine.done() {
            let (batch_generation, count) = select_leaves(&mut engine, 8);
            let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
            let values = vec![0.0f32; count as usize];
            expand_and_backprop(&mut engine, batch_generation, &policies, &values);
        }

        let arena_len_before = engine.arena.len();
        let (moves_q, moves_r, visits, _) = engine.get_results();
        let best_idx = visits
            .iter()
            .enumerate()
            .max_by_key(|(_, visits)| *visits)
            .map(|(idx, _)| idx)
            .expect("root should have children");
        let selected_visits = visits[best_idx];

        engine
            .re_root(moves_q[best_idx], moves_r[best_idx], 50)
            .expect("re_root should find child");

        assert_eq!(
            engine.root_idx, 0,
            "compacted arena should make selected child index 0"
        );
        assert_eq!(
            engine.arena[0].visit_count, selected_visits,
            "compaction must preserve selected-root visit statistics"
        );
        assert!(
            engine.arena.len() < arena_len_before,
            "re_root should drop unreachable sibling subtrees: before={} after={}",
            arena_len_before,
            engine.arena.len()
        );
    }

    #[test]
    fn mcts_child_limit_caps_global_root_and_leaf_expansion() {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening move");
        let mut engine = MCTSEngine::with_arena_sim_hint_and_child_limit(
            game,
            8,
            50,
            1.5,
            8,
            false,
            19652.0,
            0,
            Some(3),
        );
        let (_oq, _or, legal, root_generation) = init_root_parts(&mut engine);
        assert!(legal.len() > 3, "test requires a branching root");
        let global_actions: Vec<(i32, i32)> = legal.iter().map(|h| (h.q, h.r)).collect();
        let mut logits = vec![0.0f32; legal.len()];
        let preferred = legal.len() - 1;
        logits[preferred] = 10.0;

        engine
            .expand_root_with_global_priors(root_generation, &legal, &global_actions, &logits, 0.0)
            .expect("global root expansion should succeed");

        let (moves_q, moves_r, _visits, _) = engine.get_results();
        assert_eq!(moves_q.len(), 3, "root expansion should honor child cap");
        assert!(
            moves_q
                .iter()
                .zip(moves_r.iter())
                .any(|(&q, &r)| q == legal[preferred].q && r == legal[preferred].r),
            "highest-prior global action should survive the cap"
        );

        let (batch_generation, count) = select_leaves(&mut engine, 4);
        let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
        let values = vec![0.0f32; count as usize];
        expand_and_backprop(&mut engine, batch_generation, &policies, &values);

        assert!(
            engine.expanded_child_counts_within(3),
            "all expanded nodes should honor child cap"
        );
    }

    /// After any number of simulations with values in [-1, 1], root Q must stay in [-1, 1].
    #[test]
    fn mcts_root_value_bounded() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 80, 300, 1.5, 2, false, 19652.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine, &uniform, oq, or_, &legal, root_generation);

        // Use random values in [-1, 1] for each batch
        let mut seed = 42u64;
        while !engine.done() {
            let (batch_generation, count) = select_leaves(&mut engine, 8);
            let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
            let values: Vec<f32> = (0..count as usize)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (seed as f32 / u64::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            expand_and_backprop(&mut engine, batch_generation, &policies, &values);
        }
        let (_, _, _, root_q) = engine.get_results();
        assert!(
            (-1.0..=1.0).contains(&root_q),
            "root Q {} out of range [-1, 1]",
            root_q
        );
    }

    /// Verify that the canonical MCTS API returns Err on wrong-length batch.
    #[test]
    fn mcts_expand_and_backprop_wrong_length_returns_err() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 100, 300, 1.5, 2, false, 19652.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine, &uniform, oq, or_, &legal, root_generation);

        // Run a few selections to create pending leaves
        let (batch_generation, count) = {
            let batch = engine.select_leaves(8).expect("select_leaves");
            (batch.batch_generation, batch.non_terminal_count)
        };
        assert!(count > 0);

        // Give wrong-length policies (should be count * BOARD_AREA)
        let wrong_policies = vec![0.0f32; count as usize * BOARD_AREA - 1];
        let values = vec![0.0f32; count as usize];
        let err = engine
            .expand_and_backprop(batch_generation, &wrong_policies, &values)
            .expect_err("wrong-length batch should return Err");
        assert!(matches!(err, MCTSError::WrongPolicyLength { .. }));
    }

    #[test]
    fn mcts_i32_far_coordinate_roundtrip() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 10, 1.5, 2, false, 0.0, 0);
        let far = Hex::new(50_000, -50_000);
        let legal = vec![far, Hex::new(0, 0)];
        let policy = vec![0.0f32; BOARD_AREA];
        let (_, _, _, root_generation) = init_root_parts(&mut engine);

        expand_root(&mut engine, &policy, -16, -16, &legal, root_generation);

        let (moves_q, moves_r, _visits, _root_q) = engine.get_results();
        assert_eq!((moves_q[0], moves_r[0]), (far.q, far.r));
        engine.arena[1].visit_count = 1;
        let mut rng = 1;
        assert_eq!(
            engine.sample_action(0.0, &mut rng).expect("sample action"),
            (far.q, far.r)
        );
    }

    #[test]
    fn mcts_stale_root_token_rejected() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 10, 1.5, 2, false, 0.0, 0);
        let init1 = engine.init_root().expect("init_root").expect("root init");
        let init2 = engine.init_root().expect("init_root").expect("root init");
        let policy = vec![0.0f32; BOARD_AREA];

        let err = engine
            .expand_root(
                init1.root_generation,
                &policy,
                0.0,
                init1.offset_q,
                init1.offset_r,
                &init1.legal_moves,
            )
            .expect_err("stale root token should be rejected");
        assert!(matches!(
            err,
            MCTSError::StaleRootToken {
                expected,
                received
            } if expected == init2.root_generation && received == init1.root_generation
        ));

        engine
            .expand_root(
                init2.root_generation,
                &policy,
                0.0,
                init2.offset_q,
                init2.offset_r,
                &init2.legal_moves,
            )
            .expect("current root token should expand");
    }

    #[test]
    fn mcts_stale_batch_token_rejected() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 8, 50, 1.5, 2, false, 0.0, 0);
        let init = engine.init_root().expect("init_root").expect("root init");
        let policy = vec![0.0f32; BOARD_AREA];
        engine
            .expand_root(
                init.root_generation,
                &policy,
                0.0,
                init.offset_q,
                init.offset_r,
                &init.legal_moves,
            )
            .expect("expand root");

        let first_token = {
            let batch = engine.select_leaves(2).expect("first select");
            assert!(batch.non_terminal_count > 0);
            batch.batch_generation
        };
        let (second_token, second_count) = {
            let batch = engine.select_leaves(2).expect("second select");
            assert!(batch.non_terminal_count > 0);
            (batch.batch_generation, batch.non_terminal_count)
        };
        let policies = vec![0.0f32; second_count as usize * BOARD_AREA];
        let values = vec![0.0f32; second_count as usize];

        let err = engine
            .expand_and_backprop(first_token, &policies, &values)
            .expect_err("stale batch token should be rejected");
        assert!(matches!(
            err,
            MCTSError::StaleBatchToken { expected, received }
                if expected == second_token && received == first_token
        ));

        engine
            .expand_and_backprop(second_token, &policies, &values)
            .expect("current batch token should backpropagate");
    }

    #[test]
    fn mcts_canonical_happy_path_runs_to_completion() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 6, 50, 1.5, 2, false, 0.0, 0);
        let init = engine.init_root().expect("init_root").expect("root init");
        let policy = vec![0.0f32; BOARD_AREA];
        engine
            .expand_root(
                init.root_generation,
                &policy,
                0.0,
                init.offset_q,
                init.offset_r,
                &init.legal_moves,
            )
            .expect("expand root");

        while !engine.done() {
            let (batch_generation, count) = {
                let batch = engine.select_leaves(3).expect("select leaves");
                (batch.batch_generation, batch.non_terminal_count)
            };
            let policies = vec![0.0f32; count as usize * BOARD_AREA];
            let values = vec![0.0f32; count as usize];
            engine
                .expand_and_backprop(batch_generation, &policies, &values)
                .expect("backprop batch");
        }

        let (_moves_q, _moves_r, visits, root_q) = engine.get_results();
        assert_eq!(visits.iter().sum::<u32>(), 6);
        assert!(root_q.is_finite());
    }

    #[test]
    fn mcts_neutral_rollouts_complete_without_leaf_tensors() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint_and_child_limit(
            game,
            12,
            50,
            1.5,
            2,
            false,
            0.0,
            0,
            Some(8),
        );
        let init = engine.init_root().expect("init_root").expect("root init");
        let policy = vec![0.0f32; BOARD_AREA];
        engine
            .expand_root(
                init.root_generation,
                &policy,
                0.0,
                init.offset_q,
                init.offset_r,
                &init.legal_moves,
            )
            .expect("expand root");

        let first = engine
            .run_neutral_rollouts(5, 0.0)
            .expect("neutral rollout batch");
        assert_eq!(first, 5);
        assert!(!engine.done());
        let second = engine
            .run_neutral_rollouts(99, 0.0)
            .expect("neutral rollout remainder");
        assert_eq!(second, 7);
        assert!(engine.done());

        let (_moves_q, _moves_r, visits, root_q) = engine.get_results();
        assert_eq!(visits.iter().sum::<u32>(), 12);
        assert!(root_q.is_finite());
        assert!(
            engine.expanded_child_counts_within(8),
            "neutral expansion must respect the configured child cap"
        );
    }

    /// After select_leaves but before expand_and_backprop, done() must be false
    /// when num_simulations > batch_size.
    #[test]
    fn mcts_done_not_true_before_backprop() {
        let game = HexGameState::new();
        let mut engine =
            MCTSEngine::with_arena_sim_hint(game.clone(), 100, 300, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine, &uniform, oq, or_, &legal, root_generation);

        let (batch_generation, count) = select_leaves(&mut engine, 8);
        assert!(count > 0, "expected non-zero leaves");
        assert!(
            !engine.done(),
            "done() must be false after select_leaves but before backprop"
        );

        let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
        let values = vec![0.0f32; count as usize];
        expand_and_backprop(&mut engine, batch_generation, &policies, &values);
        assert!(
            !engine.done(),
            "done() must be false after only 8 of 100 sims"
        );
    }

    #[test]
    fn mcts_reroot_clears_pending_after_failed_batch() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 100, 300, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine, &uniform, oq, or_, &legal, root_generation);

        let (moves_q, moves_r, visits_before, _) = engine.get_results();
        assert!(!visits_before.is_empty(), "root should have children");
        let (_batch_generation, count) = select_leaves(&mut engine, 8);
        assert!(count > 0, "expected pending non-terminal leaves");
        assert!(
            !engine.pending_leaf_metadata().is_empty(),
            "select_leaves should leave pending metadata before backprop"
        );

        engine
            .re_root(moves_q[0], moves_r[0], 50)
            .expect("re_root should clear pending leaves and continue");
        assert!(
            engine.pending_leaf_metadata().is_empty(),
            "re_root must not carry stale pending leaves across moves"
        );
    }

    #[test]
    fn mcts_repeated_select_rolls_back_previous_virtual_loss() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 10, 50, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine, &uniform, oq, or_, &legal, root_generation);

        let root_idx = engine.root_idx as usize;
        assert_eq!(engine.arena[root_idx].visit_count, 0);
        let (_batch_generation, first_count) = select_leaves(&mut engine, 1);
        assert_eq!(first_count, 1);
        let first_pending_visit_count = engine.arena[root_idx].visit_count;
        assert_eq!(first_pending_visit_count, 1);

        let (_batch_generation, second_count) = select_leaves(&mut engine, 1);
        assert_eq!(second_count, 1);
        assert_eq!(
            engine.arena[root_idx].visit_count, first_pending_visit_count,
            "select_leaves must roll back abandoned pending virtual loss before selecting again"
        );
    }

    #[test]
    fn mcts_backprop_does_not_flip_between_same_player_placements() {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening move");
        assert_eq!(game.current_player(), 1);
        assert_eq!(game.placements_remaining(), 2);

        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine, &uniform, oq, or_, &legal, root_generation);

        let (batch_generation, count) = select_leaves(&mut engine, 1);
        assert_eq!(count, 1);
        let policies = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        let values = vec![1.0f32];
        expand_and_backprop(&mut engine, batch_generation, &policies, &values);

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
        let (_, _, _, root_generation) = init_root_parts(&mut engine);

        engine
            .expand_root_with_sparse_priors(
                root_generation,
                &dense,
                0.0,
                -16,
                -16,
                &legal,
                &sparse_actions,
                &sparse_logits,
                2,
                0.25,
            )
            .expect("sparse root priors should apply");
        let priors = engine.root_child_priors();

        assert!(
            priors[0] > priors[1],
            "outside sparse action should win stage2 prior"
        );
    }

    #[test]
    fn mcts_sparse_stage1_no_overlap_falls_back_to_dense() {
        let game = HexGameState::new();
        let mut dense_engine =
            MCTSEngine::with_arena_sim_hint(game.clone(), 1, 50, 1.5, 2, false, 0.0, 0);
        let mut sparse_engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let legal = vec![Hex::new(0, 0), Hex::new(1, 0)];
        let mut dense = vec![-10.0f32; BOARD_AREA];
        dense[16 * 33 + 16] = 3.0;
        dense[17 * 33 + 16] = 1.0;
        let (_, _, _, dense_root_generation) = init_root_parts(&mut dense_engine);
        let (_, _, _, sparse_root_generation) = init_root_parts(&mut sparse_engine);
        expand_root(
            &mut dense_engine,
            &dense,
            -16,
            -16,
            &legal,
            dense_root_generation,
        );
        sparse_engine
            .expand_root_with_sparse_priors(
                sparse_root_generation,
                &dense,
                0.0,
                -16,
                -16,
                &legal,
                &[(50, 50)],
                &[20.0],
                1,
                1.0,
            )
            .expect("sparse root priors should apply");

        assert_eq!(
            sparse_engine.root_child_priors(),
            dense_engine.root_child_priors()
        );
    }

    #[test]
    fn mcts_sparse_stage1_only_uses_sparse_at_root() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 4, 50, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let dense = vec![0.0f32; BOARD_AREA];
        let sparse_actions = vec![(legal[0].q, legal[0].r)];
        let sparse_logits = vec![10.0f32];

        engine
            .expand_root_with_sparse_priors(
                root_generation,
                &dense,
                0.0,
                oq,
                or_,
                &legal,
                &sparse_actions,
                &sparse_logits,
                1,
                1.0,
            )
            .expect("sparse root priors should apply");
        let root_sources = engine.root_child_prior_sources();
        assert_eq!(
            root_sources[0], 1,
            "stage1 root should consume sparse prior"
        );

        let (batch_generation, count) = select_leaves(&mut engine, 2);
        assert!(count > 0);
        let policies = vec![0.0f32; count as usize * BOARD_AREA];
        let values = vec![0.0f32; count as usize];
        let leaf_sparse = vec![sparse_actions.clone(); count as usize];
        let leaf_logits = vec![sparse_logits.clone(); count as usize];
        engine
            .expand_and_backprop_with_sparse(
                batch_generation,
                &policies,
                &values,
                &leaf_sparse,
                &leaf_logits,
                1,
                1.0,
            )
            .expect("sparse backprop should succeed");
        let telemetry = engine.prior_source_telemetry();
        assert_eq!(
            telemetry.leaf_sparse_count, 0,
            "stage1 leaves must stay dense/default"
        );
        assert!(telemetry.leaf_dense_count + telemetry.leaf_default_count > 0);
    }

    #[test]
    fn mcts_reports_sparse_dense_default_prior_sources() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let legal = vec![Hex::new(50, 50), Hex::new(0, 0), Hex::new(1, 0)];
        let dense = vec![0.0f32; BOARD_AREA];
        let sparse_actions = vec![(50, 50)];
        let sparse_logits = vec![10.0f32];
        let (_, _, _, root_generation) = init_root_parts(&mut engine);

        engine
            .expand_root_with_sparse_priors(
                root_generation,
                &dense,
                0.0,
                -16,
                -16,
                &legal,
                &sparse_actions,
                &sparse_logits,
                2,
                1.0,
            )
            .expect("sparse root priors should apply");

        let telemetry = engine.prior_source_telemetry();
        assert_eq!(telemetry.root_total_count, 3);
        assert_eq!(telemetry.root_sparse_count, 1);
        assert_eq!(telemetry.root_dense_count, 2);
        assert_eq!(telemetry.root_default_count, 0);

        let default_legal = vec![Hex::new(80, 80), Hex::new(0, 0)];
        let mut default_engine =
            MCTSEngine::with_arena_sim_hint(HexGameState::new(), 1, 50, 1.5, 2, false, 0.0, 0);
        let (_, _, _, root_generation) = init_root_parts(&mut default_engine);
        expand_root(
            &mut default_engine,
            &dense,
            -16,
            -16,
            &default_legal,
            root_generation,
        );
        let default_telemetry = default_engine.prior_source_telemetry();
        assert_eq!(default_telemetry.root_default_count, 1);
        assert_eq!(default_telemetry.root_dense_count, 1);
    }

    #[test]
    fn mcts_consumes_pair_policy_on_two_placement_turns() {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening move");
        assert_eq!(game.placements_remaining(), 2);
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 24, 100, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        assert!(legal.len() >= 3);
        let dense = vec![0.0f32; BOARD_AREA];
        expand_root(&mut engine, &dense, oq, or_, &legal, root_generation);

        let pair_a = legal[0];
        let pair_b = legal[1];
        engine
            .apply_root_pair_priors(&[(pair_a.q, pair_a.r, pair_b.q, pair_b.r)], &[10.0], 1.0)
            .expect("pair priors should apply");

        let priors = engine.root_child_priors();
        assert!(priors[0] > 0.0);
        assert!(priors[1] > 0.0);
        assert_eq!(priors[2], 0.0);
        let telemetry = engine.prior_source_telemetry();
        assert_eq!(telemetry.root_pair_candidate_count, 1);
        assert_eq!(telemetry.root_pair_count, 2);

        while !engine.done() {
            let (batch_generation, count) = select_leaves(&mut engine, 4);
            let policies = vec![0.0f32; count as usize * BOARD_AREA];
            let values = vec![0.0f32; count as usize];
            expand_and_backprop(&mut engine, batch_generation, &policies, &values);
        }
        let (moves_q, moves_r, visits, _) = engine.get_results();
        let pair_visits: u32 = moves_q
            .iter()
            .zip(moves_r.iter())
            .zip(visits.iter())
            .filter_map(|((&q, &r), &v)| {
                if (q == pair_a.q && r == pair_a.r) || (q == pair_b.q && r == pair_b.r) {
                    Some(v)
                } else {
                    None
                }
            })
            .sum();
        let non_pair_visits: u32 = visits.iter().sum::<u32>() - pair_visits;
        assert!(
            pair_visits > non_pair_visits,
            "pair-prior actions should dominate visits: pair={pair_visits} non_pair={non_pair_visits}"
        );
        let pair_targets = engine.root_pair_visit_targets();
        assert!(
            !pair_targets.is_empty(),
            "two-placement root search should expose observed joint pair targets"
        );
        assert!(
            pair_targets
                .iter()
                .all(|(q1, r1, q2, r2, visits)| { (*q1, *r1) != (*q2, *r2) && *visits > 0 }),
            "joint pair targets must contain distinct legal moves with positive visits"
        );
    }

    #[test]
    fn mcts_pair_policy_rejects_duplicate_and_illegal_pairs() {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening move");
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let dense = vec![0.0f32; BOARD_AREA];
        expand_root(&mut engine, &dense, oq, or_, &legal, root_generation);
        let a = legal[0];

        let duplicate = engine.apply_root_pair_priors(&[(a.q, a.r, a.q, a.r)], &[1.0], 1.0);
        assert!(duplicate.is_err());

        let illegal = engine.apply_root_pair_priors(&[(a.q, a.r, 999, 999)], &[1.0], 1.0);
        assert!(illegal.is_err());

        let b = legal[1];
        let reversed_duplicate = engine.apply_root_pair_priors(
            &[(a.q, a.r, b.q, b.r), (b.q, b.r, a.q, a.r)],
            &[1.0, 2.0],
            1.0,
        );
        assert!(
            reversed_duplicate.is_err(),
            "unordered pair policy rows must reject reversed duplicates"
        );
    }

    #[test]
    fn mcts_pair_first_policy_reports_candidate_count() {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening move");
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let dense = vec![0.0f32; BOARD_AREA];
        expand_root(&mut engine, &dense, oq, or_, &legal, root_generation);

        let logits = vec![0.0f32; legal.len()];
        engine
            .apply_root_pair_first_priors(&logits, 1.0)
            .expect("pair-first priors should apply");

        let telemetry = engine.prior_source_telemetry();
        assert_eq!(telemetry.root_pair_candidate_count, legal.len() as u32);
        assert_eq!(telemetry.root_pair_count, legal.len() as u32);
    }

    #[test]
    fn mcts_dirichlet_noise_normalizes_child_priors() {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening move");
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let dense = vec![0.0f32; BOARD_AREA];
        expand_root(&mut engine, &dense, oq, or_, &legal, root_generation);

        let mut noise = vec![0.0f32; legal.len()];
        noise[0] = 2.0;
        engine.add_dirichlet_noise(&noise, 1.0).unwrap();

        let priors = engine.root_child_priors();
        let total: f32 = priors.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "priors must sum to one");
        assert_eq!(priors[0], 1.0);
        assert!(priors.iter().skip(1).all(|p| *p == 0.0));
    }

    #[test]
    fn mcts_dirichlet_noise_rejects_non_finite_values_with_error() {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening move");
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let dense = vec![0.0f32; BOARD_AREA];
        expand_root(&mut engine, &dense, oq, or_, &legal, root_generation);

        let mut noise = vec![1.0f32; legal.len()];
        noise[0] = f32::NAN;
        assert!(matches!(
            engine.add_dirichlet_noise(&noise, 0.25),
            Err(MCTSError::InvalidNoise(
                "noise values must be finite and non-negative"
            ))
        ));
    }

    #[test]
    fn mcts_consumes_pair_policy_on_second_placement_root() {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening move");
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let dense = vec![0.0f32; BOARD_AREA];
        expand_root(&mut engine, &dense, oq, or_, &legal, root_generation);

        let first = legal[0];
        engine
            .re_root(first.q, first.r, 1)
            .expect("reroot at first placement");

        let (oq2, or2, second_legal, root_generation) = init_root_parts(&mut engine);
        assert!(second_legal.len() >= 2);
        expand_root(
            &mut engine,
            &dense,
            oq2,
            or2,
            &second_legal,
            root_generation,
        );
        let chosen_second = second_legal[1];
        engine
            .apply_root_pair_second_priors(
                &[(first.q, first.r, chosen_second.q, chosen_second.r)],
                &[8.0],
                1.0,
            )
            .expect("second-placement pair prior should apply");

        let priors = engine.root_child_priors();
        let chosen_idx = second_legal
            .iter()
            .position(|h| h.q == chosen_second.q && h.r == chosen_second.r)
            .unwrap();
        assert_eq!(priors[chosen_idx], 1.0);
        let telemetry = engine.prior_source_telemetry();
        assert_eq!(telemetry.root_pair_candidate_count, 1);
        assert_eq!(telemetry.root_pair_count, 1);
    }

    #[test]
    fn mcts_second_placement_pair_policy_rejects_wrong_first_or_illegal_second() {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening move");
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 1, 50, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let dense = vec![0.0f32; BOARD_AREA];
        expand_root(&mut engine, &dense, oq, or_, &legal, root_generation);
        let first = legal[0];
        engine
            .re_root(first.q, first.r, 1)
            .expect("reroot at first placement");
        let (oq2, or2, second_legal, root_generation) = init_root_parts(&mut engine);
        expand_root(
            &mut engine,
            &dense,
            oq2,
            or2,
            &second_legal,
            root_generation,
        );
        let second = second_legal[0];

        let wrong_first =
            engine.apply_root_pair_second_priors(&[(999, 999, second.q, second.r)], &[1.0], 1.0);
        assert!(wrong_first.is_err());

        let illegal_second =
            engine.apply_root_pair_second_priors(&[(first.q, first.r, 999, 999)], &[1.0], 1.0);
        assert!(illegal_second.is_err());
    }

    #[test]
    fn mcts_pending_leaf_metadata_matches_selected_leaves() {
        let game = HexGameState::new();
        let mut engine = MCTSEngine::with_arena_sim_hint(game, 8, 50, 1.5, 2, false, 0.0, 0);
        let (oq, or_, legal, root_generation) = init_root_parts(&mut engine);
        let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
        expand_root(&mut engine, &uniform, oq, or_, &legal, root_generation);

        let (_batch_generation, count) = select_leaves(&mut engine, 4);
        let meta = engine.pending_leaf_metadata();

        assert_eq!(meta.len(), count as usize);
        assert!(meta
            .iter()
            .all(|(_, _, legal, history)| !legal.is_empty() && !history.is_empty()));
    }
}
