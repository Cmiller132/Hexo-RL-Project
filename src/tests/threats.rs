//! Property-based threat-analysis tests.
//!
//! This module uses [`proptest`](https://docs.rs/proptest) to generate
//! hundreds of random board positions and compares the engine's fast
//! incremental threat path against the brute-force oracle.
//!
//! ## What is tested
//!
//! 1. **`threat_status_matches_oracle`** — For every random position, the
//!    [`ThreatStatus`](crate::threats::ThreatStatus) returned by the fast path
//!    must agree with the oracle on winning turns, blocking singles/pairs, and
//!    unblockable positions.
//! 2. **`turn_satisfies_threats_matches_oracle`** — Every legal turn must be
//!    classified consistently by `turn_satisfies_threats` relative to the
//!    oracle's must-play set.
//! 3. **`live_cells_contains_all_threat_cells`** — Every single winning cell
//!    and at least one cell of every winning pair must be returned by
//!    [`live_cells`](crate::threats::live_cells).  (When one cell is a single
//!    win, the oracle also reports every legal pair containing it; the
//!    arbitrary partner cell is not required to be live.)
//!
//! ## Random game generation
//!
//! Each test receives a `u64` seed from proptest. A deterministic LCG
//! uses that seed to play a random game of 1–40 moves, choosing legal
//! cells from `candidates_near2()`. After every *completed* turn (when
//! `placements_remaining` resets), the position is snapshotted and checked.
//! This yields hundreds of distinct board states per test run.

use crate::board::HexGameState;
use crate::core::Turn;
use crate::tests::oracle::{analyse, TurnAnalysis};
use crate::threats::{live_cells, threat_status, turn_satisfies_status, ThreatStatus};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

/// Simple 64-bit LCG used for reproducible random games inside proptest.
///
/// The multiplier and increment are the Numerical Recipes parameters,
/// which give a full-period generator. Using a hand-rolled PRNG avoids
/// depending on `rand` in dev-dependencies just for tests.
struct Prng(u64);

impl Prng {
    const MULTIPLIER: u64 = 6364136223846793005;
    const INCREMENT: u64 = 1442695040888963407;

    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(Self::MULTIPLIER).wrapping_add(Self::INCREMENT);
        self.0
    }

    /// Uniform value in `[0, max)`.
    fn range(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next() % max as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Comparison helper: fast path vs oracle
// ---------------------------------------------------------------------------

/// Assert that a [`ThreatStatus`](crate::threats::ThreatStatus) from the fast
/// path matches a [`TurnAnalysis`] from the brute-force oracle.
///
/// # Bidirectional checks
///
/// The oracle is treated as ground truth. For every category the helper
/// checks **both** directions:
///
/// - **Winning turns:** if the fast path says the position is a
///   `WinningTurn`, that exact turn must be in the oracle's winning set;
///   conversely, if the oracle finds any winning turn, the fast path must
///   report `WinningTurn`.
/// - **Blocking singles** (one placement remaining): the fast path's
///   `MustBlock.cells` set must equal the oracle's `blocking_single` set.
/// - **Blocking pairs** (two placements remaining): every oracle blocking
///   pair must satisfy the fast-path constraint, and every fast-path pair
///   must appear in the oracle.
fn assert_matches(fast: &ThreatStatus, oracle: &TurnAnalysis, game: &HexGameState) {
    let remaining = game.placements_remaining();
    let opp = 1 - game.current_player();
    let opp_counts = game.eval().counts(opp);
    let opp_has_threats = opp_counts.fours() > 0 || opp_counts.fives() > 0;

    // 1. Winning turn handling — bidirectional
    match fast {
        ThreatStatus::WinningTurn(t) => {
            assert!(
                oracle.winning.contains(t),
                "fast said winning {:?} but oracle disagrees",
                t
            );
            // Blocking moves are irrelevant when the current player can win.
            return;
        }
        _ => {
            assert!(
                oracle.winning.is_empty(),
                "fast missed winning turn(s): {:?}",
                oracle.winning
            );
        }
    }

    // 2. Blocking / threat handling when no winning turn for current player.
    // The oracle populates blocking_* for every move that prevents the opponent
    // from winning on their next turn. When the opponent has no immediate
    // threats (no fours or fives) this includes ALL legal moves, which is not
    // what the fast path means by MustBlock. Only compare blocking when the
    // opponent actually has hot windows.
    if opp_has_threats {
        let has_blocking = !oracle.blocking_single.is_empty() || !oracle.blocking_pairs.is_empty();

        if has_blocking {
            match fast {
                ThreatStatus::MustBlock(b) => {
                    if remaining == 1 {
                        // Bidirectional: oracle blocking singles == fast blocking cells
                        for &cell in &oracle.blocking_single {
                            assert!(
                                b.cells().contains(&cell),
                                "fast missed blocking cell {:?}",
                                cell
                            );
                        }
                        for &cell in b.cells() {
                            assert!(
                                oracle.blocking_single.contains(&cell),
                                "fast has extra blocking cell {:?}",
                                cell
                            );
                        }
                    } else {
                        // Oracle blocking pairs must be accepted by fast path.
                        for turn in &oracle.blocking_pairs {
                            assert!(
                                turn_satisfies_status(fast, *turn),
                                "oracle blocking pair {:?} not accepted by fast path",
                                turn
                            );
                        }
                        // Fast path's explicit pairs must be in oracle.
                        for &(a, b_cell) in b.pairs() {
                            let turn = Turn::pair(a, b_cell);
                            assert!(
                                oracle.blocking_pairs.contains(&turn),
                                "fast has extra blocking pair {:?}",
                                turn
                            );
                        }
                    }
                }
                ThreatStatus::Unblockable => {
                    if remaining == 1 {
                        assert!(
                            oracle.blocking_single.is_empty(),
                            "fast said unblockable but oracle found blocking singles"
                        );
                    } else {
                        assert!(
                            oracle.blocking_pairs.is_empty(),
                            "fast said unblockable but oracle found blocking pairs"
                        );
                    }
                }
                ThreatStatus::Quiet => {
                    panic!(
                        "expected MustBlock or Unblockable when opponent has threats, got Quiet"
                    );
                }
                ThreatStatus::WinningTurn(_) => unreachable!(), // handled above
            }
        } else {
            // Opponent has threats but no blocking moves exist → Unblockable.
            match fast {
                ThreatStatus::Unblockable => {}
                _ => panic!(
                    "expected Unblockable when opponent has threats but no blocking moves exist, got {:?}",
                    fast
                ),
            }
        }
    } else {
        // No opponent threats. Fast path should be Quiet.
        match fast {
            ThreatStatus::Quiet => {}
            _ => panic!("expected Quiet when opponent has no threats, got {:?}", fast),
        }
    }
}

// ---------------------------------------------------------------------------
// Property tests — first 500 cases
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 500, ..ProptestConfig::default() })]

    /// Play a compact deterministic random game and verify `threat_status`
    /// against the brute-force oracle after every completed turn.
    ///
    /// A random game of 1–40 moves is generated from the proptest seed.
    /// After each full turn (when the player switches), the fast path and
    /// the oracle are run and compared with [`assert_matches`].
    #[test]
    #[ignore = "slow oracle: run with cargo test --release -- --ignored"]
    fn threat_status_matches_oracle_random_positions(seed in any::<u64>()) {
        let mut rng = Prng::new(seed);
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        // Random game length: 1 to 40 completed turns.
        let max_moves = 1 + rng.range(40);

        while moves_played < max_moves && !game.is_over() {
            let legal = game.candidates_near2();
            if legal.is_empty() {
                break;
            }
            let idx = rng.range(legal.len());
            let cell = legal[idx];
            let turn_ended = game.place(cell.q, cell.r).unwrap();
            if turn_ended {
                moves_played += 1;
                if !game.is_over() {
                    let fast = threat_status(&game);
                    let oracle = analyse(&mut game.clone());
                    assert_matches(&fast, &oracle, &game);
                }
            }
        }
    }

    /// For each random position, verify that every legal turn is classified
    /// consistently by `turn_satisfies_threats` relative to the oracle.
    ///
    /// The oracle computes a `must_play` set: winning turns plus blocking
    /// moves (when the opponent actually has threats). Every legal turn must
    /// satisfy `turn_satisfies_threats` exactly when it is in `must_play`,
    /// except that the fast path may expose only the first winning turn when
    /// several exist.
    #[test]
    #[ignore = "slow oracle: run with cargo test --release -- --ignored"]
    fn turn_satisfies_threats_matches_oracle(seed in any::<u64>()) {
        let mut rng = Prng::new(seed);
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        let max_moves = 1 + rng.range(40);

        while moves_played < max_moves && !game.is_over() {
            let legal = game.candidates_near2();
            if legal.is_empty() {
                break;
            }
            let idx = rng.range(legal.len());
            let cell = legal[idx];
            let turn_ended = game.place(cell.q, cell.r).unwrap();
            if turn_ended {
                moves_played += 1;
                if !game.is_over() {
                    let fast = threat_status(&game);
                    let oracle = analyse(&mut game.clone());

                    let opp = 1 - game.current_player();
                    let opp_counts = game.eval().counts(opp);
                    let opp_has_threats = opp_counts.fours() > 0 || opp_counts.fives() > 0;

                    // Build the set of turns that the oracle says are special.
                    // When the opponent has no real threats the oracle flags every
                    // legal move as "blocking", which is not meaningful for the
                    // fast path. In that case we only check winning turns.
                    let mut must_play = std::collections::HashSet::new();
                    for turn in &oracle.winning {
                        must_play.insert(*turn);
                    }
                    // Blocking moves are irrelevant when a winning turn exists;
                    // the current player should always take the win.
                    if oracle.winning.is_empty() && opp_has_threats {
                        for turn in &oracle.blocking_pairs {
                            must_play.insert(*turn);
                        }
                        if game.placements_remaining() == 1 {
                            for &cell in &oracle.blocking_single {
                                must_play.insert(Turn::single(cell));
                            }
                        }
                    }

                    let status = threat_status(&game);
                    for turn in &oracle.legal {
                        let satisfies = turn_satisfies_status(&status, *turn);
                        let is_must_play = must_play.contains(turn);

                        if is_must_play {
                            // Fast path may only expose the first winning turn when
                            // several exist. Allow other winning turns through.
                            if let ThreatStatus::WinningTurn(w) = &fast {
                                if oracle.winning.contains(turn) && turn != w {
                                    continue;
                                }
                            }
                            assert!(
                                satisfies,
                                "turn {:?} is must-play but turn_satisfies_status returned false",
                                turn
                            );
                        } else {
                            match &fast {
                                ThreatStatus::Quiet | ThreatStatus::Unblockable => {
                                    assert!(
                                        satisfies,
                                        "turn {:?} should satisfy when no constraint",
                                        turn
                                    );
                                }
                                ThreatStatus::WinningTurn(_) | ThreatStatus::MustBlock(_) => {
                                    assert!(
                                        !satisfies,
                                        "turn {:?} should not satisfy under constraint {:?}",
                                        turn,
                                        fast
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Verify that `live_cells` contains every cell that participates in an
    /// oracle-winning turn or blocking single.
    ///
    /// `live_cells` is used by search and encoding to restrict attention to
    /// cells that can actually matter. If it ever dropped a winning or
    /// blocking cell, the engine would miss critical moves.
    #[test]
    #[ignore = "slow oracle: run with cargo test --release -- --ignored"]
    fn live_cells_contains_all_threat_cells(seed in any::<u64>()) {
        let mut rng = Prng::new(seed);
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        let max_moves = 1 + rng.range(40);

        while moves_played < max_moves && !game.is_over() {
            let legal = game.candidates_near2();
            if legal.is_empty() {
                break;
            }
            let idx = rng.range(legal.len());
            let cell = legal[idx];
            let turn_ended = game.place(cell.q, cell.r).unwrap();
            if turn_ended {
                moves_played += 1;
                if !game.is_over() {
                    let oracle = analyse(&mut game.clone());
                    let me = game.current_player();
                    let opp = 1 - me;

                    let mut live_current = Vec::new();
                    live_cells(&game, me, &mut live_current);

                    let mut live_opp = Vec::new();
                    live_cells(&game, opp, &mut live_opp);

                    // Every oracle-winning turn must have at least one cell in
                    // a hot window for the current player.  (If one cell is a
                    // single win, any legal second cell is also reported as a
                    // winning pair by the oracle; only the single-winning cell
                    // is required to be live.)
                    for turn in &oracle.winning {
                        if let Some(second) = turn.second() {
                            assert!(
                                live_current.contains(&turn.first())
                                    || live_current.contains(&second),
                                "neither cell of winning turn {:?} is in live_current",
                                turn
                            );
                        } else {
                            assert!(
                                live_current.contains(&turn.first()),
                                "single winning cell {:?} not in live_current",
                                turn.first()
                            );
                        }
                    }

                    // Every oracle blocking single must be live for the opponent.
                    for &cell in &oracle.blocking_single {
                        assert!(
                            live_opp.contains(&cell),
                            "blocking cell {:?} not in live_opp",
                            cell
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Property tests — second 500 cases with different seed offset
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 500, ..ProptestConfig::default() })]

    #[test]
    #[ignore = "slow oracle: run with cargo test --release -- --ignored"]
    fn threat_status_matches_oracle_random_positions_b(seed in any::<u64>()) {
        let mut rng = Prng::new(seed.wrapping_add(0xFEDC_BA98_7654_3210));
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        let max_moves = 1 + rng.range(40);

        while moves_played < max_moves && !game.is_over() {
            let legal = game.candidates_near2();
            if legal.is_empty() {
                break;
            }
            let idx = rng.range(legal.len());
            let cell = legal[idx];
            let turn_ended = game.place(cell.q, cell.r).unwrap();
            if turn_ended {
                moves_played += 1;
                if !game.is_over() {
                    let fast = threat_status(&game);
                    let oracle = analyse(&mut game.clone());
                    assert_matches(&fast, &oracle, &game);
                }
            }
        }
    }

    #[test]
    #[ignore = "slow oracle: run with cargo test --release -- --ignored"]
    fn turn_satisfies_threats_matches_oracle_b(seed in any::<u64>()) {
        let mut rng = Prng::new(seed.wrapping_add(0xFEDC_BA98_7654_3210));
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        let max_moves = 1 + rng.range(40);

        while moves_played < max_moves && !game.is_over() {
            let legal = game.candidates_near2();
            if legal.is_empty() {
                break;
            }
            let idx = rng.range(legal.len());
            let cell = legal[idx];
            let turn_ended = game.place(cell.q, cell.r).unwrap();
            if turn_ended {
                moves_played += 1;
                if !game.is_over() {
                    let fast = threat_status(&game);
                    let oracle = analyse(&mut game.clone());

                    let opp = 1 - game.current_player();
                    let opp_counts = game.eval().counts(opp);
                    let opp_has_threats = opp_counts.fours() > 0 || opp_counts.fives() > 0;

                    let mut must_play = std::collections::HashSet::new();
                    for turn in &oracle.winning {
                        must_play.insert(*turn);
                    }
                    if oracle.winning.is_empty() && opp_has_threats {
                        for turn in &oracle.blocking_pairs {
                            must_play.insert(*turn);
                        }
                        if game.placements_remaining() == 1 {
                            for &cell in &oracle.blocking_single {
                                must_play.insert(Turn::single(cell));
                            }
                        }
                    }

                    let status = threat_status(&game);
                    for turn in &oracle.legal {
                        let satisfies = turn_satisfies_status(&status, *turn);
                        let is_must_play = must_play.contains(turn);

                        if is_must_play {
                            if let ThreatStatus::WinningTurn(w) = &fast {
                                if oracle.winning.contains(turn) && turn != w {
                                    continue;
                                }
                            }
                            assert!(
                                satisfies,
                                "turn {:?} is must-play but turn_satisfies_status returned false",
                                turn
                            );
                        } else {
                            match &fast {
                                ThreatStatus::Quiet | ThreatStatus::Unblockable => {
                                    assert!(
                                        satisfies,
                                        "turn {:?} should satisfy when no constraint",
                                        turn
                                    );
                                }
                                ThreatStatus::WinningTurn(_) | ThreatStatus::MustBlock(_) => {
                                    assert!(
                                        !satisfies,
                                        "turn {:?} should not satisfy under constraint {:?}",
                                        turn,
                                        fast
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "slow oracle: run with cargo test --release -- --ignored"]
    fn live_cells_contains_all_threat_cells_b(seed in any::<u64>()) {
        let mut rng = Prng::new(seed.wrapping_add(0xFEDC_BA98_7654_3210));
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        let max_moves = 1 + rng.range(40);

        while moves_played < max_moves && !game.is_over() {
            let legal = game.candidates_near2();
            if legal.is_empty() {
                break;
            }
            let idx = rng.range(legal.len());
            let cell = legal[idx];
            let turn_ended = game.place(cell.q, cell.r).unwrap();
            if turn_ended {
                moves_played += 1;
                if !game.is_over() {
                    let oracle = analyse(&mut game.clone());
                    let me = game.current_player();
                    let opp = 1 - me;

                    let mut live_current = Vec::new();
                    live_cells(&game, me, &mut live_current);

                    let mut live_opp = Vec::new();
                    live_cells(&game, opp, &mut live_opp);

                    for turn in &oracle.winning {
                        if let Some(second) = turn.second() {
                            assert!(
                                live_current.contains(&turn.first())
                                    || live_current.contains(&second),
                                "neither cell of winning turn {:?} is in live_current",
                                turn
                            );
                        } else {
                            assert!(
                                live_current.contains(&turn.first()),
                                "single winning cell {:?} not in live_current",
                                turn.first()
                            );
                        }
                    }

                    for &cell in &oracle.blocking_single {
                        assert!(
                            live_opp.contains(&cell),
                            "blocking cell {:?} not in live_opp",
                            cell
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fast smoke tests — NOT ignored, run in CI
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 10, ..ProptestConfig::default() })]

    #[test]
    fn threat_status_matches_oracle_smoke(seed in any::<u64>()) {
        let mut rng = Prng::new(seed);
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        // Short games for CI speed: 1–5 completed turns.
        let max_moves = 1 + rng.range(5);

        while moves_played < max_moves && !game.is_over() {
            let legal = game.candidates_near2();
            if legal.is_empty() {
                break;
            }
            let idx = rng.range(legal.len());
            let cell = legal[idx];
            let turn_ended = game.place(cell.q, cell.r).unwrap();
            if turn_ended {
                moves_played += 1;
                if !game.is_over() {
                    let fast = threat_status(&game);
                    let oracle = analyse(&mut game.clone());
                    assert_matches(&fast, &oracle, &game);
                }
            }
        }
    }

    #[test]
    fn turn_satisfies_threats_matches_oracle_smoke(seed in any::<u64>()) {
        let mut rng = Prng::new(seed);
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        let max_moves = 1 + rng.range(5);

        while moves_played < max_moves && !game.is_over() {
            let legal = game.candidates_near2();
            if legal.is_empty() {
                break;
            }
            let idx = rng.range(legal.len());
            let cell = legal[idx];
            let turn_ended = game.place(cell.q, cell.r).unwrap();
            if turn_ended {
                moves_played += 1;
                if !game.is_over() {
                    let fast = threat_status(&game);
                    let oracle = analyse(&mut game.clone());

                    let opp = 1 - game.current_player();
                    let opp_counts = game.eval().counts(opp);
                    let opp_has_threats = opp_counts.fours() > 0 || opp_counts.fives() > 0;

                    let mut must_play = std::collections::HashSet::new();
                    for turn in &oracle.winning {
                        must_play.insert(*turn);
                    }
                    if oracle.winning.is_empty() && opp_has_threats {
                        for turn in &oracle.blocking_pairs {
                            must_play.insert(*turn);
                        }
                        if game.placements_remaining() == 1 {
                            for &cell in &oracle.blocking_single {
                                must_play.insert(Turn::single(cell));
                            }
                        }
                    }

                    let status = threat_status(&game);
                    for turn in &oracle.legal {
                        let satisfies = turn_satisfies_status(&status, *turn);
                        let is_must_play = must_play.contains(turn);

                        if is_must_play {
                            if let ThreatStatus::WinningTurn(w) = &fast {
                                if oracle.winning.contains(turn) && turn != w {
                                    continue;
                                }
                            }
                            assert!(
                                satisfies,
                                "turn {:?} is must-play but turn_satisfies_status returned false",
                                turn
                            );
                        } else {
                            match &fast {
                                ThreatStatus::Quiet | ThreatStatus::Unblockable => {
                                    assert!(
                                        satisfies,
                                        "turn {:?} should satisfy when no constraint",
                                        turn
                                    );
                                }
                                ThreatStatus::WinningTurn(_) | ThreatStatus::MustBlock(_) => {
                                    assert!(
                                        !satisfies,
                                        "turn {:?} should not satisfy under constraint {:?}",
                                        turn,
                                        fast
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn live_cells_contains_all_threat_cells_smoke(seed in any::<u64>()) {
        let mut rng = Prng::new(seed);
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        let max_moves = 1 + rng.range(5);

        while moves_played < max_moves && !game.is_over() {
            let legal = game.candidates_near2();
            if legal.is_empty() {
                break;
            }
            let idx = rng.range(legal.len());
            let cell = legal[idx];
            let turn_ended = game.place(cell.q, cell.r).unwrap();
            if turn_ended {
                moves_played += 1;
                if !game.is_over() {
                    let oracle = analyse(&mut game.clone());
                    let me = game.current_player();
                    let opp = 1 - me;

                    let mut live_current = Vec::new();
                    live_cells(&game, me, &mut live_current);

                    let mut live_opp = Vec::new();
                    live_cells(&game, opp, &mut live_opp);

                    for turn in &oracle.winning {
                        if let Some(second) = turn.second() {
                            assert!(
                                live_current.contains(&turn.first())
                                    || live_current.contains(&second),
                                "neither cell of winning turn {:?} is in live_current",
                                turn
                            );
                        } else {
                            assert!(
                                live_current.contains(&turn.first()),
                                "single winning cell {:?} not in live_current",
                                turn.first()
                            );
                        }
                    }

                    for &cell in &oracle.blocking_single {
                        assert!(
                            live_opp.contains(&cell),
                            "blocking cell {:?} not in live_opp",
                            cell
                        );
                    }
                }
            }
        }
    }
}
