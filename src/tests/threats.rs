use crate::board::HexGameState;
use crate::core::Turn;
use crate::tests::oracle::{analyse, TurnAnalysis};
use crate::threats::{live_cells, threat_status, turn_satisfies_status, turn_satisfies_threats, ThreatStatus};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

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

fn assert_matches(fast: &ThreatStatus, oracle: &TurnAnalysis, game: &HexGameState) {
    let remaining = game.placements_remaining();
    let opp = 1 - game.current_player();
    let opp_counts = game.eval().counts(opp);
    let opp_has_threats = opp_counts.fours > 0 || opp_counts.fives > 0;

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
                                b.cells.contains(&cell),
                                "fast missed blocking cell {:?}",
                                cell
                            );
                        }
                        for &cell in &b.cells {
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
                        for &(a, b_cell) in &b.pairs {
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
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 10, ..ProptestConfig::default() })]

    /// Play a compact deterministic random game and verify threat_status against
    /// the brute-force oracle after every completed turn.
    #[test]
    #[ignore = "slow oracle: run with cargo test --release -- --ignored"]
    fn threat_status_matches_oracle_random_positions(seed in any::<u64>()) {
        let mut rng = Prng::new(seed);
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        const MAX_MOVES: usize = 6;

        while moves_played < MAX_MOVES && !game.is_over() {
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

    /// For each position reached, verify that every legal turn from the oracle
    /// either satisfies or doesn't satisfy turn_satisfies_threats consistently.
    #[test]
    #[ignore = "slow oracle: run with cargo test --release -- --ignored"]
    fn turn_satisfies_threats_matches_oracle(seed in any::<u64>()) {
        let mut rng = Prng::new(seed);
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        const MAX_MOVES: usize = 6;

        while moves_played < MAX_MOVES && !game.is_over() {
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
                    let opp_has_threats = opp_counts.fours > 0 || opp_counts.fives > 0;

                    // Build the set of turns that the oracle says are special.
                    // When the opponent has no real threats the oracle flags every
                    // legal move as "blocking", which is not meaningful for the
                    // fast path. In that case we only check winning turns.
                    let mut must_play = std::collections::HashSet::new();
                    for turn in &oracle.winning {
                        must_play.insert(*turn);
                    }
                    if opp_has_threats {
                        for turn in &oracle.blocking_pairs {
                            must_play.insert(*turn);
                        }
                        if game.placements_remaining() == 1 {
                            for &cell in &oracle.blocking_single {
                                must_play.insert(Turn::single(cell));
                            }
                        }
                    }

                    for turn in &oracle.legal {
                        let satisfies = turn_satisfies_threats(&game, *turn);
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
                                "turn {:?} is must-play but turn_satisfies_threats returned false",
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

    /// Verify that live_cells contains all cells from oracle winning turns and
    /// blocking singles.
    #[test]
    #[ignore = "slow oracle: run with cargo test --release -- --ignored"]
    fn live_cells_contains_all_threat_cells(seed in any::<u64>()) {
        let mut rng = Prng::new(seed);
        let mut game = HexGameState::new();
        let mut moves_played = 0;
        const MAX_MOVES: usize = 6;

        while moves_played < MAX_MOVES && !game.is_over() {
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

                    // Every cell in an oracle-winning turn must be live for the
                    // current player.
                    for turn in &oracle.winning {
                        let first = turn.first();
                        assert!(
                            live_current.contains(&first),
                            "winning cell {:?} not in live_current",
                            first
                        );
                        if let Some(second) = turn.second() {
                            assert!(
                                live_current.contains(&second),
                                "winning cell {:?} not in live_current",
                                second
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
