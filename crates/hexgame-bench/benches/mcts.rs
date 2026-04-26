use criterion::{criterion_group, criterion_main, Criterion, black_box};
use hexgame_core::board::HexGameState;
use hexgame_core::MCTSEngine;

fn bench_single_mcts_sim(c: &mut Criterion) {
    let mut game = HexGameState::new();
    game.place(0, 0).unwrap();
    let placements = [
        (1, 0), (0, 1),
        (2, 0), (1, 1), (0, 2),
        (3, 0), (2, 1), (1, 2), (0, 3),
        (4, 0), (3, 1), (2, 2),
    ];
    for &(q, r) in &placements {
        let _ = game.place(q, r);
    }

    c.bench_function("single_mcts_full_sim", |b| {
        b.iter(|| {
            let mut engine = MCTSEngine::new(
                game.clone(),
                10,  // 10 simulations per bench iteration
                1.5,  // c_puct
                2,    // near_radius
                false, // no threat constraints
                0,    // seed
            );

            if let Some((_tensor, oq, or_, legal)) = engine.init_root() {
                let policy = vec![0.0f32; 1089]; // BOARD_AREA uniform logits
                engine.expand_root(&policy, 0.0, oq, or_, &legal);

                while !engine.done() {
                    let (_tensor_slice, count) = engine.select_leaves(1);
                    if count == 0 {
                        engine.expand_and_backprop(&[], &[]);
                    } else {
                        let mock_policy = vec![0.0f32; count as usize * 1089]; // BOARD_AREA
                        let mock_values = vec![0.1f32; count as usize];
                        engine.expand_and_backprop(&mock_policy, &mock_values);
                    }
                }
            }
            black_box(engine.done());
        });
    });
}

criterion_group!(benches, bench_single_mcts_sim);
criterion_main!(benches);
