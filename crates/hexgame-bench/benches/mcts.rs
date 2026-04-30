use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hexgame_core::mcts::MCTSEngine;
use hexgame_core::rules::HexGameState;

fn bench_single_mcts_sim(c: &mut Criterion) {
    let mut game = HexGameState::new();
    game.place(0, 0).unwrap();
    let placements = [
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
        (4, 0),
        (3, 1),
        (2, 2),
    ];
    for &(q, r) in &placements {
        let _ = game.place(q, r);
    }

    c.bench_function("single_mcts_full_sim", |b| {
        b.iter(|| {
            let mut engine = MCTSEngine::new(
                game.clone(),
                10,    // 10 simulations per bench iteration
                1.5,   // c_puct
                2,     // near_radius
                false, // no threat constraints
                0,     // seed
            );

            if let Some(root) = engine
                .init_root()
                .expect("bench root initialization should succeed")
            {
                let policy = vec![0.0f32; 1089]; // BOARD_AREA uniform logits
                engine
                    .expand_root(
                        root.root_generation,
                        &policy,
                        0.0,
                        root.offset_q,
                        root.offset_r,
                        &root.legal_moves,
                    )
                    .expect("bench root expansion should succeed");

                while !engine.done() {
                    let (batch_generation, count) = {
                        let batch = engine
                            .select_leaves(1)
                            .expect("bench leaf selection should succeed");
                        (batch.batch_generation, batch.non_terminal_count)
                    };
                    let mock_policy = vec![0.0f32; count as usize * 1089]; // BOARD_AREA
                    let mock_values = vec![0.1f32; count as usize];
                    engine
                        .expand_and_backprop(batch_generation, &mock_policy, &mock_values)
                        .expect("bench backpropagation should succeed");
                }
            }
            black_box(engine.done());
        });
    });
}

criterion_group!(benches, bench_single_mcts_sim);
criterion_main!(benches);
