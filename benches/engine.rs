use criterion::{criterion_group, criterion_main, Criterion};
use hexgame::board::HexGameState;
use hexgame::encoder;

fn bench_encode_board(c: &mut Criterion) {
    let mut game = HexGameState::new();
    let _ = game.place(0, 0);
    let placements = [
        (1, 0), (0, 1),
        (2, 0), (1, 1), (0, 2),
        (3, 0), (2, 1), (1, 2), (0, 3),
        (4, 0), (3, 1), (2, 2),
    ];
    for &(q, r) in &placements {
        let _ = game.place(q, r);
    }

    c.bench_function("encode_board_into", |b| {
        let mut out = vec![0.0f32; encoder::TENSOR_SIZE];
        let mut hot_buf = Vec::new();
        let mut legal_out = Vec::new();
        b.iter(|| {
            encoder::encode_board_into(&game, 2, false, &mut out, &mut hot_buf, &mut legal_out);
        });
    });
}

fn bench_legal_moves(c: &mut Criterion) {
    let mut game = HexGameState::new();
    let _ = game.place(0, 0);
    let placements = [
        (1, 0), (0, 1),
        (2, 0), (1, 1), (0, 2),
        (3, 0), (2, 1), (1, 2),
        (4, 0), (3, 1),
        (5, 0), (4, 1), (3, 2), (2, 3),
    ];
    for &(q, r) in &placements {
        let _ = game.place(q, r);
    }

    c.bench_function("legal_moves_near_radius2", |b| {
        b.iter(|| {
            game.legal_moves_near(2)
        });
    });

    c.bench_function("candidates_near2", |b| {
        b.iter(|| {
            game.candidates_near2()
        });
    });
}

criterion_group!(benches, bench_encode_board, bench_legal_moves);
criterion_main!(benches);
