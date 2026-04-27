use criterion::{criterion_group, criterion_main, Criterion};
use hexgame_core::board::HexGameState;
use hexgame_core::core::{hex_distance, Hex, PLACEMENT_RADIUS};
use hexgame_core::encoder;
use std::collections::HashSet;

fn bench_encode_board(c: &mut Criterion) {
    let mut game = HexGameState::new();
    let _ = game.place(0, 0);
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

    c.bench_function("encode_board_into", |b| {
        let mut out = vec![0.0f32; encoder::TENSOR_SIZE];
        let mut hot_buf = Vec::new();
        let mut legal_out = Vec::new();
        b.iter(|| {
            encoder::encode_board_into(&game, 2, false, &mut out, &mut hot_buf, &mut legal_out);
        });
    });

    c.bench_function("encode_board_into_radius8", |b| {
        let mut out = vec![0.0f32; encoder::TENSOR_SIZE];
        let mut hot_buf = Vec::new();
        let mut legal_out = Vec::new();
        b.iter(|| {
            encoder::encode_board_into(
                &game,
                PLACEMENT_RADIUS,
                false,
                &mut out,
                &mut hot_buf,
                &mut legal_out,
            );
        });
    });
}

fn bench_legal_moves(c: &mut Criterion) {
    let mut game = HexGameState::new();
    let _ = game.place(0, 0);
    let placements = [
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (4, 0),
        (3, 1),
        (5, 0),
        (4, 1),
        (3, 2),
        (2, 3),
    ];
    for &(q, r) in &placements {
        let _ = game.place(q, r);
    }

    c.bench_function("legal_moves_near_radius2", |b| {
        b.iter(|| game.legal_moves_near(2));
    });

    c.bench_function("legal_moves_near_radius8", |b| {
        b.iter(|| game.legal_moves_near(PLACEMENT_RADIUS));
    });

    c.bench_function("legal_moves_near_into_radius8", |b| {
        let mut out = Vec::new();
        b.iter(|| game.legal_moves_near_into(PLACEMENT_RADIUS, &mut out));
    });

    c.bench_function("legal_moves_near_radius8_bruteforce", |b| {
        b.iter(|| brute_legal_moves_near(&game, PLACEMENT_RADIUS));
    });

    c.bench_function("candidates_near2", |b| {
        b.iter(|| game.candidates_near2());
    });
}

fn brute_legal_moves_near(game: &HexGameState, radius: i32) -> Vec<Hex> {
    if game.is_over() {
        return Vec::new();
    }
    if game.stones().is_empty() {
        return vec![Hex::ORIGIN];
    }

    let r = radius.clamp(0, PLACEMENT_RADIUS);
    let mut candidates = HashSet::new();
    for &cell in game.stones().keys() {
        for dq in -r..=r {
            for dr in -r..=r {
                let cand = Hex::new(cell.q + dq, cell.r + dr);
                if !game.stones().contains_key(&cand) && hex_distance(cell, cand) <= r {
                    candidates.insert(cand);
                }
            }
        }
    }
    candidates.into_iter().collect()
}

criterion_group!(benches, bench_encode_board, bench_legal_moves);
criterion_main!(benches);
