use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hexgame_core::tactics::threat_status;
use hexgame_core::HexGameState;

fn bench_threat_status(c: &mut Criterion) {
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
        (5, 0),
        (4, 1),
        (3, 2),
        (2, 3),
        (6, 0),
        (5, 1),
        (4, 2),
        (3, 3),
    ];
    for &(q, r) in &placements {
        let _ = game.place(q, r);
    }

    c.bench_function("threat_status", |b| {
        b.iter(|| {
            black_box(threat_status(&game));
        });
    });
}

criterion_group!(benches, bench_threat_status);
criterion_main!(benches);
