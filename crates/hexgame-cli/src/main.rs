use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "play" => cmd_play(),
        "bench" => cmd_bench(),
        "perft" => cmd_perft(),
        _ => {
            eprintln!("hexgame-cli — debug/profiling tool for hexgame-core");
            eprintln!();
            eprintln!("USAGE:");
            eprintln!("  hexgame-cli play     Run an interactive text-mode game");
            eprintln!("  hexgame-cli bench    Run a quick self-play benchmark");
            eprintln!("  hexgame-cli perft    Run perft (performance test) from initial position");
            eprintln!();
            eprintln!("This is a Phase 1 stub. Full functionality coming in Phase 6.");
        }
    }
}

fn cmd_play() {
    println!("interactive play mode — not yet implemented (Phase 6)");
}

fn cmd_bench() {
    use hexgame_core::encoder::BOARD_AREA;
    use hexgame_core::{HexGameState, MCTSEngine};
    use std::time::Instant;

    println!("Running quick self-play benchmark...");
    let mut game = HexGameState::new();
    game.place(0, 0).unwrap();

    let moves = [
        (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
        (3, 0), (2, 1), (1, 2), (0, 3),
        (4, 0), (3, 1), (2, 2),
    ];
    for &(q, r) in &moves {
        let _ = game.place(q, r);
    }

    let start = Instant::now();
    let num_games = 10u32;
    for i in 0..num_games {
        let mut engine = MCTSEngine::new(
            game.clone(),
            50,
            1.5,
            2,
            false,
            i as u64,
        );
        if let Some((_, _, _, legal)) = engine.init_root() {
            let uniform = vec![0.0f32; BOARD_AREA];
            engine.expand_root(&uniform, 0.0, 0, 0, &legal);
            while !engine.done() {
                let count = {
                    let (_, c) = engine.select_leaves(2);
                    c
                };
                if count == 0 {
                    break;
                }
                let p = vec![0.0f32; count as usize * BOARD_AREA];
                let v = vec![0.1f32; count as usize];
                engine.expand_and_backprop(&p, &v);
            }
        }
    }
    let elapsed = start.elapsed();
    println!(
        "Ran {} MCTS games in {:.2?} ({:.2?} per game)",
        num_games,
        elapsed,
        elapsed / num_games,
    );
    println!("Benchmark complete.");
}

fn cmd_perft() {
    use hexgame_core::HexGameState;

    println!("perft from initial position...");
    let game = HexGameState::new();
    let legal = game.legal_moves();
    println!(
        "Initial position: {} legal moves (opening must be at origin)",
        legal.len()
    );
    for h in &legal {
        println!("  ({}, {})", h.q, h.r);
    }
    println!("Depth-1 perft: {} nodes", legal.len());
    println!("perft complete.");
}
