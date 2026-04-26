use hexgame_core::{Hex, HexGameState, MCTSEngine};
use std::env;
use std::io::{self, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "play" => cmd_play(),
        "bench" => cmd_bench(),
        "perft" => {
            let depth = args.get(2).and_then(|s| s.parse::<u32>().ok()).unwrap_or(1);
            cmd_perft(depth);
        }
        _ => print_help(),
    }
}

fn print_help() {
    eprintln!("hexgame-cli - debug/profiling tool for hexgame-core");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  hexgame-cli play          Run an interactive text-mode game");
    eprintln!("  hexgame-cli bench         Run a quick MCTS benchmark");
    eprintln!("  hexgame-cli perft [D]     Count legal placement nodes to depth D");
}

fn cmd_play() {
    let mut game = HexGameState::new();

    println!("Hexo text mode");
    println!("Enter moves as: q r");
    println!("Player 0 opens at (0, 0); each later turn has two placements.");
    println!("Type 'legal' to print legal moves, 'board' to redraw, or 'quit'.");

    loop {
        render_board(&game);
        if let Some(winner) = game.winner() {
            println!("Player {winner} wins.");
            if let Some(line) = game.winning_line() {
                let cells: Vec<String> = line.iter().map(|h| h.to_string()).collect();
                println!("Winning line: {}", cells.join(" "));
            }
            return;
        }

        print!(
            "P{} placement {}/{} > ",
            game.current_player(),
            turn_placement_number(&game),
            total_turn_placements(&game)
        );
        io::stdout().flush().expect("stdout flush failed");

        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() {
            println!("Input closed.");
            return;
        }

        let line = line.trim();
        match line {
            "" => continue,
            "quit" | "exit" => return,
            "board" => continue,
            "legal" => {
                print_legal_moves(&game, 80);
                continue;
            }
            _ => {}
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            println!("Expected two integers, for example: 0 0");
            continue;
        }
        let q = match parts[0].parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                println!("Invalid q coordinate: {}", parts[0]);
                continue;
            }
        };
        let r = match parts[1].parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                println!("Invalid r coordinate: {}", parts[1]);
                continue;
            }
        };

        match game.place(q, r) {
            Ok(_) => {}
            Err(e) => println!("{e}"),
        }
    }
}

fn turn_placement_number(game: &HexGameState) -> u8 {
    total_turn_placements(game) - game.placements_remaining() + 1
}

fn total_turn_placements(game: &HexGameState) -> u8 {
    if game.move_count() == 0 {
        1
    } else {
        2
    }
}

fn render_board(game: &HexGameState) {
    if game.move_history().is_empty() {
        println!();
        println!("Board is empty. First move must be (0, 0).");
        println!();
        return;
    }

    let mut min_q = 0;
    let mut max_q = 0;
    let mut min_r = 0;
    let mut max_r = 0;
    for mv in game.move_history() {
        let h = mv.cell();
        min_q = min_q.min(h.q);
        max_q = max_q.max(h.q);
        min_r = min_r.min(h.r);
        max_r = max_r.max(h.r);
    }
    min_q -= 1;
    max_q += 1;
    min_r -= 1;
    max_r += 1;

    println!();
    for r in min_r..=max_r {
        let indent = (r - min_r).max(0) as usize;
        print!("{}", " ".repeat(indent));
        for q in min_q..=max_q {
            let marker = stone_at(game, Hex::new(q, r)).unwrap_or('.');
            print!("{marker} ");
        }
        println!("  r={r}");
    }
    println!("q range: {min_q}..{max_q}");
    println!();
}

fn stone_at(game: &HexGameState, cell: Hex) -> Option<char> {
    game.move_history()
        .iter()
        .rev()
        .find(|mv| mv.cell() == cell)
        .map(|mv| match mv.player() {
            0 => 'X',
            1 => 'O',
            _ => '?',
        })
}

fn print_legal_moves(game: &HexGameState, max_to_print: usize) {
    let mut legal = game.legal_moves();
    legal.sort();
    let count = legal.len();
    let shown = count.min(max_to_print);
    for h in legal.iter().take(shown) {
        print!("{h} ");
    }
    if shown < count {
        print!("... ");
    }
    println!();
    println!("{count} legal placements.");
}

fn cmd_bench() {
    use hexgame_core::encoder::BOARD_AREA;

    println!("Running quick MCTS benchmark...");
    let mut game = HexGameState::new();
    game.place(0, 0).unwrap();

    let moves = [
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
    for &(q, r) in &moves {
        let _ = game.place(q, r);
    }

    let start = Instant::now();
    let num_games = 10u32;
    for i in 0..num_games {
        let mut engine = MCTSEngine::new(game.clone(), 50, 1.5, 2, false, i as u64);
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
        "Ran {} MCTS searches in {:.2?} ({:.2?} per search)",
        num_games,
        elapsed,
        elapsed / num_games
    );
}

fn cmd_perft(depth: u32) {
    let mut game = HexGameState::new();
    let start = Instant::now();
    let nodes = perft(&mut game, depth);
    let elapsed = start.elapsed();
    println!("Depth-{depth} placement perft: {nodes} nodes in {elapsed:.2?}");
}

fn perft(game: &mut HexGameState, depth: u32) -> u64 {
    if depth == 0 || game.is_over() {
        return 1;
    }

    let legal = game.legal_moves();
    if depth == 1 {
        return legal.len() as u64;
    }

    let mut nodes = 0u64;
    for h in legal {
        if game.place(h.q, h.r).is_ok() {
            nodes += perft(game, depth - 1);
            game.unplace();
        }
    }
    nodes
}
