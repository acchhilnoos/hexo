mod board;
use board::{Board, CellState};
mod mcts;
use mcts::Tree;

fn main() {
    let mut board = Board::new();
    let mut current_player = CellState::X;
    let mut turn = 1;
    println!("{}", board);
    loop {
        println!("\nTurn {}: Player {}", turn, current_player);
        let mut mcts = Tree::new(1.414);
        let best = mcts.search(500000, board);
        mcts.peek();
        board.set(best.0, best.1, current_player);
        println!("Player {} at ({}, {})", current_player, best.0, best.1);
        println!("{}", board);
        if board.check_win(best.0, best.1, current_player) {
            println!("\nplayer {} win", current_player);
            break;
        }
        if board.legal().is_empty() {
            println!("\ndraw");
            break;
        }
        current_player = board.next_player();
        
        turn += 1;
    }
}
