mod board;
use board::{Board, CellState};
mod mcts;
use mcts::Tree;
mod network;
use network::Network;

fn main() {
    let mut board = Board::new();
    let network = Network::new(225, 128, 128, 1, 1, 1);
    let mut player = CellState::X;
    let mut tree = Tree::new(1.414);
    let mut turn = 1;
    println!("{}", board);
    loop {
        println!("\nTurn {}: Player {}", turn, player);
        let best = tree.search(10000, board);
        tree.peek();
        board.set(best.0, best.1, player);
        tree.jump(best.2);
        println!("Player {} at ({}, {})", player, best.0, best.1);
        println!("{}", board);
        if board.check_win(best.0, best.1, player) {
            println!("\nplayer {} win", player);
            break;
        }
        if board.legal().is_empty() {
            println!("\ndraw");
            break;
        }
        player = board.next_player();

        turn += 1;
    }
}
