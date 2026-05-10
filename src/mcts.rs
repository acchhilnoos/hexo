use std::collections::VecDeque;

use crate::board::{Board, CellState, BOARD_SIZE};
use crate::network::{Matrix, Network};
use rand::Rng;

#[derive(Clone, Debug, Eq, PartialEq)]
enum GameState {
    Win,
    Loss,
    Draw,
    Incomplete,
}

#[derive(Clone, Debug)]
struct Node {
    // parent of root is None
    parent: Option<usize>,
    // indices of child nodes in enclosing tree
    children: Vec<usize>,

    // number of evaluations in subtree rooted at self
    n: usize,
    // sum of evaluations in subtree rooted at self
    v: f32,
    // probability self.a is chosen at parent
    p: f32,

    // a of root is None
    a: Option<(i32, i32)>,
    s: GameState,
}

pub struct Tree {
    nodes: Vec<Node>,
    // exploration factor
    c: f32,
}

impl Node {
    fn ucb1(&self, c: f32, parent_n: usize) -> f32 {
        if self.n == 0 {
            return f32::INFINITY;
        }

        let q = self.v / self.n as f32;
        let u = c * self.p * (parent_n as f32).sqrt() / (1.0 + self.n as f32);

        q + u
    }
}

impl Tree {
    pub fn new(c: f32) -> Self {
        Self {
            nodes: vec![Node {
                parent: None,
                children: Vec::new(),
                n: 0,
                v: 0.0,
                p: 0.0,
                a: None,
                s: GameState::Incomplete,
            }],
            c,
        }
    }

    // find the index and corresponding state of a leaf node
    fn select(&self, root_idx: usize, mut board: Board) -> (usize, Board) {
        let mut parent_idx = root_idx;

        while !self.nodes[parent_idx].children.is_empty() {
            let parent = &self.nodes[parent_idx];
            let player = board.next_player();

            // NOTE:
            // let max_child_idx = parent
            //     .children
            //     .iter()
            //     .max_by_key(|&&c| self.nodes[c].ucb1(self.c, parent.n));
            // doesn't work because comparison isn't defined on f32::NAN

            let mut max_child_v = f32::NEG_INFINITY;
            let mut max_child_idx = 0;

            for &child_idx in &parent.children {
                let child_v = self.nodes[child_idx].ucb1(self.c, parent.n);
                if child_v > max_child_v {
                    max_child_v = child_v;
                    max_child_idx = child_idx;
                }
            }

            parent_idx = max_child_idx;
            let a = self.nodes[parent_idx]
                .a
                .expect("Tree::select() state found without action");
            board.set(a.0, a.1, player);
        }

        (parent_idx, board)
    }

    // create child nodes of self.nodes[leaf_idx] if not terminal else assign state
    fn expand(&mut self, leaf_idx: usize, leaf_board: Board) {
        if let Some((x, y)) = self.nodes[leaf_idx].a {
            let parent_player = {
                let mut temp = leaf_board;
                temp.n_pieces -= 1;
                temp.next_player()
            };

            if leaf_board.check_win(x, y, parent_player) {
                self.nodes[leaf_idx].s = if parent_player == leaf_board.next_player() {
                    GameState::Win
                } else {
                    GameState::Loss
                }
            }
        } else {
            let actions = leaf_board.legal();

            if actions.is_empty() {
                self.nodes[leaf_idx].s = GameState::Draw;
            } else {
                let p = 1.0 / actions.len() as f32;

                for action in actions {
                    let child_idx = self.nodes.len();
                    self.nodes.push(Node {
                        parent: Some(leaf_idx),
                        children: Vec::new(),
                        n: 0,
                        v: 0.0,
                        p,
                        a: Some(action),
                        s: GameState::Incomplete,
                    });
                    self.nodes[leaf_idx].children.push(child_idx);
                }
            }
        }
    }

    // determine final game state after random rollout from leaf_board
    fn simulate(&self, mut leaf_board: Board) -> GameState {
        let leaf_player = leaf_board.next_player();
        let mut rng = rand::thread_rng();

        let mut actions = leaf_board.legal();
        if actions.is_empty() {
            return GameState::Draw;
        }

        for i in (1..actions.len()).rev() {
            let j = rng.gen_range(0..=i);
            actions.swap(i, j);
        }

        for a in actions {
            let player = leaf_board.next_player();
            leaf_board.set(a.0, a.1, player);

            if leaf_board.check_win(a.0, a.1, player) {
                return if player == leaf_player {
                    GameState::Win
                } else {
                    GameState::Loss
                };
            }
        }

        GameState::Draw
    }

    // update ancestors of self.nodes[child_idx]
    fn backpropagate(&mut self, mut child_idx: usize, mut board: Board, mut v: f32) {
        loop {
            let node = &mut self.nodes[child_idx];
            node.n += 1;
            node.v += v;

            if let Some(parent_idx) = node.parent {
                let player = board.next_player();
                board.n_pieces -= 1;
                let parent_player = board.next_player();

                child_idx = parent_idx;
                if parent_player != player {
                    v = -v;
                }
            } else {
                break;
            }
        }
    }

    pub fn search(&mut self, iterations: usize, root_board: Board) -> (i32, i32, usize) {
        for _ in 0..iterations {
            let (leaf_idx, leaf_board) = self.select(0, root_board);
            self.expand(leaf_idx, leaf_board);

            let v = match self.nodes[leaf_idx].s {
                GameState::Win => 1.0,
                GameState::Loss => -1.0,
                GameState::Draw => 0.0,
                GameState::Incomplete => match self.simulate(leaf_board) {
                    GameState::Win => 1.0,
                    GameState::Loss => -1.0,
                    GameState::Draw => 0.0,
                    _ => unreachable!(),
                },
            };

            self.backpropagate(leaf_idx, leaf_board, v);
        }

        let &max_child_idx = self.nodes[0]
            .children
            .iter()
            .max_by_key(|&&child_idx| self.nodes[child_idx].n)
            .expect("Tree::search() root is terminal");

        let max_a = &self.nodes[max_child_idx]
            .a
            .expect("Tree::search() state found without action");

        (max_a.0, max_a.1, max_child_idx)
    }

    pub fn jump(&mut self, idx: usize) {
        let mut subtree = vec![self.nodes[idx].clone()];
        let mut q: VecDeque<usize> = VecDeque::new();
        for &i in self.nodes[idx].children.iter() {
            q.push_back(i);
        }
        loop {
            if let Some(i) = q.pop_front() {
                subtree.push(self.nodes[i].clone());
                for &j in &self.nodes[i].children {
                    q.push_back(j);
                }
            } else {
                break;
            }
        }
        self.nodes = subtree;
    }

    pub fn peek(&self) {
        let root = &self.nodes[0];

        let mut children: Vec<&Node> = root.children.iter().map(|&idx| &self.nodes[idx]).collect();
        children.sort_by(|a, b| b.n.cmp(&a.n));

        for child in children.iter().take(5) {
            if let Some((x, y)) = child.a {
                let q = if child.n > 0 {
                    child.v / child.n as f32
                } else {
                    0.0
                };

                println!(
                    "Move ({x:3}, {y:3}) | n: {n:<5} | q: {q:>6.3} | P: {p:.3}",
                    x = x,
                    y = y,
                    n = child.n,
                    q = q,
                    p = child.p
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_expansion_and_backprop() {
        let board = Board::new();
        let mut tree = Tree::new(1.414);

        let (root_idx, root_board) = tree.select(0, board);
        assert_eq!(root_idx, 0);

        tree.expand(root_idx, root_board);
        assert_eq!(tree.nodes[root_idx].s, GameState::Incomplete);
        assert_eq!(tree.nodes[0].children.len(), 961);

        tree.backpropagate(root_idx, root_board, -1.0);
        assert_eq!(tree.nodes[0].n, 1);
        assert_eq!(tree.nodes[0].v, -1.0);

        let (child_idx, _child_board) = tree.select(0, board);

        assert_ne!(child_idx, 0);
        assert_eq!(tree.nodes[child_idx].parent, Some(0));
    }
}
