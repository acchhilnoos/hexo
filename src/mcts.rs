use crate::board::Board;
use rand::Rng;

#[derive(Debug, Eq, PartialEq)]
enum GameState {
    Win,
    Loss,
    Draw,
    Incomplete,
}

#[derive(Clone, Debug)]
struct Node {
    parent: Option<usize>,
    children: Vec<usize>,

    n: usize,
    v: f32,
    p: f32,

    a: Option<(i32, i32)>,
    t: bool,
}

pub struct Tree {
    nodes: Vec<Node>,
    c: f32,
}

impl Node {
    fn ucb1(&self, c: f32, n_parent: usize) -> f32 {
        if self.n == 0 {
            return f32::INFINITY;
        }

        let q = self.v / self.n as f32;
        let u = c * self.p * (n_parent as f32).sqrt() / (1.0 + self.n as f32);

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
                t: false,
            }],
            c: c,
        }
    }

    fn select(&self, root_idx: usize, mut root_board: Board) -> (usize, Board) {
        let mut parent_idx = root_idx;

        while !self.nodes[parent_idx].children.is_empty() {
            let parent = &self.nodes[parent_idx];

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

            let player = root_board.next_player();
            let a = self.nodes[parent_idx]
                .a
                .expect("Tree::select() state found without action");
            root_board.set(a.0, a.1, player);
        }

        (parent_idx, root_board)
    }

    fn expand(&mut self, leaf_idx: usize, leaf_board: Board) -> GameState {
        if let Some((x, y)) = self.nodes[leaf_idx].a {
            let parent_player = {
                let mut temp = leaf_board;
                temp.n_pieces -= 1;
                temp.next_player()
            };
            if leaf_board.check_win(x, y, parent_player) {
                self.nodes[leaf_idx].t = true;

                return if parent_player == leaf_board.next_player() {
                    GameState::Win
                } else {
                    GameState::Loss
                };
            }
        }

        let actions = leaf_board.legal();
        if actions.is_empty() {
            self.nodes[leaf_idx].t = true;
            return GameState::Draw;
        }

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
                t: false,
            });
            self.nodes[leaf_idx].children.push(child_idx);
        }

        GameState::Incomplete
    }

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

    pub fn search(&mut self, iterations: usize, root_board: Board) -> (i32, i32) {
        for _ in 0..iterations {
            let (leaf_idx, leaf_board) = self.select(0, root_board);

            let v = match self.expand(leaf_idx, leaf_board) {
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

        self.nodes[max_child_idx]
            .a
            .expect("Tree::search() state found without action")
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

        let expanded = tree.expand(root_idx, root_board);
        assert_eq!(expanded, GameState::Incomplete);
        assert_eq!(tree.nodes[0].children.len(), 225);

        tree.backpropagate(root_idx, root_board, -1.0);
        assert_eq!(tree.nodes[0].n, 1);
        assert_eq!(tree.nodes[0].v, -1.0);

        let (child_idx, _child_board) = tree.select(0, board);

        assert_ne!(child_idx, 0);
        assert_eq!(tree.nodes[child_idx].parent, Some(0));
    }
}
