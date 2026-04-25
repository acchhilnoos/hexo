use std::fmt;

pub const BOARD_SIZE: usize = 31;
pub const BOARD_OFFSET: i32 = (BOARD_SIZE / 2) as i32; 

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CellState {
    Empty,
    X,
    O,
}

impl fmt::Display for CellState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CellState::Empty => ".",
                CellState::X => "X",
                CellState::O => "O",
            }
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Board {
    cells: [CellState; BOARD_SIZE * BOARD_SIZE],
    pub n_pieces: usize,
}

impl Board {
    pub fn new() -> Self {
        Self {
            cells: [CellState::Empty; BOARD_SIZE * BOARD_SIZE],
            n_pieces: 0,
        }
    }

    fn ctoi(x: i32, y: i32) -> Option<usize> {
        let sh_x = x + BOARD_OFFSET;
        let sh_y = y + BOARD_OFFSET;

        if sh_x < 0 || sh_x >= BOARD_SIZE as i32 || sh_y < 0 || sh_y >= BOARD_SIZE as i32 {
            None
        } else {
            Some((sh_y as usize) * BOARD_SIZE + (sh_x as usize))
        }
    }

    fn itoc(idx: usize) -> Option<(i32, i32)> {
        if idx >= BOARD_SIZE * BOARD_SIZE {
            None
        } else {
            let sh_y = idx / BOARD_SIZE;
            let sh_x = idx % BOARD_SIZE;
            Some(((sh_x as i32 - BOARD_OFFSET), (sh_y as i32 - BOARD_OFFSET)))
        }
    }

    pub fn next_player(&self) -> CellState {
        if ((self.n_pieces + 1) / 2) % 2 == 0 {
            CellState::X
        } else {
            CellState::O
        }
    }

    pub fn get(&self, x: i32, y: i32) -> Option<CellState> {
        Self::ctoi(x, y).map(|idx| self.cells[idx])
    }

    pub fn set(&mut self, x: i32, y: i32, cs: CellState) {
        let idx = Self::ctoi(x, y).expect("Board::set() index out of bounds");
        assert!(
            self.cells[idx] == CellState::Empty,
            "Board::set() index occupied"
        );

        self.cells[idx] = cs;
        self.n_pieces += 1;
    }

    pub fn legal(&self) -> Vec<(i32, i32)> {
        self.cells
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s == CellState::Empty)
            .filter_map(|(idx, _)| Self::itoc(idx))
            .collect()
    }

    pub fn check_win(&self, x: i32, y: i32, cs: CellState) -> bool {
        if cs == CellState::Empty {
            return false;
        }
        fn go_f(b: &Board, cs: CellState, mut x: i32, mut y: i32, dx: i32, dy: i32) -> i32 {
            let mut count = 0;
            loop {
                x += dx;
                y += dy;
                if b.get(x, y) == Some(cs) {
                    count += 1;
                } else {
                    break;
                }
            }
            count
        }
        let go = |dx, dy| go_f(self, cs, x, y, dx, dy) + go_f(self, cs, x, y, -dx, -dy);

        go(1, 0) >= 5 || go(0, 1) >= 5 || go(1, -1) >= 5
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..BOARD_SIZE {
            for _ in 0..y {
                write!(f, " ")?;
            }
            for x in 0..BOARD_SIZE {
                let idx = y * BOARD_SIZE + x;
                let state = self.cells[idx];
                write!(f, "{} ", state)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get() {
        let mut b = Board::new();

        b.set(0, 0, CellState::X);
        assert_eq!(b.get(0, 0), Some(CellState::X));
    }

    #[test]
    #[should_panic]
    fn test_set_occupied() {
        let mut b = Board::new();

        b.set(0, 0, CellState::X);
        assert_eq!(b.get(0, 0), Some(CellState::X));

        b.set(0, 0, CellState::O);
    }

    #[test]
    #[should_panic]
    fn test_set_out_of_bounds() {
        let mut b = Board::new();

        b.set(100, 100, CellState::X);
    }

    #[test]
    fn test_legal_moves() {
        let mut b = Board::new();

        let legal = b.legal();
        assert_eq!(legal.len(), BOARD_SIZE * BOARD_SIZE);
        b.set(0, 0, CellState::X);

        let legal = b.legal();
        assert_eq!(legal.len(), (BOARD_SIZE * BOARD_SIZE) - 1);
        assert!(!legal.contains(&(0, 0)));
    }

    #[test]
    fn test_check_win_horizontal() {
        let mut b = Board::new();
        for x in 0..5 {
            b.set(x, 0, CellState::X);
            assert!(!b.check_win(x, 0, CellState::X));
        }
        b.set(5, 0, CellState::X);
        assert!(b.check_win(5, 0, CellState::X));
    }

    #[test]
    fn test_check_win_vertical() {
        let mut b = Board::new();
        for y in -5..0 {
            b.set(0, y, CellState::O);
            assert!(!b.check_win(0, y, CellState::O));
        }
        b.set(0, 0, CellState::O);
        assert!(b.check_win(0, 0, CellState::O));
    }
    #[test]
    fn test_check_win_diagonal() {
        let mut b = Board::new();
        for i in 0..6 {
            if i != 2 {
                b.set(i, -i, CellState::X);
                assert!(!b.check_win(i, -i, CellState::O));
            }
        }
        b.set(2, -2, CellState::X);
        assert!(b.check_win(2, -2, CellState::X));
    }
}
