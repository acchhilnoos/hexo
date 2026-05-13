#ifndef BOARD_H
#define BOARD_H

#define BOARD_SIZE 15
#define BOARD_OFFSET (BOARD_SIZE / 2)

#define xy_to_idx(x, y) (((y) + BOARD_OFFSET) * BOARD_SIZE + (x) + BOARD_OFFSET)
#define idx_to_x(idx) ((int)((idx) % BOARD_SIZE) - BOARD_OFFSET)
#define idx_to_y(idx) ((int)((idx) / BOARD_SIZE) - BOARD_OFFSET)

#include <stdbool.h>
#include <stddef.h>

enum Player { PLAYER_X, PLAYER_O };
enum GameState { GAME_X, GAME_O, GAME_DRAW, GAME_CONT };
enum CellState { CELL_X, CELL_O, CELL_EMPTY };
struct Board {
  enum CellState cells[BOARD_SIZE * BOARD_SIZE];
  size_t cell_to_empty[BOARD_SIZE * BOARD_SIZE];
  size_t empty_to_cell[BOARD_SIZE * BOARD_SIZE];
  size_t n_empty;
};

struct Board *board_new();
void board_copy(struct Board *src, struct Board *dst);

enum Player get_player(struct Board *b);
enum GameState get_state(struct Board *b, size_t idx, enum Player p);
bool legal(struct Board *b, size_t idx);

enum GameState play(struct Board *b, size_t idx);

void board_print(struct Board *b);

#endif
