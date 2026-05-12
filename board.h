#ifndef BOARD_H
#define BOARD_H

#define BOARD_SIZE 31
#define BOARD_OFFSET (BOARD_SIZE / 2)

#include <stdbool.h>
#include <stddef.h>

enum Player { PLAYER_X, PLAYER_O };
enum GameState { GAME_X, GAME_O, GAME_DRAW, GAME_CONT };
enum CellState { CELL_X, CELL_O, CELL_EMPTY };
struct Board {
  enum CellState cells[BOARD_SIZE * BOARD_SIZE];
  size_t         n_empty;
};

struct Board *board_new();

enum Player get_player(struct Board *b);
enum GameState get_state(struct Board *b, int x, int y, enum Player p);
bool legal(struct Board *b, int x, int y);

enum GameState play(struct Board *b, int x, int y);

void board_print(struct Board *b);

#endif
