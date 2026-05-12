#include "board.h"
#include <stdio.h>
#include <stdlib.h>

#define max(x, y) ((x) > (y) ? x : y)
#define min(x, y) ((x) < (y) ? x : y)
#define xy_to_idx(x, y) (((y) + BOARD_OFFSET) * BOARD_SIZE + (x) + BOARD_OFFSET)
#define get(b, x, y) (enum Player)(b)->cells[xy_to_idx(x, y)]
#define set(b, x, y, s) (b)->cells[xy_to_idx(x, y)] = (s)

struct Board *board_new() {
  struct Board *b = malloc(sizeof(*b));
  if (!b)
    return NULL;

  for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
    b->cells[i] = CELL_EMPTY;
  b->n_empty = BOARD_SIZE * BOARD_SIZE;

  return b;
}

enum Player check_player(struct Board *b) {
  return (((b->n_empty + 1) / 2) % 2 == 1) ? PLAYER_X : PLAYER_O;
}

bool legal(struct Board *b, int x, int y) {
  return b->cells[xy_to_idx(x, y)] == CELL_EMPTY;
}

enum GameState check_state(struct Board *b, int x, int y, enum Player p) {
  if (b->n_empty == 0)
    return GAME_DRAW;

  int lb_x = max(x - 5, -BOARD_OFFSET);
  int lb_y = max(y - 5, -BOARD_OFFSET);
  int ub_x = min(x + 5, BOARD_OFFSET);
  int ub_y = min(y + 5, BOARD_OFFSET);

  size_t count_h = 1;
  size_t count_v = 1;
  size_t count_d = 1;

  for (int i = x + 1; i <= ub_x && get(b, i, y) == p; i++)
    count_h++;
  for (int i = x - 1; i >= lb_x && get(b, i, y) == p; i--)
    count_h++;

  for (int j = y + 1; j <= ub_y && get(b, x, j) == p; j++)
    count_v++;
  for (int j = y - 1; j >= lb_y && get(b, x, j) == p; j--)
    count_v++;

  for (int i = x + 1, j = y - 1; i <= ub_x && j >= lb_y && get(b, i, j) == p;
       i++, j--)
    count_d++;
  for (int i = x - 1, j = y + 1; i >= lb_x && j <= ub_y && get(b, i, j) == p;
       i--, j++)
    count_d++;

  if (count_h >= 6 || count_v >= 6 || count_d >= 6)
    if (p == PLAYER_X)
      return GAME_X;
    else
      return GAME_O;
  else
    return GAME_CONT;
}

enum GameState play(struct Board *b, int x, int y) {
  enum Player p = check_player(b);
  set(b, x, y, (enum CellState)p);
  b->n_empty--;
  return check_state(b, x, y, p);
}

void board_print(struct Board *b) {
  for (size_t i = BOARD_SIZE; i-- > 0;) {
    for (size_t j = 0; j < i; j++) {
      printf(" ");
    }
    for (size_t j = 0; j < BOARD_SIZE; j++) {
      switch (b->cells[i * BOARD_SIZE + j]) {
      case CELL_X:
        printf("x ");
        break;
      case CELL_O:
        printf("o ");
        break;
      case CELL_EMPTY:
        printf(". ");
        break;
      }
    }
    printf("\n");
  }
}
