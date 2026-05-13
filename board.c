#include "board.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define get(b, x, y) (enum Player)(b)->cells[xy_to_idx(x, y)]

struct Board *board_new() {
  struct Board *b = malloc(sizeof(*b));
  if (!b)
    return NULL;

  for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
    b->cells[i]         = CELL_EMPTY;
    b->cell_to_empty[i] = i;
    b->empty[i]         = i;
  }
  b->n_empty = BOARD_SIZE * BOARD_SIZE;

  return b;
}

void board_copy(struct Board *src, struct Board *dst) { *dst = *src; }

enum Player get_player(struct Board *b) {
  return (((b->n_empty + 1) / 2) % 2 == 1) ? PLAYER_X : PLAYER_O;
}

bool legal(struct Board *b, size_t idx) { return b->cells[idx] == CELL_EMPTY; }

enum GameState get_state(struct Board *b, size_t idx, enum Player p) {
  int x    = idx_to_x(idx);
  int y    = idx_to_y(idx);
  int lb_x = ((x - 5) > (-BOARD_OFFSET) ? x - 5 : -BOARD_OFFSET);
  int lb_y = ((y - 5) > (-BOARD_OFFSET) ? y - 5 : -BOARD_OFFSET);
  int ub_x = ((x + 5) < (BOARD_OFFSET) ? x + 5 : BOARD_OFFSET);
  int ub_y = ((y + 5) < (BOARD_OFFSET) ? y + 5 : BOARD_OFFSET);

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
  else if (b->n_empty == 0)
    return GAME_DRAW;
  else
    return GAME_CONT;
}

enum GameState play(struct Board *b, size_t idx) {
  enum Player p = get_player(b);
  b->cells[idx] = ((enum CellState)p);

  size_t empty_idx = b->cell_to_empty[idx];
  size_t last_idx  = b->empty[b->n_empty - 1];

  b->empty[empty_idx]        = last_idx;
  b->cell_to_empty[last_idx] = empty_idx;
  b->n_empty--;
  return get_state(b, idx, p);
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
