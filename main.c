#include "board.h"
#include "tree.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
  srand(time(NULL));
  struct Board *b = board_new();
  board_print(b);

  enum GameState state = GAME_CONT;
  while (state == GAME_CONT) {
    enum Player  p     = get_player(b);
    struct Tree *t     = tree_new(b, 1.414f);
    size_t       c_idx = search(t, b, 100000);

    size_t move_idx = t->nodes.buf[c_idx].move_idx;

    tree_free(t);

    state = play(b, move_idx);

    board_print(b);
    printf("\nPlayer %c (%d, %d)\n", p == PLAYER_X ? 'X' : 'O',
           idx_to_x(move_idx), idx_to_y(move_idx));
  }

  if (state == GAME_X) {
    printf("\nX wins!\n");
  } else if (state == GAME_O) {
    printf("\nO wins!\n");
  } else {
    printf("\nDraw!\n");
  }

  free(b);
  return 0;
}
