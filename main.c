#include "board.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  struct Board *b = board_new();

  play(b, -10, 10);
  play(b, 0,0);
  play(b, 1,1);
  play(b, -11, 11);
  play(b, -12, 12);
  play(b, 2,2);
  play(b, 3,3);
  play(b, -13, 13);
  play(b, -14, 14);
  play(b, 4,4);
  play(b, 5,5);
  enum GameState x = play(b, -15, 15);
  board_print(b);

  free(b);
  return x;
}
