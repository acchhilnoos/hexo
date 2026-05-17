#ifndef MEMORY_H
#define MEMORY_H

#include "board.h"
#include "tensor.h"

#define MEM_BUF_MAX 8192

struct Memory {
  float state[2 * BOARD_SIZE * BOARD_SIZE];
  float pol[BOARD_SIZE * BOARD_SIZE];
  enum Player turn;
  float z;
};

struct MemoryBuffer {
  struct Memory buf[MEM_BUF_MAX];
  size_t head;
  size_t n;
};

void mem_push(struct MemoryBuffer *b, struct Tensor *s, struct Tensor *pol,
              enum Player p, float z);

#endif
