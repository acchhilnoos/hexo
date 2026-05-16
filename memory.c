#include "memory.h"
#include <string.h>

void mem_push(struct MemoryBuffer *b, struct Tensor *state, struct Tensor *pol,
              enum Player turn, float z) {
  if (b->n + 1 >= MEM_BUF_MAX)
    b->n = 0;

  memcpy(b->buf[b->n].state, state->buf, sizeof(b->buf[b->n].state));
  memcpy(b->buf[b->n].pol, pol->buf, sizeof(b->buf[b->n].pol));
  b->buf[b->n].turn = turn;
  b->buf[b->n].z    = z;

  b->n++;
}
