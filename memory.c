#include "memory.h"
#include <string.h>

void mem_push(struct MemoryBuffer *b, struct Tensor *state, struct Tensor *pol,
              enum Player turn, float z) {
  memcpy(b->buf[b->head].state, state->buf, sizeof(b->buf[b->head].state));
  memcpy(b->buf[b->head].pol, pol->buf, sizeof(b->buf[b->head].pol));
  b->buf[b->head].turn = turn;
  b->buf[b->head].z    = z;

  b->head = (b->head + 1) % MEM_BUF_MAX;
  if (b->n < MEM_BUF_MAX)
    b->n++;
}
