#ifndef VEC_H
#define VEC_H

#include <stddef.h>
#include <stdlib.h>

#define def_vec(t)                                                             \
  struct t##_vec {                                                             \
    struct t *buf;                                                             \
    size_t len, cap;                                                           \
  }

#define vec_init(v, n)                                                         \
  do {                                                                         \
    (v).buf = calloc((n), sizeof(*(v).buf));                                   \
    if (!(v).buf)                                                              \
      exit(1);                                                                 \
    (v).len = 0;                                                               \
    (v).cap = (n);                                                             \
  } while (0)

#define vec_at(v, i) (v).buf[(i)]

#define vec_push(v, x)                                                         \
  do {                                                                         \
    if ((v).len >= (v).cap) {                                                  \
      (v).cap = (v).cap == 0 ? 4 : (v).cap * 2;                                \
      void *_vec_push_temp = realloc((v).buf, (v).cap * sizeof(*(v).buf));     \
      if (!_vec_push_temp)                                                     \
        exit(1);                                                               \
      (v).buf = _vec_push_temp;                                                \
    }                                                                          \
    (v).buf[(v).len++] = (x);                                                  \
  } while (0)

#define vec_free(v)                                                            \
  do {                                                                         \
    free((v).buf);                                                             \
    (v).buf = NULL;                                                            \
  } while (0)

#endif
