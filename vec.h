#ifndef VEC_H
#define VEC_H

#define def_vec(t)                                                             \
  struct t##_vec {                                                             \
    struct t *data;                                                            \
    size_t len;                                                                \
    size_t cap;                                                                \
  }

#define vec_init(v, n)                                                         \
  do {                                                                         \
    (v).data = calloc((n), sizeof(*(v).data));                                 \
    (v).len = 0;                                                               \
    (v).cap = (n);                                                             \
  } while (0)

#define vec_at(v, i) (v).data[(i)]

#define vec_push(v, x)                                                         \
  do {                                                                         \
    if ((v).len + 1 >= (v).cap) {                                              \
      (v).data = realloc((v).data, (v).cap * 2 * sizeof(*(v).data));           \
      (v).cap *= 2;                                                            \
      (v).data[(v).len++] = (x);                                               \
    }                                                                          \
  } while (0)

#endif
