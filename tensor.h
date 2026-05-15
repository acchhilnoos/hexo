#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define tensor_size(t) (t)->n *(t)->c *(t)->y *(t)->x
#define tensor_at(t, N, C, Y, X)                                               \
  t->buf[(((N) * t->c + (C)) * t->y + (Y)) * t->x + (X)]
#define tensor_grad(t, N, C, Y, X)                                             \
  t->grad[(((N) * t->c + (C)) * t->y + (Y)) * t->x + (X)]

struct Tensor {
  float *buf, *grad;
  // TODO: NHWC with loop order b -> y -> x -> ky -> kx -> ic -> oc
  size_t n, c, y, x;
};

void tensor_init(struct Tensor *t, size_t n, size_t c, size_t y, size_t x);
void tensor_free(struct Tensor *t);

void tensor_reshape(struct Tensor *t, size_t n, size_t c, size_t y, size_t x);

void tensor_zero(struct Tensor *t);
void tensor_zero_grad(struct Tensor *t);

void tensor_add(struct Tensor *src, struct Tensor *dst);
void tensor_add_grad(struct Tensor *src, struct Tensor *dst);

void tensor_relu(struct Tensor *t);
void tensor_relu_grad(struct Tensor *t);
void tensor_tanh(struct Tensor *t);
void tensor_tanh_grad(struct Tensor *t);
void tensor_softmax(struct Tensor *t);

void tensor_conv(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                 struct Tensor *out, size_t pad);
void tensor_conv_grad(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                      struct Tensor *out, size_t pad);

#endif
