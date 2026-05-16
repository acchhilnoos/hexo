#ifndef NETWORK_H
#define NETWORK_H

#include "board.h"
#include "tensor.h"

struct Network {
  /* T:
   * 16x 2x3x3
   * 16x16x3x3, relu, 16x16x3x3, res, relu
   * 16x16x3x3, relu, 16x16x3x3, res, relu
   *
   * P:
   * 1x 16x1x1, mask, softmax
   *
   * V:
   * 1x 16x1x1, relu, reshape, 32xBS*BSx1x1, reshape, 1x32x1x1
   */
  struct Tensor ks[9];
  struct Tensor bs[9];
  struct Tensor as[9];
};

struct Network *network_new();
void network_free(struct Network *n);

void network_zero_grad(struct Network *n);

void network_forward(struct Network *n, struct Tensor *inputs, struct Board *b);
void network_backward(struct Network *n, struct Tensor *inputs,
                      struct Tensor *loss_p, struct Tensor *loss_v);

void network_sgd(struct Network *n, float alpha);

void network_save(struct Network *n, const char *path);
void network_load(struct Network *n, const char *path);

void network_benchmark();

#endif
