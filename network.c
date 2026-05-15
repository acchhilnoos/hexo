#include "network.h"
#include "tensor.h"
#include <float.h>
#include <stdlib.h>
#include <string.h>

struct Network *network_new() {
  struct Network *n = malloc(sizeof(*n));

  tensor_init(&n->ks[0], 16, 2, 3, 3);
  tensor_init(&n->ks[1], 16, 16, 3, 3);
  tensor_init(&n->ks[2], 16, 16, 3, 3);
  tensor_init(&n->ks[3], 16, 16, 3, 3);
  tensor_init(&n->ks[4], 16, 16, 3, 3);
  tensor_init(&n->ks[5], 1, 16, 1, 1);
  tensor_init(&n->ks[6], 1, 16, 1, 1);
  tensor_init(&n->ks[7], 32, BOARD_SIZE * BOARD_SIZE, 1, 1);
  tensor_init(&n->ks[8], 1, 32, 1, 1);

  tensor_init(&n->bs[0], 1, 16, 1, 1);
  tensor_init(&n->bs[1], 1, 16, 1, 1);
  tensor_init(&n->bs[2], 1, 16, 1, 1);
  tensor_init(&n->bs[3], 1, 16, 1, 1);
  tensor_init(&n->bs[4], 1, 16, 1, 1);
  tensor_init(&n->bs[5], 1, 1, 1, 1);
  tensor_init(&n->bs[6], 1, 1, 1, 1);
  tensor_init(&n->bs[7], 1, 32, 1, 1);
  tensor_init(&n->bs[8], 1, 1, 1, 1);

  tensor_init(&n->as[0], 1, 16, BOARD_SIZE, BOARD_SIZE);
  tensor_init(&n->as[1], 1, 16, BOARD_SIZE, BOARD_SIZE);
  tensor_init(&n->as[2], 1, 16, BOARD_SIZE, BOARD_SIZE);
  tensor_init(&n->as[3], 1, 16, BOARD_SIZE, BOARD_SIZE);
  tensor_init(&n->as[4], 1, 16, BOARD_SIZE, BOARD_SIZE);
  tensor_init(&n->as[5], 1, 1, BOARD_SIZE, BOARD_SIZE);
  tensor_init(&n->as[6], 1, 1, BOARD_SIZE, BOARD_SIZE);
  tensor_init(&n->as[7], 1, 32, 1, 1);
  tensor_init(&n->as[8], 1, 1, 1, 1);

  return n;
}

void network_free(struct Network *n) {
  for (size_t i = 0; i < 9; i++) {
    tensor_free(&n->ks[i]);
    tensor_free(&n->bs[i]);
    tensor_free(&n->as[i]);
  }
  free(n);
}

void network_forward(struct Network *n, struct Tensor *inputs,
                     struct Board *b) {
  tensor_conv(inputs, &n->ks[0], &n->bs[0], &n->as[0], 1);
  tensor_relu(&n->as[0]);

  tensor_conv(&n->as[0], &n->ks[1], &n->bs[1], &n->as[1], 1);
  tensor_relu(&n->as[1]);
  tensor_conv(&n->as[1], &n->ks[2], &n->bs[2], &n->as[2], 1);
  tensor_add(&n->as[0], &n->as[2]);
  tensor_relu(&n->as[2]);

  tensor_conv(&n->as[2], &n->ks[3], &n->bs[3], &n->as[3], 1);
  tensor_relu(&n->as[3]);
  tensor_conv(&n->as[3], &n->ks[4], &n->bs[4], &n->as[4], 1);
  tensor_add(&n->as[2], &n->as[4]);
  tensor_relu(&n->as[4]);

  tensor_conv(&n->as[4], &n->ks[5], &n->bs[5], &n->as[5], 0);
  for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
    if (!legal(b, i))
      n->as[5].buf[i] = -FLT_MAX;
  tensor_softmax(&n->as[5]);

  tensor_conv(&n->as[4], &n->ks[6], &n->bs[6], &n->as[6], 0);
  tensor_relu(&n->as[6]);
  tensor_reshape(&n->as[6], 1, BOARD_SIZE * BOARD_SIZE, 1, 1);
  tensor_conv(&n->as[6], &n->ks[7], &n->bs[7], &n->as[7], 0);
  tensor_reshape(&n->as[6], 1, 1, BOARD_SIZE, BOARD_SIZE);

  tensor_relu(&n->as[7]);
  tensor_conv(&n->as[7], &n->ks[8], &n->bs[8], &n->as[8], 0);
  tensor_tanh(&n->as[8]);
}

void network_backward(struct Network *n, struct Tensor *inputs,
                      struct Tensor *loss_p, struct Tensor *loss_v) {
  for (size_t i = 0; i < 9; i++) {
    tensor_zero_grad(&n->ks[i]);
    tensor_zero_grad(&n->bs[i]);
    tensor_zero_grad(&n->as[i]);
  }
  if (inputs->grad)
    tensor_zero_grad(inputs);

  memcpy(n->as[5].grad, loss_p->buf,
         tensor_size(loss_p) * sizeof(*loss_p->buf));
  memcpy(n->as[8].grad, loss_v->buf,
         tensor_size(loss_v) * sizeof(*loss_v->buf));

  tensor_tanh_grad(&n->as[8]);
  tensor_conv_grad(&n->as[7], &n->ks[8], &n->bs[8], &n->as[8], 0);
  tensor_relu_grad(&n->as[7]);

  tensor_reshape(&n->as[6], 1, BOARD_SIZE * BOARD_SIZE, 1, 1);
  tensor_conv_grad(&n->as[6], &n->ks[7], &n->bs[7], &n->as[7], 0);
  tensor_reshape(&n->as[6], 1, 1, BOARD_SIZE, BOARD_SIZE);
  tensor_relu_grad(&n->as[6]);
  tensor_conv_grad(&n->as[4], &n->ks[6], &n->bs[6], &n->as[6], 0);

  tensor_conv_grad(&n->as[4], &n->ks[5], &n->bs[5], &n->as[5], 0);

  tensor_relu_grad(&n->as[4]);
  tensor_add_grad(&n->as[2], &n->as[4]);
  tensor_conv_grad(&n->as[3], &n->ks[4], &n->bs[4], &n->as[4], 1);
  tensor_relu_grad(&n->as[3]);
  tensor_conv_grad(&n->as[2], &n->ks[3], &n->bs[3], &n->as[3], 1);

  tensor_relu_grad(&n->as[2]);
  tensor_add_grad(&n->as[0], &n->as[2]);
  tensor_conv_grad(&n->as[1], &n->ks[2], &n->bs[2], &n->as[2], 1);
  tensor_relu_grad(&n->as[1]);
  tensor_conv_grad(&n->as[0], &n->ks[1], &n->bs[1], &n->as[1], 1);

  tensor_relu_grad(&n->as[0]);
  tensor_conv_grad(inputs, &n->ks[0], &n->bs[0], &n->as[0], 1);
}

void network_sgd(struct Network *n, float alpha) {
  for (size_t i = 0; i < 9; i++) {
    for (size_t j = 0; j < tensor_size(&n->ks[i]); j++)
      n->ks[i].buf[j] -= alpha * n->ks[i].grad[j];
    for (size_t j = 0; j < tensor_size(&n->bs[i]); j++)
      n->bs[i].buf[j] -= alpha * n->bs[i].grad[j];
  }
}
