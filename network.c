#include "network.h"
#include "tensor.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_BATCH_SIZE 64

struct Network *network_new(void) {
  struct Network *n = malloc(sizeof(*n));

  // kernels: [ky, kx, ic, oc]
  tensor_init(&n->ks[0], 3, 3, 2, 16);
  tensor_init(&n->ks[1], 3, 3, 16, 16);
  tensor_init(&n->ks[2], 3, 3, 16, 16);
  tensor_init(&n->ks[3], 3, 3, 16, 16);
  tensor_init(&n->ks[4], 3, 3, 16, 16);
  tensor_init(&n->ks[5], 1, 1, 16, 1);
  tensor_init(&n->ks[6], 1, 1, 16, 1);
  tensor_init(&n->ks[7], 1, 1, BOARD_SIZE * BOARD_SIZE, 32);
  tensor_init(&n->ks[8], 1, 1, 32, 1);

  // biases: [1, 1, 1, oc]
  tensor_init(&n->bs[0], 1, 1, 1, 16);
  tensor_init(&n->bs[1], 1, 1, 1, 16);
  tensor_init(&n->bs[2], 1, 1, 1, 16);
  tensor_init(&n->bs[3], 1, 1, 1, 16);
  tensor_init(&n->bs[4], 1, 1, 1, 16);
  tensor_init(&n->bs[5], 1, 1, 1, 1);
  tensor_init(&n->bs[6], 1, 1, 1, 1);
  tensor_init(&n->bs[7], 1, 1, 1, 32);
  tensor_init(&n->bs[8], 1, 1, 1, 1);

  // activations: [MAX_BATCH_SIZE, y, x, c]
  tensor_init(&n->as[0], MAX_BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, 16);
  tensor_init(&n->as[1], MAX_BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, 16);
  tensor_init(&n->as[2], MAX_BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, 16);
  tensor_init(&n->as[3], MAX_BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, 16);
  tensor_init(&n->as[4], MAX_BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, 16);
  tensor_init(&n->as[5], MAX_BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, 1);
  tensor_init(&n->as[6], MAX_BATCH_SIZE, BOARD_SIZE, BOARD_SIZE, 1);
  tensor_init(&n->as[7], MAX_BATCH_SIZE, 1, 1, 32);
  tensor_init(&n->as[8], MAX_BATCH_SIZE, 1, 1, 1);

  for (size_t i = 0; i < 9; i++) {
    struct Tensor *ks = &n->ks[i];
    struct Tensor *bs = &n->bs[i];

    tensor_init(&n->m_ks[i], ks->n, ks->y, ks->x, ks->c);
    tensor_init(&n->m_bs[i], bs->n, bs->y, bs->x, bs->c);

    float fan_in = (float)(ks->n * ks->y * ks->x);
    float limit  = sqrtf(6.0f / (fan_in + ks->c));

    for (size_t j = 0; j < tensor_size(&n->ks[i]); j++)
      ks->buf[j] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * limit;
  }

  for (int i = 0; i < 9; i++)
    n->as[i].n = 1;

  return n;
}

void network_free(struct Network *n) {
  for (size_t i = 0; i < 9; i++) {
    tensor_free(&n->ks[i]);
    tensor_free(&n->bs[i]);
    tensor_free(&n->as[i]);
    tensor_free(&n->m_ks[i]);
    tensor_free(&n->m_bs[i]);
  }
  free(n);
}

void network_zero_grad(struct Network *n) {
  for (size_t i = 0; i < 9; i++) {
    tensor_zero_grad(&n->ks[i]);
    tensor_zero_grad(&n->bs[i]);
    tensor_zero_grad(&n->as[i]);
  }
}

void network_forward(struct Network *n, struct Tensor *inputs,
                     struct Board *b) {
  // trunk
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

  // policy head
  tensor_conv(&n->as[4], &n->ks[5], &n->bs[5], &n->as[5], 0);
  if (b)
    for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
      if (!legal(b, i))
        n->as[5].buf[i] = -FLT_MAX;
  tensor_softmax(&n->as[5]);

  // value head
  tensor_conv(&n->as[4], &n->ks[6], &n->bs[6], &n->as[6], 0);
  tensor_relu(&n->as[6]);

  tensor_reshape(&n->as[6], n->as[6].n, 1, 1, BOARD_SIZE * BOARD_SIZE);
  tensor_conv(&n->as[6], &n->ks[7], &n->bs[7], &n->as[7], 0);
  tensor_reshape(&n->as[6], n->as[6].n, BOARD_SIZE, BOARD_SIZE, 1);

  tensor_relu(&n->as[7]);
  tensor_conv(&n->as[7], &n->ks[8], &n->bs[8], &n->as[8], 0);
  tensor_tanh(&n->as[8]);
}

void network_backward(struct Network *n, struct Tensor *inputs,
                      struct Tensor *loss_p, struct Tensor *loss_v) {
  for (size_t i = 0; i < 9; i++)
    tensor_zero_grad(&n->as[i]);

  memcpy(n->as[5].grad, loss_p->buf,
         tensor_size(loss_p) * sizeof(*loss_p->buf));
  memcpy(n->as[8].grad, loss_v->buf,
         tensor_size(loss_v) * sizeof(*loss_v->buf));

  tensor_tanh_grad(&n->as[8]);
  tensor_conv_grad(&n->as[7], &n->ks[8], &n->bs[8], &n->as[8], 0);
  tensor_relu_grad(&n->as[7]);

  tensor_reshape(&n->as[6], n->as[6].n, 1, 1, BOARD_SIZE * BOARD_SIZE);
  tensor_conv_grad(&n->as[6], &n->ks[7], &n->bs[7], &n->as[7], 0);
  tensor_reshape(&n->as[6], n->as[6].n, BOARD_SIZE, BOARD_SIZE, 1);

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
  float m = 0.9f;
  for (size_t i = 0; i < 9; i++) {
    for (size_t j = 0; j < tensor_size(&n->ks[i]); j++) {
      n->m_ks[i].buf[j] = m * n->m_ks[i].buf[j] + n->ks[i].grad[j];
      n->ks[i].buf[j] -= alpha * n->m_ks[i].buf[j];
    }
    for (size_t j = 0; j < tensor_size(&n->bs[i]); j++) {
      n->m_bs[i].buf[j] = m * n->m_bs[i].buf[j] + n->bs[i].grad[j];
      n->bs[i].buf[j] -= alpha * n->m_bs[i].buf[j];
    }
  }
}

void network_save(struct Network *n, const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f)
    return;
  for (size_t i = 0; i < 9; i++) {
    fwrite(n->ks[i].buf, sizeof(float), tensor_size(&n->ks[i]), f);
    fwrite(n->bs[i].buf, sizeof(float), tensor_size(&n->bs[i]), f);
  }
  fclose(f);
}

void network_load(struct Network *n, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return;
  for (size_t i = 0; i < 9; i++) {
    fread(n->ks[i].buf, sizeof(float), tensor_size(&n->ks[i]), f);
    fread(n->bs[i].buf, sizeof(float), tensor_size(&n->bs[i]), f);
  }
  fclose(f);
}

void network_benchmark(void) {
  printf("Benchmarking (1000 iterations)...\n");

  struct Network *n = network_new();
  struct Board   *b = board_new();

  for (int i = 0; i < 20; i++) {
    if (b->n_empty > 0) {
      play(b, b->empty[rand() % b->n_empty]);
    }
  }

  struct Tensor inputs, loss_p, loss_v;
  tensor_init(&inputs, 1, BOARD_SIZE, BOARD_SIZE, 2);
  tensor_init(&loss_p, 1, BOARD_SIZE, BOARD_SIZE, 1);
  tensor_init(&loss_v, 1, 1, 1, 1);

  for (size_t i = 0; i < tensor_size(&inputs); i++) {
    inputs.buf[i] = (float)(rand() % 2);
  }
  for (size_t i = 0; i < tensor_size(&loss_p); i++) {
    loss_p.buf[i] = (float)rand() / RAND_MAX;
  }
  loss_v.buf[0] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  for (int i = 0; i < 100; i++) {
    network_forward(n, &inputs, b);
    network_backward(n, &inputs, &loss_p, &loss_v);
  }

  clock_t start = clock();
  for (int i = 0; i < 1000; i++) {
    network_forward(n, &inputs, b);
    network_backward(n, &inputs, &loss_p, &loss_v);
  }
  clock_t end = clock();

  double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Time for 1000 Forward+Backward passes: %f seconds\n", time_spent);
  printf("Average time per pass: %f ms\n", (time_spent / 1000.0) * 1000.0);
  printf("Estimated Evals per Second: %.2f\n\n", 1000.0 / time_spent);

  tensor_free(&loss_v);
  tensor_free(&loss_p);
  tensor_free(&inputs);
  network_free(n);
  free(b);
}
