#include "tensor.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void tensor_init(struct Tensor *t, size_t n, size_t c, size_t y, size_t x) {
  t->buf  = calloc(n * c * y * x, sizeof(*t->buf));
  t->grad = calloc(n * c * y * x, sizeof(*t->grad));

  t->n = n;
  t->c = c;
  t->y = y;
  t->x = x;
}

void tensor_reshape(struct Tensor *t, size_t n, size_t c, size_t y, size_t x) {
  t->n = n;
  t->c = c;
  t->y = y;
  t->x = x;
}

void tensor_zero(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++)
    t->buf[i] = 0;
}
void tensor_zero_grad(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++)
    t->grad[i] = 0;
}

void tensor_add(struct Tensor *src, struct Tensor *dst) {
  for (size_t i = 0; i < tensor_size(src); i++)
    dst->buf[i] += src->buf[i];
}

void tensor_add_grad(struct Tensor *src, struct Tensor *dst) {
  for (size_t i = 0; i < tensor_size(src); i++)
    src->grad[i] += dst->grad[i];
}

void tensor_free(struct Tensor *t) {
  free(t->buf);
  free(t->grad);
}

void tensor_relu(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++) {
    if (t->buf[i] < 0)
      t->buf[i] = 0;
  }
}
void tensor_relu_grad(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++) {
    if (t->buf[i] <= 0.0f)
      t->grad[i] = 0.0f;
  }
}

void tensor_tanh(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++)
    t->buf[i] = tanhf(t->buf[i]);
}
void tensor_tanh_grad(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++)
    t->grad[i] *= (1.0f - t->buf[i] * t->buf[i]);
}

void tensor_softmax(struct Tensor *t) {
  float max = -FLT_MAX;
  for (size_t i = 0; i < tensor_size(t); i++)
    if (t->buf[i] > max)
      max = t->buf[i];

  float sum = 0.0f;
  for (size_t i = 0; i < tensor_size(t); i++) {
    t->buf[i] = expf(t->buf[i] - max);
    sum += t->buf[i];
  }

  for (size_t i = 0; i < tensor_size(t); i++)
    t->buf[i] /= sum + 1e-8f;
}

void tensor_conv(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                 struct Tensor *out, size_t pad) {
  /* in  [b] [oc][y] [x]
   * k   [oc][ic][ky][kx]
   * out [b] [oc][y] [x]
   */
  memset(out->buf, 0, out->n * out->c * out->y * out->x * sizeof(*out->buf));
  for (size_t b = 0; b < in->n; b++) {
    for (size_t oc = 0; oc < out->c; oc++) {
      for (size_t ic = 0; ic < in->c; ic++) {
        for (size_t ky = 0; ky < k->y; ky++) {
          for (size_t kx = 0; kx < k->x; kx++) {
            float w = tensor_at(k, oc, ic, ky, kx);

            for (size_t y = 0; y < out->y; y++) {
              int in_y = (int)y + (int)ky - (int)pad;
              if (in_y < 0 || in_y >= (int)in->y)
                continue;

              for (size_t x = 0; x < out->x; x++) {
                int in_x = (int)x + (int)kx - (int)pad;
                if (in_x < 0 || in_x >= (int)in->x)
                  continue;

                tensor_at(out, b, oc, y, x) +=
                    tensor_at(in, b, ic, in_y, in_x) * w;
              }
            }
          }
        }
      }
      for (size_t y = 0; y < out->y; y++)
        for (size_t x = 0; x < out->x; x++)
          tensor_at(out, b, oc, y, x) += tensor_at(bias, 0, oc, 0, 0);
    }
  }
}

void tensor_conv_grad(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                      struct Tensor *out, size_t pad) {
  for (size_t b = 0; b < in->n; b++) {
    for (size_t oc = 0; oc < out->c; oc++) {
      for (size_t ic = 0; ic < in->c; ic++) {
        for (size_t ky = 0; ky < k->y; ky++) {
          for (size_t kx = 0; kx < k->x; kx++) {

            for (size_t y = 0; y < out->y; y++) {
              int in_y = (int)y + (int)ky - (int)pad;
              if (in_y < 0 || in_y >= (int)in->y)
                continue;

              for (size_t x = 0; x < out->x; x++) {
                int in_x = (int)x + (int)kx - (int)pad;
                if (in_x < 0 || in_x >= (int)in->x)
                  continue;

                float grad = tensor_grad(out, b, oc, y, x);

                tensor_grad(in, b, ic, in_y, in_x) +=
                    grad * tensor_at(k, oc, ic, ky, kx);
                tensor_grad(k, oc, ic, ky, kx) +=
                    grad * tensor_at(in, b, ic, in_y, in_x);
              }
            }
          }
        }
      }
      for (size_t y = 0; y < out->y; y++)
        for (size_t x = 0; x < out->x; x++)
          tensor_grad(bias, 0, oc, 0, 0) += tensor_grad(out, b, oc, y, x);
    }
  }
}
