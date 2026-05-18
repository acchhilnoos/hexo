#include "tensor.h"
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void tensor_init(struct Tensor *t, size_t n, size_t y, size_t x, size_t c) {
  t->buf  = calloc(n * y * x * c, sizeof(*t->buf));
  t->grad = calloc(n * y * x * c, sizeof(*t->grad));

  t->n = n;
  t->y = y;
  t->x = x;
  t->c = c;
}

void tensor_reshape(struct Tensor *t, size_t n, size_t y, size_t x, size_t c) {
  t->n = n;
  t->y = y;
  t->x = x;
  t->c = c;
}

void tensor_zero(struct Tensor *t) {
  memset(t->buf, 0, tensor_size(t) * sizeof(*t->buf));
}
void tensor_zero_grad(struct Tensor *t) {
  memset(t->grad, 0, tensor_size(t) * sizeof(*t->grad));
}

void tensor_add(struct Tensor *src, struct Tensor *dst) {
  size_t oc         = src->c;
  size_t plane_size = dst->y * dst->x;

  if (src->n == 1 && src->y == 1 && src->x == 1)
    for (size_t b = 0; b < dst->n; b++)
      for (size_t s = 0; s < dst->y * dst->x; s++)
        for (size_t c = 0; c < oc; c++)
          dst->buf[(b * plane_size + s) * oc + c] += src->buf[c];
  else
    for (size_t i = 0; i < tensor_size(dst); i++)
      dst->buf[i] += src->buf[i];
}

void tensor_add_grad(struct Tensor *src, struct Tensor *dst) {
  size_t oc         = src->c;
  size_t plane_size = dst->y * dst->x;

  if (src->n == 1 && src->y == 1 && src->x == 1)
    for (size_t b = 0; b < dst->n; b++)
      for (size_t s = 0; s < plane_size; s++)
        for (size_t c = 0; c < oc; c++)
          src->grad[c] += dst->grad[(b * plane_size + s) * oc + c];
  else
    for (size_t i = 0; i < tensor_size(dst); i++)
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
  size_t stride = t->y * t->x * t->c;
  for (size_t b = 0; b < t->n; b++) {
    float *buf = t->buf + b * stride;
    float  max = -FLT_MAX;
    for (size_t i = 0; i < stride; i++)
      if (buf[i] > max)
        max = buf[i];

    float sum = 0.0f;
    for (size_t i = 0; i < stride; i++) {
      buf[i] = expf(buf[i] - max);
      sum += buf[i];
    }

    for (size_t i = 0; i < stride; i++)
      buf[i] /= sum + 1e-8f;
  }
}

void tensor_conv(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                 struct Tensor *out, size_t pad) {
  /* in:    [b] [y] [x][ic]
   * k:    [ky][kx][ic][oc]
   * bias:  [1] [1] [1][oc]
   * out:   [b] [y] [x][oc]
   */

  // for (b) for (y) for (x)
  for (size_t b = 0; b < in->n; b++) {
    for (size_t y = 0; y < out->y; y++) {
      for (size_t x = 0; x < out->x; x++) {
        // init out[b][y][x] with bias
        float out_vals[out->c];
        for (size_t oc = 0; oc < out->c; oc++)
          out_vals[oc] = bias ? tensor_at(bias, 0, 0, 0, oc) : 0.0f;

        // for (ky) for (kx) for (ic)
        for (size_t ky = 0; ky < k->n; ky++) {
          int in_y = (int)y + (int)ky - (int)pad;
          if (in_y < 0 || in_y >= (int)in->y)
            continue;

          for (size_t kx = 0; kx < k->y; kx++) {
            int in_x = (int)x + (int)kx - (int)pad;
            if (in_x < 0 || in_x >= (int)in->x)
              continue;

            for (size_t ic = 0; ic < in->c; ic++) {
              // in[b][y+ky][x+kx][ic]
              float in_val = tensor_at(in, b, in_y, in_x, ic);
              // for (oc)
              for (size_t oc = 0; oc < out->c; oc++) {
                // out[b][y][x][oc] +=
                //   in[b][y+ky][x+kx][ic] * k[ky][kx][ic][oc]
                out_vals[oc] += in_val * tensor_at(k, ky, kx, ic, oc);
              }
            }
          }
        }
        // write out[b][y][x]
        for (size_t oc = 0; oc < out->c; oc++) {
          tensor_at(out, b, y, x, oc) = out_vals[oc];
        }
      }
    }
  }
}

void tensor_conv_grad(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                      struct Tensor *out, size_t pad) {
  /* in:    [b] [y] [x][ic]
   * k:    [ky][kx][ic][oc]
   * bias:  [1] [1] [1][oc]
   * out:   [b] [y] [x][oc]
   */

  // for (b) for (y) for (x)
  for (size_t b = 0; b < in->n; b++) {
    for (size_t in_y = 0; in_y < in->y; in_y++) {
      for (size_t in_x = 0; in_x < in->x; in_x++) {
        // in_grad[b][y][x]
        float in_grads[in->c];
        for (size_t ic = 0; ic < in->c; ic++)
          in_grads[ic] = tensor_grad(in, b, in_y, in_x, ic);

        // for (ky) for (kx) for (ic) for (oc)
        for (size_t ky = 0; ky < k->n; ky++) {
          int y = (int)in_y - (int)ky + (int)pad;
          if (y < 0 || y >= (int)out->y)
            continue;

          for (size_t kx = 0; kx < k->y; kx++) {
            int x = (int)in_x - (int)kx + (int)pad;
            if (x < 0 || x >= (int)out->x)
              continue;

            for (size_t ic = 0; ic < in->c; ic++) {
              // in_grad[b][y][x][ic] +=
              //   out_grad[b][y-ky][x-kx][oc] * k[ky][kx][ic][oc]
              float sum = 0.0f;
              for (size_t oc = 0; oc < out->c; oc++) {
                sum += tensor_grad(out, b, y, x, oc) *
                       tensor_at(k, ky, kx, ic, oc);
              }
              in_grads[ic] += sum;
            }
          }
        }
        // write in_grad[b][y][x]
        for (size_t ic = 0; ic < in->c; ic++)
          tensor_grad(in, b, in_y, in_x, ic) = in_grads[ic];
      }
    }
  }

  // for (ky) for (kx) for (b) for (y) for (x) for (ic)
  for (size_t ky = 0; ky < k->n; ky++) {
    for (size_t kx = 0; kx < k->y; kx++) {
      for (size_t b = 0; b < in->n; b++) {
        for (size_t y = 0; y < out->y; y++) {
          int in_y = (int)y + (int)ky - (int)pad;
          if (in_y < 0 || in_y >= (int)in->y)
            continue;

          for (size_t x = 0; x < out->x; x++) {
            int in_x = (int)x + (int)kx - (int)pad;
            if (in_x < 0 || in_x >= (int)in->x)
              continue;

            for (size_t ic = 0; ic < in->c; ic++) {
              // in[b][y+ky][x+kx][ic]
              float in_val = tensor_at(in, b, in_y, in_x, ic);
              // for (oc)
              for (size_t oc = 0; oc < out->c; oc++) {
                // k_grad[ky][kx][ic][oc] +=
                //   out_grad[b][y][x][oc] * in[b][y+ky][x+kx][ic]
                tensor_grad(k, ky, kx, ic, oc) +=
                    tensor_grad(out, b, y, x, oc) * in_val;
              }
            }
          }
        }
      }
    }
  }

  if (bias) {
    tensor_add_grad(bias, out);
  }
}
