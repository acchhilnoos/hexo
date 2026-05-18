#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void randomize_tensor(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++) {
    t->buf[i]  = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    t->grad[i] = 0.0f;
  }
}

float compute_loss(struct Tensor *out, struct Tensor *out_grad) {
  float loss = 0.0f;
  for (size_t i = 0; i < tensor_size(out); i++) {
    loss += out->buf[i] * out_grad->buf[i];
  }
  return loss;
}

int check_tensor_grad(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                      struct Tensor *out, struct Tensor *out_grad, size_t pad,
                      struct Tensor *target, const char *name) {
  float epsilon  = 1e-3f;
  int   errors   = 0;
  float max_diff = 0.0f;

  for (size_t i = 0; i < tensor_size(target); i++) {
    float orig = target->buf[i];

    target->buf[i] = orig + epsilon;
    tensor_conv(in, k, bias, out, pad);
    float loss_plus = compute_loss(out, out_grad);

    target->buf[i] = orig - epsilon;
    tensor_conv(in, k, bias, out, pad);
    float loss_minus = compute_loss(out, out_grad);

    target->buf[i] = orig;

    float num_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
    float ana_grad = target->grad[i];

    float diff      = fabsf(num_grad - ana_grad);
    float rel_error = diff / (fabsf(ana_grad) + fabsf(num_grad) + 1e-8f);

    if (diff > max_diff)
      max_diff = diff;

    if (rel_error > 5e-2f && diff > 1e-3f) {
      printf("  [FAIL] %s grad at index %zu: num=%f, ana=%f, diff=%f\n", name,
             i, num_grad, ana_grad, diff);
      errors++;
      if (errors > 5) {
        printf("  ... too many errors, aborting %s check.\n", name);
        return errors;
      }
    }
  }
  if (errors == 0) {
    printf("  [OK] %s gradients match perfectly! (max diff: %f)\n", name,
           max_diff);
  }
  return errors;
}

void test_conv2d() {
  printf("Testing Conv2D Forward and Backward (Finite Difference)...\n");

  struct Tensor in, k, bias, out, out_grad;
  tensor_init(&in, 2, 5, 5, 2);
  tensor_init(&k, 3, 3, 2, 3);
  tensor_init(&bias, 1, 1, 1, 3);
  tensor_init(&out, 2, 5, 5, 3);
  tensor_init(&out_grad, 2, 5, 5, 3);

  randomize_tensor(&in);
  randomize_tensor(&k);
  randomize_tensor(&bias);
  randomize_tensor(&out);
  randomize_tensor(&out_grad);

  for (size_t i = 0; i < tensor_size(&out); i++) {
    out.grad[i] = out_grad.buf[i];
  }

  tensor_zero_grad(&in);
  tensor_zero_grad(&k);
  tensor_zero_grad(&bias);

  tensor_conv_grad(&in, &k, &bias, &out, 1);

  int errs = 0;
  errs +=
      check_tensor_grad(&in, &k, &bias, &out, &out_grad, 1, &k, "Kernel (k)");
  errs +=
      check_tensor_grad(&in, &k, &bias, &out, &out_grad, 1, &in, "Input (in)");
  errs += check_tensor_grad(&in, &k, &bias, &out, &out_grad, 1, &bias,
                            "Bias (bias)");

  if (errs == 0) {
    printf("All Conv2D gradient checks passed! The math is solid.\n\n");
  } else {
    printf("Conv2D gradient checks FAILED. Total errors: %d\n\n", errs);
  }

  tensor_free(&in);
  tensor_free(&k);
  tensor_free(&bias);
  tensor_free(&out);
  tensor_free(&out_grad);
}

int main() {
  srand(42);
  test_conv2d();
  return 0;
}
