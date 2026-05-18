#include "board.h"
#include "memory.h"
#include "network.h"
#include "tensor.h"
#include "tree.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// void play_human_vs_ai(struct Network *network) {
//   struct Board *b = board_new();
//
//   // Starting move
//   play(b, xy_to_idx(0, 0));
//   board_print(b);
//
//   enum GameState gs = GAME_CONT;
//   while (gs == GAME_CONT) {
//     enum Player p = get_player(b);
//
//     if (p == PLAYER_O) {
//       int x, y;
//       printf("Your turn (x y): ");
//       if (scanf("%d %d", &x, &y) != 2)
//         break;
//
//       size_t idx = xy_to_idx(x, y);
//       if (x < -BOARD_OFFSET || x > BOARD_OFFSET || y < -BOARD_OFFSET || y >
//       BOARD_OFFSET || !legal(b, idx)) {
//         printf("Illegal move!\n");
//         continue;
//       }
//       gs = play(b, idx);
//     } else {
//       printf("AI is thinking...\n");
//       struct Tree *t     = tree_new(b, 1.414f);
//       size_t       c_idx = search(t, b, network, 1200); // Stronger search
//       for play
//
//       tree_peek(t);
//       size_t move_idx = t->nodes.buf[c_idx].move_idx;
//       gs              = play(b, move_idx);
//       tree_free(t);
//     }
//
//     board_print(b);
//   }
//
//   if (gs == GAME_X)
//     printf("AI (X) Wins!\n");
//   else if (gs == GAME_O)
//     printf("Human (O) Wins!\n");
//   else
//     printf("Draw!\n");
//
//   free(b);
// }

void train_batch(struct Network *n, struct MemoryBuffer *mem, int batch_size) {
  if ((int)mem->n < batch_size)
    return;

  network_zero_grad(n);

  struct Tensor state, pol, loss_p, loss_v;
  tensor_init(&state, batch_size, BOARD_SIZE, BOARD_SIZE, 2);
  tensor_init(&pol, batch_size, BOARD_SIZE, BOARD_SIZE, 1);
  tensor_init(&loss_p, batch_size, BOARD_SIZE, BOARD_SIZE, 1);
  tensor_init(&loss_v, batch_size, 1, 1, 1);

  size_t s_stride  = BOARD_SIZE * BOARD_SIZE * 2;
  size_t p_stride  = BOARD_SIZE * BOARD_SIZE;
  float *target_vs = malloc(batch_size * sizeof(float));

  for (int i = 0; i < batch_size; i++) {
    size_t         idx = rand() % mem->n;
    struct Memory *m   = &mem->buf[idx];

    memcpy(state.buf + i * s_stride, m->state, s_stride * sizeof(float));
    memcpy(pol.buf + i * p_stride, m->pol, p_stride * sizeof(float));
    target_vs[i] = (m->turn == PLAYER_X) ? m->z : -m->z;
  }

  for (int i = 0; i < 9; i++)
    n->as[i].n = batch_size;
  network_forward(n, &state, NULL);

  for (int i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < p_stride; j++) {
      size_t idx      = i * p_stride + j;
      loss_p.buf[idx] = n->as[5].buf[idx] - pol.buf[idx];
    }
    loss_v.buf[i] = 2.0f * (n->as[8].buf[i] - target_vs[i]);
  }

  network_backward(n, &state, &loss_p, &loss_v);
  network_sgd(n, 0.01f / batch_size);

  for (int i = 0; i < 9; i++)
    n->as[i].n = 1;

  free(target_vs);
  tensor_free(&loss_v);
  tensor_free(&loss_p);
  tensor_free(&pol);
  tensor_free(&state);
}

int main(int argc, char *argv[]) {
  srand(time(NULL));

  struct Network *network = network_new();

  // if (argc > 1 && strcmp(argv[1], "play") == 0) {
  //   printf("Loading weights from weights.bin...\n");
  //   network_load(network, "weights.bin");
  //   play_human_vs_ai(network);
  //   network_free(network);
  //   return 0;
  // }

  printf("Starting self-play training loop...\n");
  // network_load(network, "weights.bin");

  struct MemoryBuffer *mem = malloc(sizeof(*mem));
  if (!mem)
    return 1;
  mem->head = 0;
  mem->n    = 0;

  for (int game = 1; game <= 1000; game++) {
    size_t        game_start_idx = mem->head;
    struct Board *b              = board_new();

    enum GameState gs = play(b, xy_to_idx(0, 0));
    ;
    int move_count = 0;

    while (gs == GAME_CONT) {
      struct Tree *t = tree_new(b, 1.414f);
      enum Player  p = get_player(b);

      struct Tensor pol, state;
      tensor_init(&pol, 1, BOARD_SIZE, BOARD_SIZE, 1);
      tensor_init(&state, 1, BOARD_SIZE, BOARD_SIZE, 2);

      size_t c_idx  = search(t, b, network, 800);
      float  visits = (float)t->nodes.buf[0].visits - 1.0f;

      for (size_t i = 0; i < t->nodes.buf[0].num_c; i++) {
        struct Node *c       = &t->nodes.buf[t->nodes.buf[0].c_idxs + i];
        pol.buf[c->move_idx] = (float)c->visits / (visits + 1e-8f);
      }

      board_to_tensor(b, &state);
      mem_push(mem, &state, &pol, p, 0.0f);

      size_t move_idx = t->nodes.buf[c_idx].move_idx;

      if (move_count < 10) {
        float r   = (float)rand() / RAND_MAX;
        float sum = 0.0f;
        for (size_t i = 0; i < t->nodes.buf[0].num_c; i++) {
          struct Node *c = &t->nodes.buf[t->nodes.buf[0].c_idxs + i];
          sum += pol.buf[c->move_idx];
          if (r <= sum) {
            move_idx = c->move_idx;
            break;
          }
        }
      }

      gs = play(b, move_idx);
      move_count++;

      // if (game % 10 == 0) {
      board_print(b);
      printf("Game %d | Move %d | Player %c\n", game, move_count,
             p == PLAYER_X ? 'X' : 'O');
      tree_peek(t);
      // }

      tensor_free(&state);
      tensor_free(&pol);
      tree_free(t);
    }

    float  z = (gs == GAME_X) ? 1.0f : (gs == GAME_O) ? -1.0f : 0.0f;
    size_t moves_in_game = (mem->head >= game_start_idx)
                               ? (mem->head - game_start_idx)
                               : (MEM_BUF_MAX - game_start_idx + mem->head);

    for (size_t i = 0; i < moves_in_game; i++)
      mem->buf[(game_start_idx + i) % MEM_BUF_MAX].z = z;

    train_batch(network, mem, 64);

    if (game % 10 == 0) {
      network_save(network, "weights.bin");
    }

    free(b);
  }

  network_free(network);
  free(mem);
  return 0;
}
