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

void play_human_vs_ai(struct Network *network) {
  struct Board *b = board_new();
  play(b, xy_to_idx(0, 0));
  board_print(b);

  enum GameState gs = GAME_CONT;
  while (gs == GAME_CONT) {
    enum Player p = get_player(b);

    if (p == PLAYER_O) {
      int x, y;
      printf("Your turn (x y): ");
      if (scanf("%d %d", &x, &y) != 2)
        break;

      size_t idx = xy_to_idx(x, y);
      if (!legal(b, idx)) {
        printf("Illegal move!\n");
        continue;
      }
      gs = play(b, idx);
    } else {
      printf("AI is thinking...\n");
      struct Tree *t     = tree_new(b, 1.414f);
      size_t       c_idx = search(t, b, network, 800);

      tree_peek(t);
      size_t move_idx = t->nodes.buf[c_idx].move_idx;
      gs              = play(b, move_idx);
      tree_free(t);
    }

    board_print(b);
  }

  if (gs == GAME_X)
    printf("AI (X) Wins!\n");
  else if (gs == GAME_O)
    printf("Human (O) Wins!\n");
  else
    printf("Draw!\n");

  free(b);
}

void train_batch(struct Network *n, struct MemoryBuffer *mem, int batch_size) {
  if (mem->n < batch_size)
    return;

  network_zero_grad(n);

  struct Tensor state, pol, loss_p, loss_v;
  tensor_init(&state, 1, 2, BOARD_SIZE, BOARD_SIZE);
  tensor_init(&pol, 1, 1, BOARD_SIZE, BOARD_SIZE);
  tensor_init(&loss_p, 1, 1, BOARD_SIZE, BOARD_SIZE);
  tensor_init(&loss_v, 1, 1, 1, 1);

  for (size_t i = 0; i < batch_size; i++) {
    size_t         idx = rand() % mem->n;
    struct Memory *m   = &mem->buf[idx];

    memcpy(state.buf, m->state, sizeof(m->state));
    memcpy(pol.buf, m->pol, sizeof(m->pol));

    network_forward(n, &state, NULL);

    float target_v = (m->turn == PLAYER_X) ? m->z : -m->z;

    for (size_t j = 0; j < BOARD_SIZE * BOARD_SIZE; j++)
      loss_p.buf[j] = n->as[5].buf[j] - pol.buf[j];
    loss_v.buf[0] = 2.0f * (n->as[8].buf[0] - target_v);

    network_backward(n, &state, &loss_p, &loss_v);
  }

  network_sgd(n, 0.01f / batch_size);

  tensor_free(&loss_v);
  tensor_free(&loss_p);
  tensor_free(&pol);
  tensor_free(&state);
}

int main(int argc, char *argv[]) {
  srand(time(NULL));

  struct Network *network = network_new();

  // if (argc > 1 && strcmp(argv[1], "play") == 0) {
  //   network_load(network, "weights.bin");
  //   play_human_vs_ai(network);
  //   network_free(network);
  //   return 0;
  // }

  printf("Starting self-play training loop...\n");
  // printf("Loading weights...\n");
  // network_load(network, "weights.bin");

  struct MemoryBuffer *mem = malloc(sizeof(*mem));
  mem->head                = 0;
  mem->n                   = 0;

  for (int game = 0; game < 1000; game++) {
    size_t game_start_idx = mem->head;

    struct Board  *b  = board_new();
    enum GameState gs = play(b, xy_to_idx(0, 0));

    while (gs == GAME_CONT) {
      struct Tree *t    = tree_new(b, 1.414f);
      struct Node *root = &vec_at(t->nodes, 0);
      enum Player  p    = get_player(b);

      struct Tensor pol, state;
      tensor_init(&pol, 1, 1, BOARD_SIZE, BOARD_SIZE);
      tensor_init(&state, 1, 2, BOARD_SIZE, BOARD_SIZE);

      size_t c_idx  = search(t, b, network, 1000);
      float  visits = (float)root->visits - 1.0f;

      for (size_t i = 0; i < root->num_c; i++) {
        struct Node *c       = &vec_at(t->nodes, root->c_idxs + i);
        pol.buf[c->move_idx] = (float)c->visits / visits;
      }

      board_to_tensor(b, &state);
      mem_push(mem, &state, &pol, p, 0.0f);

      size_t move_idx = vec_at(t->nodes, c_idx).move_idx;

      // if (BOARD_SIZE * BOARD_SIZE - b->n_empty < 15) {
      //   float r   = (float)rand() / RAND_MAX;
      //   float sum = 0.0f;
      //   move_idx  = vec_at(t->nodes, root->c_idxs).move_idx;
      //
      //   for (size_t i = 0; i < root->num_c; i++) {
      //     struct Node *c = &vec_at(t->nodes, root->c_idxs + i);
      //     sum += pol.buf[c->move_idx];
      //     if (r <= sum) {
      //       move_idx = c->move_idx;
      //       break;
      //     }
      //   }
      // }

      gs = play(b, move_idx);

      board_print(b);
      printf("Player %c:\n", p == PLAYER_X ? 'X' : 'O');
      tree_peek(t);

      tensor_free(&state);
      tensor_free(&pol);
      tree_free(t);
    }

    float z = gs == GAME_X ? 1.0f : gs == GAME_O ? -1.0f : 0.0f;

    size_t moves_played = (mem->head >= game_start_idx)
                              ? (mem->head - game_start_idx)
                              : (MEM_BUF_MAX - game_start_idx + mem->head);

    for (size_t i = 0; i < moves_played; i++)
      mem->buf[(game_start_idx + i) % MEM_BUF_MAX].z = z;

    train_batch(network, mem, 64);

    if (game % 10 == 0)
      network_save(network, "weights.bin");

    free(b);
  }

  printf("Saving weights...\n");
  network_save(network, "weights.bin");

  free(mem);
  network_free(network);
  return 0;
}
