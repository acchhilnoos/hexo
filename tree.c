#include "tree.h"
#include "board.h"
#include "vec.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

static inline float puct(struct Node *c, struct Node *p, float tree_c) {
  float q = c->visits > 0   ? c->val / c->visits
            : p->visits > 0 ? p->val / p->visits
                            : 0.0f;
  q       = p->turn == PLAYER_O ? -q : q;

  return q + tree_c * c->pol * sqrtf((float)p->visits) / (1 + c->visits);
}

struct Tree *tree_new(struct Board *b, float c) {
  struct Tree *t = malloc(sizeof(*t));
  if (!t)
    return NULL;

  vec_init(t->nodes, BOARD_SIZE * BOARD_SIZE);
  if (!t->nodes.buf)
    return NULL;

  struct Node root =
      (struct Node){.pol = 1.0f, .turn = get_player(b), .state = GAME_CONT};
  vec_push(t->nodes, root);

  t->c = c;

  return t;
}

void tree_free(struct Tree *t) {
  vec_free(t->nodes);
  free(t);
}

static inline float value(enum GameState s) {
  switch (s) {
  case GAME_X:
    return 1.0f;
  case GAME_O:
    return -1.0f;
  case GAME_DRAW:
    return 0.0f;
  case GAME_CONT:
    return -FLT_MAX;
  }
}

void board_to_tensor(struct Board *b, struct Tensor *t) {
  tensor_zero(t);
  enum Player turn = get_player(b);

  for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
    if (b->cells[i] == CELL_EMPTY)
      continue;

    if (b->cells[i] == (enum CellState)turn)
      t->buf[i] = 1.0f;
    else
      t->buf[BOARD_SIZE * BOARD_SIZE + i] = 1.0f;
  }
}

size_t select(struct Tree *t, struct Board *b) {
  struct Node *cur_n = &vec_at(t->nodes, 0);
  size_t       max_i = cur_n->c_idxs;

  while (cur_n->num_c != 0) {
    struct Node *max_c = &vec_at(t->nodes, cur_n->c_idxs);
    float        max_v = -FLT_MAX;

    for (size_t i = 0; i < cur_n->num_c; i++) {
      struct Node *cur_c = &vec_at(t->nodes, cur_n->c_idxs + i);
      float        v     = puct(cur_c, cur_n, t->c);

      if (v > max_v) {
        max_i = cur_n->c_idxs + i;
        max_c = cur_c;
        max_v = v;
      }
    }

    cur_n        = max_c;
    cur_n->state = play(b, cur_n->move_idx);
  }

  return max_i;
}

void expand(struct Tree *t, struct Board *b, size_t leaf_idx) {
  b->n_empty--;
  enum Player c_turn = get_player(b);
  b->n_empty++;

  vec_at(t->nodes, leaf_idx).c_idxs = t->nodes.len;
  vec_at(t->nodes, leaf_idx).num_c  = b->n_empty;

  for (size_t i = 0; i < b->n_empty; i++) {
    struct Node c = (struct Node){.parent_idx = leaf_idx,
                                  .pol        = 0.1f,
                                  .move_idx   = b->empty[i],
                                  .turn       = c_turn,
                                  .state      = GAME_CONT};
    vec_push(t->nodes, c);
  }
}

enum GameState simulate(struct Tree *t, struct Board *b) {
  while (b->n_empty > 0) {
    size_t idx = b->empty[rand() % b->n_empty];

    enum GameState s = play(b, idx);
    if (s != GAME_CONT)
      return s;
  }

  return GAME_DRAW;
}

void backpropagate(struct Tree *t, size_t c_idx, float v) {
  while (1) {
    struct Node *cur_c = &vec_at(t->nodes, c_idx);
    cur_c->visits++;
    cur_c->val += v;

    if (c_idx != 0)
      c_idx = cur_c->parent_idx;
    else
      break;
  }
}

size_t search(struct Tree *t, struct Board *b, struct Network *n, size_t it) {
  struct Board *b_copy = malloc(sizeof(*b_copy));
  struct Tensor inputs;
  tensor_init(&inputs, 1, 2, BOARD_SIZE, BOARD_SIZE);

  for (size_t i = 0; i < it; i++) {
    board_copy(b, b_copy);

    size_t       leaf_idx = select(t, b_copy);
    struct Node *leaf     = &vec_at(t->nodes, leaf_idx);

    float v = value(leaf->state);
    if (v == -FLT_MAX) {
      expand(t, b_copy, leaf_idx);

      board_to_tensor(b_copy, &inputs);
      network_forward(n, &inputs, b_copy);

      leaf = &vec_at(t->nodes, leaf_idx);
      for (size_t c_idx = 0; c_idx < leaf->num_c; c_idx++) {
        struct Node *c = &vec_at(t->nodes, leaf->c_idxs + c_idx);
        c->pol         = n->as[5].buf[c->move_idx];
      }

      if (leaf_idx == 0) {
        float noise[BOARD_SIZE * BOARD_SIZE];
        float sum   = 0.0f;
        float alpha = 0.03f;

        for (size_t i = 0; i < leaf->num_c; i++) {
          float u  = (float)rand() / RAND_MAX + 1e-8f;
          noise[i] = powf(u, 1.0f / alpha);
          sum += noise[i];
        }

        for (size_t i = 0; i < leaf->num_c; i++) {
          struct Node *c = &vec_at(t->nodes, leaf->c_idxs + i);
          c->pol         = 0.75f * c->pol + 0.25f * noise[i] / sum;
        }
      }

      v = n->as[8].buf[0];
      if (leaf->turn == PLAYER_O)
        v = -v;

      // leaf           = &vec_at(t->nodes, leaf_idx);
      // leaf_idx       = leaf->c_idxs + (rand() % leaf->num_c);
      // struct Node *c = &vec_at(t->nodes, leaf_idx);
      //
      // play(b_copy, c->move_idx);
      // v = value(simulate(t, b_copy));
    }

    backpropagate(t, leaf_idx, v);
  }

  tensor_free(&inputs);
  free(b_copy);

  struct Node *root       = &vec_at(t->nodes, 0);
  size_t       max_i      = root->c_idxs;
  size_t       max_visits = 0;
  for (size_t i = 0; i < root->num_c; i++) {
    size_t       c_idx = root->c_idxs + i;
    struct Node *c     = &vec_at(t->nodes, c_idx);

    if (c->visits > max_visits) {
      max_i      = c_idx;
      max_visits = c->visits;
    }
  }
  return max_i;
}

void tree_peek(struct Tree *t) {
  struct Node *root = &vec_at(t->nodes, 0);

  size_t k = root->num_c < 5 ? root->num_c : 5;

  size_t *top_idxs = malloc(root->num_c * sizeof(*top_idxs));
  for (size_t i = 0; i < root->num_c; i++)
    top_idxs[i] = root->c_idxs + i;

  for (size_t i = 0; i < k; i++) {
    size_t max_i = i;
    for (size_t j = i + 1; j < root->num_c; j++) {
      if (vec_at(t->nodes, top_idxs[j]).visits >
          vec_at(t->nodes, top_idxs[max_i]).visits)
        max_i = j;
    }
    size_t t        = top_idxs[i];
    top_idxs[i]     = top_idxs[max_i];
    top_idxs[max_i] = t;
  }

  for (size_t i = 0; i < k; i++) {
    struct Node *c = &vec_at(t->nodes, top_idxs[i]);
    int          x = idx_to_x(c->move_idx);
    int          y = idx_to_y(c->move_idx);
    float        q = c->visits > 0 ? c->val / c->visits : 0.0f;
    q              = root->turn == PLAYER_O ? -q : q;

    printf("%zu. move (%3d, %3d) | Visits: %6zu | Q: %6.3f | P: %.3f\n", i + 1,
           x, y, c->visits, q, c->pol);
  }
  free(top_idxs);
}
