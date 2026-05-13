#include "tree.h"
#include "board.h"
#include "vec.h"
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static inline float puct(struct Node *c, struct Node *p, float tree_c) {
  if (c->visits == 0)
    return FLT_MAX;

  float q = c->val / c->visits;
  q       = p->turn == PLAYER_O ? -q : q;

  return q + c->pol * sqrtf((float)p->visits) / (1 + c->visits);
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

  for (int i = 0; i < b->n_empty; i++) {
    struct Node c = (struct Node){.parent_idx = leaf_idx,
                                  .pol        = 1.0f,
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

size_t search(struct Tree *t, struct Board *b, size_t it) {
  struct Board *b_copy = malloc(sizeof(*b_copy));

  for (size_t i = 0; i < it; i++) {
    board_copy(b, b_copy);

    size_t       leaf_idx = select(t, b_copy);
    struct Node *leaf     = &vec_at(t->nodes, leaf_idx);

    float v = value(leaf->state);
    if (v == -FLT_MAX) {
      expand(t, b_copy, leaf_idx);

      leaf           = &vec_at(t->nodes, leaf_idx);
      leaf_idx       = leaf->c_idxs + (rand() % leaf->num_c);
      struct Node *c = &vec_at(t->nodes, leaf_idx);

      play(b_copy, c->move_idx);
      v = value(simulate(t, b_copy));
    }

    backpropagate(t, leaf_idx, v);
  }

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
