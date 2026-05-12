#include "tree.h"
#include "board.h"
#include "vec.h"
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

struct size_t_vec {
  size_t *data;
  size_t  len;
  size_t  cap;
};
def_vec(Node);

struct Node {
  size_t            parent_idx;
  struct size_t_vec c_idxs;

  size_t n;
  float  v, p;

  struct {
    int x, y;
  } a;
  enum GameState s;
};

struct Tree {
  struct Node_vec nodes;
  float           c;
};

static inline float puct(struct Node *n, size_t parent_n, float c) {
  return n->n == 0
             ? FLT_MAX
             : n->v / n->n + c * n->p * sqrtf((float)parent_n) / (1 + n->n);
}

struct Tree *tree_new(float c) {
  struct Tree *t = malloc(sizeof(*t));
  if (!t)
    return NULL;

  vec_init(t->nodes, 1024);
  if (!t->nodes.data)
    return NULL;

  vec_init(vec_at(t->nodes, 0).c_idxs, 1024);

  t->c = c;

  return t;
}

struct Node *select(struct Tree *t, struct Board *b) {
  struct Node *n = &vec_at(t->nodes, 0);

  while (n->c_idxs.len != 0) {
    struct Node *max_n = NULL;
    float        max_v = -FLT_MAX;

    for (size_t i = 0; i < n->c_idxs.len; i++) {
      struct Node *cur_n = &vec_at(t->nodes, vec_at(n->c_idxs, i));
      float        v     = puct(cur_n, n->n, t->c);

      if (v > max_v) {
        max_n = cur_n;
        max_v = v;
      }
    }

    n    = max_n;
    n->s = play(b, n->a.x, n->a.y);
  }

  return n;
}

void expand(struct Tree *t, struct Board *b, size_t leaf_idx) {
  if (leaf_idx != 0)
    // TODO: check win
    return;

  for (int i = -BOARD_OFFSET; i <= BOARD_OFFSET; i++)
    for (int j = -BOARD_OFFSET; j <= BOARD_OFFSET; j++)
      if (legal(b, i, j)) {
        struct Node c = (struct Node){
            .parent_idx = leaf_idx,
            .a.x        = i,
            .a.y        = j,
            .s          = GAME_CONT,
        };
        vec_init(c.c_idxs, 1024);

        vec_push(vec_at(t->nodes, leaf_idx).c_idxs, t->nodes.len);
        vec_push(t->nodes, c);
      }
}

enum GameState simulate(struct Tree *t, struct Board *b) {
  if (b->n_empty == 0)
    return GAME_DRAW;
    // fn simulate(&self, mut leaf_board: Board) -> GameState {
    //     let leaf_player = leaf_board.next_player();
    //     let mut rng = rand::thread_rng();
    //
    //     let mut actions = leaf_board.legal();
    //     if actions.is_empty() {
    //         return GameState::Draw;
    //     }
    //
    //     for i in (1..actions.len()).rev() {
    //         let j = rng.gen_range(0..=i);
    //         actions.swap(i, j);
    //     }
    //
    //     for a in actions {
    //         let player = leaf_board.next_player();
    //         leaf_board.set(a.0, a.1, player);
    //
    //         if leaf_board.check_win(a.0, a.1, player) {
    //             return if player == leaf_player {
    //                 GameState::Win
    //             } else {
    //                 GameState::Loss
    //             };
    //         }
    //     }
    //
    //     GameState::Draw
    // }
}
