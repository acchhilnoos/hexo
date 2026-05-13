#ifndef TREE_H
#define TREE_H

#include "board.h"
#include "vec.h"

struct size_t_vec {
  size_t *buf;
  size_t len;
  size_t cap;
};

struct Node {
  size_t parent_idx;
  // TODO: children are contiguous in
  //   the enclosing tree->nodes due to
  //   expand, replace the following with
  // size_t c_idxs;
  // size_t num_c;
  struct size_t_vec c_idxs;

  size_t visits;
  float val, pol;

  size_t move_idx;
  enum Player turn;
  enum GameState state;
};
def_vec(Node);

struct Tree {
  struct Node_vec nodes;
  float c;
};

struct Tree *tree_new(struct Board *b, float c);
void tree_free(struct Tree *t);

size_t select(struct Tree *t, struct Board *b);
void expand(struct Tree *t, struct Board *b, size_t leaf_idx);
enum GameState simulate(struct Tree *t, struct Board *b);
void backpropagate(struct Tree *t, size_t child_idx, float v);
size_t search(struct Tree *t, struct Board *b, size_t it);

#endif
