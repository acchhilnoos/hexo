#ifndef TREE_H
#define TREE_H

#include "board.h"
#include "vec.h"

struct Node {
  size_t parent_idx;

  // NOTE: children are contiguous
  // + in enclosing Tree.nodes as a
  // + result of expand() behaviour;
  // + Node.c_idxs is the index in
  // + Tree.nodes of the first child
  size_t c_idxs;
  size_t num_c;

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
