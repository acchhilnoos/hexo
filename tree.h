#ifndef TREE_H
#define TREE_H

#include "board.h"
#include <stddef.h>

struct Node;
struct Tree;

struct Tree *tree_new(float c);

struct Node *select(struct Tree *t, struct Board *b);
void expand(struct Tree *t, struct Board *b, size_t leaf_idx);
enum GameState simulate(struct Tree *t, struct Board *b);
void backpropagate(struct Tree *t);

#endif
