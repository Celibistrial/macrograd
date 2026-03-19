#ifndef macrograd_h
#define macrograd_h
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CHILDREN 2
#define ARENA_PAGE_SIZE 65536
#define MAX_PAGES 1024

typedef struct Value Value;
typedef void (*BackwardFn)(Value *self);

struct Value {
  double data;
  double grad;
  Value *children[MAX_CHILDREN];
  int n_children;
  char op[8];
  BackwardFn backward;
  int visited;
};

typedef struct {
  Value *pages[MAX_PAGES];
  int num_pages;
  int top;
} Arena;

static Arena g_arena = {{NULL}, 0, 0};

static inline void arena_init() {
  if (g_arena.num_pages > 0)
    return;
  g_arena.pages[0] = (Value *)malloc(ARENA_PAGE_SIZE * sizeof(Value));
  if (!g_arena.pages[0]) {
    fprintf(stderr, "Arena init failed\n");
    exit(1);
  }
  g_arena.num_pages = 1;
  g_arena.top = 0;
}

static inline void arena_grow() {
  if (g_arena.num_pages >= MAX_PAGES) {
    fprintf(stderr, "Arena overflow\n");
    exit(1);
  }
  g_arena.pages[g_arena.num_pages] =
      (Value *)malloc(ARENA_PAGE_SIZE * sizeof(Value));
  if (!g_arena.pages[g_arena.num_pages]) {
    fprintf(stderr, "Arena grow failed\n");
    exit(1);
  }
  g_arena.num_pages++;
}

static inline int get_arena_top() { return g_arena.top; }

static inline void reset_arena_and_zero_grad(int mark) {
  g_arena.top = mark;
  for (int i = 0; i < mark; i++) {
    int p = i / ARENA_PAGE_SIZE;
    int o = i % ARENA_PAGE_SIZE;
    g_arena.pages[p][o].grad = 0.0;
  }
}

static inline void free_arena() {
  for (int i = 0; i < g_arena.num_pages; i++) {
    free(g_arena.pages[i]);
    g_arena.pages[i] = NULL;
  }
  g_arena.num_pages = 0;
  g_arena.top = 0;
}

static inline Value *new_value(double data, const char *op) {
  if (g_arena.num_pages == 0)
    arena_init();
  if (g_arena.top >= g_arena.num_pages * ARENA_PAGE_SIZE)
    arena_grow();

  int p = g_arena.top / ARENA_PAGE_SIZE;
  int o = g_arena.top % ARENA_PAGE_SIZE;
  Value *v = &g_arena.pages[p][o];
  g_arena.top++;

  v->data = data;
  v->grad = 0.0;
  v->n_children = 0;
  v->backward = NULL;
  v->visited = 0;
  snprintf(v->op, sizeof(v->op), "%s", op);
  return v;
}

static void add_backward(Value *self) {
  self->children[0]->grad += self->grad;
  self->children[1]->grad += self->grad;
}

static inline Value *val_add(Value *a, Value *b) {
  Value *out = new_value(a->data + b->data, "+");
  out->children[0] = a;
  out->children[1] = b;
  out->n_children = 2;
  out->backward = add_backward;
  return out;
}

static void mul_backward(Value *self) {
  self->children[0]->grad += self->children[1]->data * self->grad;
  self->children[1]->grad += self->children[0]->data * self->grad;
}

static inline Value *val_mul(Value *a, Value *b) {
  Value *out = new_value(a->data * b->data, "*");
  out->children[0] = a;
  out->children[1] = b;
  out->n_children = 2;
  out->backward = mul_backward;
  return out;
}

static void tanh_backward(Value *self) {
  double t = self->data;
  self->children[0]->grad += (1.0 - t * t) * self->grad;
}

static inline Value *val_tanh(Value *a) {
  Value *out = new_value(tanh(a->data), "tanh");
  out->children[0] = a;
  out->n_children = 1;
  out->backward = tanh_backward;
  return out;
}

static void relu_backward(Value *self) {
  self->children[0]->grad += ((self->data > 0) ? 1.0 : 0.0) * self->grad;
}

static inline Value *val_relu(Value *a) {
  Value *out = new_value((a->data > 0) ? a->data : 0.0, "relu");
  out->children[0] = a;
  out->n_children = 1;
  out->backward = relu_backward;
  return out;
}
static void log_backward(Value *self) {
  Value *a = self->children[0];
  a->grad += (1.0 / a->data) * self->grad;
}

static inline Value *val_log(Value *a) {
  Value *out = new_value(log(a->data), "log");
  out->children[0] = a;
  out->n_children = 1;
  out->backward = log_backward;
  return out;
}

static void exp_backward(Value *self) {
  Value *a = self->children[0];
  a->grad += self->data * self->grad;
}

static inline Value *val_exp(Value *a) {
  Value *out = new_value(exp(a->data), "exp");
  out->children[0] = a;
  out->n_children = 1;
  out->backward = exp_backward;
  return out;
}

static inline void backwardPass(Value *root) {
  if (!root)
    return;
  root->grad = 1.0;
  for (int i = g_arena.top - 1; i >= 0; i--) {
    int p = i / ARENA_PAGE_SIZE;
    int o = i % ARENA_PAGE_SIZE;
    Value *v = &g_arena.pages[p][o];
    if (v->backward)
      v->backward(v);
  }
}

#endif
