#include "./src/nn.h"
#include <time.h>

double X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
double Y[4] = {0, 1, 1, 0};

int main() {
  srand(time(NULL));
  int sizes[] = {2, 1};
  Activation acts[] = {ACT_TANH, ACT_TANH};
  MLP *mlp = new_mlp(2, sizes, acts, 2);
  double lr = 0.1;
  int start_top = get_arena_top();

  printf("Training XOR...\n");
  for (int epoch = 0; epoch < 1000; epoch++) {
    reset_arena_and_zero_grad(start_top);
    
    Value *preds[4];
    for (int i = 0; i < 4; i++) {
      Value *x[2];
      x[0] = new_value(X[i][0], "x0");
      x[1] = new_value(X[i][1], "x1");
      Value **out = mlp_forward(mlp, x);
      preds[i] = out[0];
      free(out);
    }
    
    Value *loss = mse(preds, Y, 4);
    backwardPass(loss);
    update_params(mlp, lr);

    if (epoch % 100 == 0 || epoch == 999) {
      printf("Epoch %d, Loss: %.4f\n", epoch, loss->data);
    }
  }

  printf("\nPredictions after training:\n");
  for (int i = 0; i < 4; i++) {
    Value *x[2];
    x[0] = new_value(X[i][0], "x0");
    x[1] = new_value(X[i][1], "x1");
    Value **out = mlp_forward(mlp, x);
    printf("X: [%.1f, %.1f], Y: %.1f, Pred: %.4f\n", X[i][0], X[i][1], Y[i], out[0]->data);
    free(out);
  }

  return 0;
}
