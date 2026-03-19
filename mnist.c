#include "./src/nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
// reading the file format of mnist is lowk magic
static int read_int(FILE *f) {
  unsigned char buf[4];
  if (fread(buf, 1, 4, f) != 4) {
    fprintf(stderr, "Failed to read int from file\n");
    exit(1);
  }
  return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}

static double *load_images(const char *path, int *out_n) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path);
    exit(1);
  }

  int magic = read_int(f);
  if (magic != 2051) {
    fprintf(stderr, "Bad image magic: %d\n", magic);
    exit(1);
  }

  int n = read_int(f);
  int rows = read_int(f);
  int cols = read_int(f);
  int pixels = rows * cols;

  double *data = (double *)malloc((size_t)n * pixels * sizeof(double));
  if (!data) {
    fprintf(stderr, "OOM loading images\n");
    exit(1);
  }

  for (int i = 0; i < n * pixels; i++) {
    unsigned char px;
    if (fread(&px, 1, 1, f) != 1) {
      fprintf(stderr, "Truncated image file\n");
      exit(1);
    }
    data[i] = px / 255.0;
  }

  fclose(f);
  *out_n = n;
  return data;
}

static int *load_labels(const char *path, int *out_n) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path);
    exit(1);
  }

  int magic = read_int(f);
  if (magic != 2049) {
    fprintf(stderr, "Bad label magic: %d\n", magic);
    exit(1);
  }

  int n = read_int(f);
  int *labels = (int *)malloc(n * sizeof(int));
  if (!labels) {
    fprintf(stderr, "OOM loading labels\n");
    exit(1);
  }

  for (int i = 0; i < n; i++) {
    unsigned char lb;
    if (fread(&lb, 1, 1, f) != 1) {
      fprintf(stderr, "Truncated label file\n");
      exit(1);
    }
    labels[i] = (int)lb;
  }

  fclose(f);
  *out_n = n;
  return labels;
}

static int argmax(Value **preds, int n) {
  int best = 0;
  for (int i = 1; i < n; i++) {
    if (preds[i]->data > preds[best]->data)
      best = i;
  }
  return best;
}

static double evaluate(MLP *mlp, double *images, int *labels, int n) {
  int correct = 0;
  int start = get_arena_top();

  for (int i = 0; i < n; i++) {
    reset_arena_and_zero_grad(start);

    Value *x[784];
    double *img = images + i * 784;
    for (int j = 0; j < 784; j++)
      x[j] = new_value(img[j], "x");

    Value **out = mlp_forward(mlp, x);
    if (argmax(out, 10) == labels[i])
      correct++;
    free(out);
  }

  return (double)correct / n;
}

int main(int argc, char **argv) {
  const char *train_img_path =
      argc > 1 ? argv[1] : "mnist/train-images-idx3-ubyte";
  const char *train_lbl_path =
      argc > 2 ? argv[2] : "mnist/train-labels-idx1-ubyte";
  const char *test_img_path =
      argc > 3 ? argv[3] : "mnist/t10k-images-idx3-ubyte";
  const char *test_lbl_path =
      argc > 4 ? argv[4] : "mnist/t10k-labels-idx1-ubyte";

  int n_train, n_train_lbl, n_test, n_test_lbl;
  double *train_images = load_images(train_img_path, &n_train);
  int *train_labels = load_labels(train_lbl_path, &n_train_lbl);
  double *test_images = load_images(test_img_path, &n_test);
  int *test_labels = load_labels(test_lbl_path, &n_test_lbl);

  printf("Train: %d  Test: %d\n", n_train, n_test);

  srand(time(NULL));
  int sizes[] = {32, 10};
  Activation acts[] = {ACT_TANH, ACT_SOFTMAX};
  MLP *mlp = new_mlp(784, sizes, acts, 2);

  double y_onehot[10];

  double lr = 0.01;
  int epochs = 10;
  int log_every = 10000;
  int eval_samples = 10000;
  int train_start = get_arena_top();

  printf("\nTraining (784->%d->10, SGD, lr=%.4f)...\n\n", sizes[0], lr);

  for (int epoch = 0; epoch < epochs; epoch++) {
    double epoch_loss = 0.0;

    for (int i = 0; i < n_train; i++) {
      reset_arena_and_zero_grad(train_start);

      Value *x[784];
      double *img = train_images + i * 784;
      for (int j = 0; j < 784; j++)
        x[j] = new_value(img[j], "x");

      Value **out = mlp_forward(mlp, x);
      memset(y_onehot, 0, sizeof(y_onehot));
      y_onehot[train_labels[i]] = 1.0;

      Value *loss = cross_entropy(out, y_onehot, 10);
      free(out);

      backwardPass(loss);
      update_params(mlp, lr);

      epoch_loss += loss->data;

      if ((i + 1) % log_every == 0) {
        printf("  Epoch %d  [%5d/%d]  avg loss: %.4f\n", epoch + 1, i + 1,
               n_train, epoch_loss / (i + 1));
        fflush(stdout);
      }
    }

    double acc = evaluate(mlp, test_images, test_labels, eval_samples);
    printf("Epoch %d done | avg loss: %.4f | test acc (%d samples): %.2f%%\n\n",
           epoch + 1, epoch_loss / n_train, eval_samples, acc * 100.0);
  }

  printf("Final test accuracy (full 10k): ");
  fflush(stdout);
  double final_acc = evaluate(mlp, test_images, test_labels, n_test);
  printf("%.2f%%\n", final_acc * 100.0);

  free(train_images);
  free(train_labels);
  free(test_images);
  free(test_labels);
  free_arena();
  return 0;
}
