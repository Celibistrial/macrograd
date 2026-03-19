# macrograd
<img src="./resources/cool-image.jpg" alt="Cool-Image" width="800">

A minimal autograd engine implemented in pure C, designed around a contiguous memory model. Builds scalar computation graphs and performs gradient propagation efficiently, with a lightweight neural network layer on top. Distributed as a single-header library.

Originally motivated by Andrej Karpathy’s *micrograd*, but redesigned in C with an emphasis on memory scalability.


## Overview

Deep learning libraries abstract away gradient computation. This project reconstructs that mechanism at the most granular level: each scalar value participates as a node in a computation graph, and gradients are propagated using reverse-mode differentiation.

Unlike higher-level implementations, this system exposes the mechanics directly—there are no hidden abstractions, only explicit data flow and memory operations.

## Capabilities

* Single-header design (`macrograd.h`, `nn.h`)
* Reverse-mode autodiff at scalar granularity
* Elementary operations: addition, multiplication, `exp`, `log`, `tanh`, `relu`
* Neural network primitives: neurons, dense layers, MLPs
* Loss functions: mean squared error, cross-entropy
* Softmax output layer
* MNIST-scale training support


## Core Design

### Scalar Nodes (`Value`)

Each numerical quantity is represented as a heap-allocated node storing:

* its scalar data
* accumulated gradient
* references to parent nodes
* a backward function

Operations construct new nodes while encoding how gradients should flow during the backward phase.

```c
Value *x = new_value(2.0, "x");
Value *y = new_value(3.0, "y");
Value *z = val_mul(x, y);  // z = x * y

backwardPass(z);           // dz/dx = 3, dz/dy = 2
```


### Memory Strategy: Arena Allocation

Managing graph lifetime is the primary difficulty when implementing autodiff in C. Instead of reference counting or garbage collection, this project uses a **linear arena allocator**.

All nodes are stored sequentially in a contiguous block of memory.

Key consequences:

* **Constant-time reset**: clearing intermediate graph state requires only resetting a pointer
* **Implicit ordering**: node creation order naturally encodes a valid traversal order
* **Cache efficiency**: memory locality improves traversal performance

```c
int checkpoint = get_arena_top();

for (int i = 0; i < steps; i++) {
    reset_arena_and_zero_grad(checkpoint);

    Value *loss = compute_loss(model, input, target);

    backwardPass(loss);
    update_params(model, lr);
}
```


### Gradient Propagation

Backward execution does not require explicit graph sorting. Since nodes are appended in dependency order, reverse iteration over the arena suffices.

Each node contributes to its parents via its stored backward function, making the entire process a single linear pass.



## Example: XOR

A minimal nonlinear learning task demonstrating representational capacity:

```c
int sizes[] = {2, 1};
Activation acts[] = {ACT_TANH, ACT_TANH};

MLP *model = new_mlp(2, sizes, acts, 2);

Value **pred = mlp_forward(model, x);
Value *loss = mse(pred, y, 4);

backwardPass(loss);
update_params(model, 0.1);
```

Compile and run:

```bash
gcc xor.c -lm -o xor && ./xor
```


## MNIST Training

The implementation scales to moderately sized networks (e.g., 784 → 32 → 10) and achieves reasonable accuracy using standard SGD.

```bash
gcc mnist.c -O3 -lm -o mnist && ./mnist
```


## Repository Layout

```
macrograd/
├── src/
│   ├── macrograd.h   # autodiff engine and allocator
│   └── nn.h          # neural network components
├── xor.c             # XOR training example
├── mnist.c           # MNIST training pipeline
└── mnist/            # dataset files
```


## Notes and Observations

### Arena allocation as a natural fit

Autodiff graphs have a predictable lifecycle: build → use → discard. A linear allocator aligns perfectly with this pattern, avoiding the overhead of general-purpose memory management.

### Stability considerations

Direct implementations of functions like softmax can easily overflow or underflow. Techniques such as log-sum-exp and epsilon offsets are necessary to maintain numerical reliability.

### Low-level perspective

Implementing neural networks in C highlights their underlying simplicity. Training reduces to repeated evaluation of expressions and mutation of memory locations—no implicit framework behavior, only explicit computation.
