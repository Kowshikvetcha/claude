---
title: PyTorch
type: entity
domain: deep-learning
roles: [ai-engineer, ml-engineer]
difficulty: core
frequency: high
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# PyTorch

## What it is
The dominant deep-learning framework in research and, increasingly, production — dynamic computation graphs, eager execution, and a Pythonic API for building and training neural networks on GPUs. Almost every modern LLM/vision paper ships PyTorch code; it's the de facto language of deep learning interviews.

## Core concepts
- **Tensors + autograd**: a `torch.Tensor` with `requires_grad=True` records operations in a dynamic graph; `.backward()` walks it in reverse to populate `.grad` on every leaf tensor. This is what makes PyTorch "define-by-run" — the graph is rebuilt every forward pass, so control flow (loops, conditionals) can depend on tensor values.
- **`nn.Module`**: the unit of composition — define `__init__` (layers as attributes) and `forward` (the computation); parameters are auto-registered and discoverable via `.parameters()`.
- **Optimizer + zero_grad**: gradients accumulate by default, so every training step calls `optimizer.zero_grad()` before `loss.backward()`, then `optimizer.step()` to apply the update.
- **`Dataset` / `DataLoader`**: `Dataset` defines `__getitem__`/`__len__`; `DataLoader` handles batching, shuffling, and multiprocess prefetching (`num_workers`).
- **`.train()` / `.eval()` mode**: toggles layers with different train/inference behaviour — dropout turns off, BatchNorm switches from batch statistics to running statistics. Forgetting this is the single most common inference bug.
- **`torch.no_grad()` / `torch.inference_mode()`**: disables autograd tracking during inference/evaluation, cutting memory and compute — without it, activations are needlessly retained for a backward pass that never happens.
- **Device placement**: tensors and modules must explicitly move to the same device (`.to("cuda")`); a CPU/GPU mismatch is a runtime error, not a silent fallback.
- **Mixed precision & `torch.compile`**: `torch.cuda.amp.autocast` + `GradScaler` for fp16/bf16 training; `torch.compile` JIT-compiles the graph for a speed-up with minimal code change.

## Code
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(20, 64, 2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

model.train()
for x, y in train_loader:            # DataLoader over a Dataset
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    val_logits = model(val_x.to(device))
```

## When to use it vs alternatives
- **vs TensorFlow/Keras**: PyTorch's eager, Pythonic execution is easier to debug and dominates research and most new production model-serving code; TensorFlow/Keras still shows up in older production stacks and TFX pipelines, and has a slight edge in mobile/edge deployment tooling (TFLite) — see [[tensorflow-and-keras]].
- **vs JAX**: JAX's functional, `grad`/`jit`/`vmap` style suits large-scale research needing composable transformations (e.g. custom parallelism), at the cost of a steeper learning curve; PyTorch is more approachable for typical model-building work.
- **vs scikit-learn**: wrong tool for classical tabular ML — PyTorch is for models where you need custom architectures, autograd, and GPU tensors.

## Interview angle
**Q. Why call `optimizer.zero_grad()` before `.backward()` — what happens if you forget?**
Gradients accumulate into `.grad` by default (useful for gradient accumulation across micro-batches); without zeroing, each step's gradient is added to the previous step's, corrupting the update direction.

**Q. Explain `model.eval()` — what specifically changes?**
It flips the behaviour of modules that differ between training and inference: Dropout stops zeroing activations, and BatchNorm uses its running mean/variance instead of the current batch's statistics. It does **not** disable gradient tracking — that's `torch.no_grad()`, a separate, often-paired call.

**Q. How does PyTorch's autograd actually compute gradients — what is the graph made of?**
Every differentiable op on a tensor with `requires_grad=True` creates a node recording its inputs and the local Jacobian-vector-product function needed for the chain rule. `.backward()` traverses this graph in reverse topological order (reverse-mode autodiff), accumulating gradients into each leaf tensor's `.grad`. The graph is discarded after the backward pass unless `retain_graph=True`.

## Traps
- Forgetting `model.eval()` before inference — BatchNorm statistics silently drift, producing inconsistent predictions between runs.
- Leaving `requires_grad=True` tensors around during evaluation without `no_grad()` — memory balloons from retained activation graphs.
- Mismatched devices (`model` on GPU, batch on CPU) — throws at the first forward pass, not at load time.
- Treating `loss.item()` calls inside the training loop as free — pulling a scalar off the GPU forces a sync point; do it sparingly (e.g. every N steps) in performance-sensitive loops.

## Related
[[backpropagation]], [[mixed-precision-and-memory]], [[distributed-training]], [[tensorflow-and-keras]], [[huggingface-ecosystem]]
