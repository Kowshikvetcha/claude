---
title: TensorFlow and Keras
type: entity
domain: deep-learning
roles: [ai-engineer, ml-engineer]
difficulty: core
frequency: medium
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# TensorFlow and Keras

## What it is
TensorFlow is Google's deep-learning framework; Keras is its high-level API (and, since TF2, the default way most people write TensorFlow models). Less common than PyTorch in new research and LLM work, but still present in production stacks built before ~2021, in TFX-based pipelines, and where mobile/edge deployment (TFLite) or TF Serving is already the standard.

## Core concepts
- **Keras model-building styles**: `Sequential` (a linear stack of layers, simplest), the **Functional API** (a DAG of layers — supports multi-input/output and shared layers), and **subclassing** `tf.keras.Model` (full imperative control, closest to PyTorch's `nn.Module`).
- **`compile` / `fit`**: Keras separates model definition from training config — `model.compile(optimizer, loss, metrics)` then `model.fit(data, epochs, callbacks)` runs the loop for you, versus PyTorch's manual loop.
- **`tf.data.Dataset`**: the input pipeline abstraction — chain `.map()`, `.batch()`, `.shuffle()`, `.prefetch()` for an efficient, prefetching data pipeline that overlaps I/O with compute.
- **Graph mode via `tf.function`**: decorating a Python function traces it into a static graph for speed and portability — TF2's eager-by-default execution can still be compiled this way, similar in spirit to `torch.compile`.
- **Callbacks**: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard` — injected into `.fit()` rather than written inline as in a manual loop.
- **SavedModel format**: TensorFlow's serialization format bundles the graph and weights together, which is what TF Serving and TFLite consume directly — a smoother path to production serving than PyTorch's historically fragmented options (though TorchScript/`torch.export` close that gap now).

## Code
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
]

model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)
model.save("saved_model_dir")   # SavedModel format for TF Serving
```

## When to use it vs alternatives
- **vs PyTorch**: choose TensorFlow/Keras when you're maintaining an existing TF codebase, need TFLite for on-device/mobile inference, or want `fit`/`compile` to remove training-loop boilerplate for standard architectures; choose PyTorch for research flexibility, custom training loops, and the current LLM/HF ecosystem — see [[pytorch]].
- **vs JAX**: JAX is faster-moving for research needing custom parallelism/transformations; TensorFlow's tooling (TFX, TF Serving, TensorBoard) is more mature for a stable production pipeline.

## Interview angle
**Q. When would you drop down from `model.fit()` to a custom training loop in Keras?**
When training needs logic `fit` doesn't expose cleanly — custom loss combinations that depend on intermediate outputs, GANs (alternating generator/discriminator updates), reinforcement learning, or per-step custom gradient manipulation. Keras supports this via subclassing `train_step` or writing the loop with `tf.GradientTape` directly.

**Q. What does `tf.function` actually buy you, and what can break inside it?**
It traces Python code into a static `tf.Graph`, removing Python-interpreter overhead per call and enabling graph-level optimizations (fusion, device placement) — a real speed-up for repeated calls. Python-level side effects (print statements, list appends, non-tensor control flow depending on Python values rather than tensor values) behave differently or only execute during tracing, which is a common source of subtle bugs.

**Q. Why might a team pick TensorFlow specifically for a mobile deployment?**
TFLite is a mature, first-class conversion and runtime path for on-device inference (quantization, hardware delegates for mobile GPUs/NPUs) that PyTorch's mobile story has historically lagged, though PyTorch Mobile/ExecuTorch has narrowed this gap.

## Traps
- Mixing `tf.Tensor` and NumPy operations carelessly inside a `tf.function` — causes retracing on every call (a major hidden performance cliff) if shapes/dtypes vary.
- Assuming `model.fit()` handles everything — for non-standard training regimes it silently produces the wrong thing unless you override `train_step`.
- Forgetting `.prefetch(tf.data.AUTOTUNE)` on the input pipeline, leaving the GPU idle waiting on data loading.
- Comparing TF1-era graph-mode habits (explicit sessions, placeholders) to TF2 — TF2 is eager by default and that mental model no longer applies.

## Related
[[pytorch]], [[neural-network-fundamentals]], [[model-packaging-and-containers]], [[model-serving-patterns]]
