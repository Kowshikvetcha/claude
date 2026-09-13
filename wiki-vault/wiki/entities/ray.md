---
title: Ray
type: entity
domain: mlops
roles: [ml-engineer, mlops-engineer, ai-engineer]
difficulty: intermediate
frequency: medium
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# Ray

## What it is
A distributed computing framework for Python that scales arbitrary Python code (not just SQL-style DataFrame ops) across a cluster — its ML-specific libraries (Train, Serve, Tune, Data) make it the common backbone for distributed training, hyperparameter search, and model serving in modern Python ML stacks, especially for deep learning workloads that Spark isn't well-suited to.

## Core concepts
- **Tasks and Actors**: `@ray.remote` turns a plain function into a **task** (stateless, runs once, returns a value) or a class into an **Actor** (stateful, persists across calls, holds e.g. a loaded model or a mutable counter) — this is the core primitive everything else in Ray is built from.
- **Ray Train**: distributes model training across multiple GPUs/nodes with a thin wrapper around existing training loops (PyTorch, HuggingFace `Trainer`, etc.) — handles data sharding, gradient synchronization, and fault tolerance without hand-rolling `DistributedDataParallel` boilerplate yourself.
- **Ray Serve**: a model-serving library with Python-native model composition (chaining multiple models/steps as a graph), dynamic batching, and autoscaling — designed specifically for ML serving patterns that raw Kubernetes Deployments don't express as directly.
- **Ray Tune**: hyperparameter search at scale, with early-stopping schedulers (ASHA, PBT — population-based training) that kill underperforming trials early to make a fixed compute budget go further than naive grid/random search.
- **Ray Data**: a distributed data-loading/preprocessing library for ML pipelines, playing a similar role to `tf.data`/Spark DataFrames but tuned for streaming batches into GPU training jobs.
- **Cluster model**: a head node coordinates scheduling and a global control-plane state; worker nodes execute tasks/actors — Ray can run on a single machine (using all local cores) up to a large multi-node autoscaling cluster with the same code.

## Code
```python
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

ray.init()

def train_loop_per_worker(config):
    import torch
    model = build_model()                     # your model definition
    for epoch in range(config["epochs"]):
        for batch in get_batches():
            loss = train_step(model, batch)
        train.report({"loss": loss.item(), "epoch": epoch})

trainer = TorchTrainer(
    train_loop_per_worker,
    train_loop_config={"epochs": 10},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
result = trainer.fit()

# Ray Serve: deploy a model behind an HTTP endpoint
from ray import serve

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class Classifier:
    def __init__(self):
        self.model = load_model()
    async def __call__(self, request):
        payload = await request.json()
        return {"prediction": self.model.predict(payload["features"])}

serve.run(Classifier.bind())
```

## When to use it vs alternatives
- **vs Spark for distributed compute**: Spark is optimized for SQL-style structured data transformations at scale; Ray is optimized for general Python — arbitrary functions, stateful actors, deep-learning training loops. Distributed model training/serving overwhelmingly favours Ray; large-scale ETL/joins/aggregations favour Spark. Many stacks use both.
- **vs `accelerate`/native PyTorch DDP for distributed training**: those work well when you're staying within a single framework's idioms and don't need Ray's broader ecosystem (Tune, Serve) alongside training; Ray Train is worth it when you want one consistent framework across training, tuning, and serving, or need to scale beyond what a single script's DDP setup handles cleanly.
- **vs Kubernetes directly for serving**: Ray Serve runs on top of Kubernetes (or standalone) and gives Python-native model composition and batching without writing that logic as raw Kubernetes resources — see [[kubernetes]].

## Interview angle
**Q. Why would you choose Ray Train over plain PyTorch DistributedDataParallel for a training job?**
Ray Train handles cluster provisioning, fault tolerance (worker failure recovery), and data sharding across nodes with far less boilerplate than configuring DDP process groups by hand, and integrates with Ray Tune for hyperparameter search using the same cluster — worth the extra dependency once you're past single-machine multi-GPU into multi-node territory or need it alongside tuning/serving in one platform.

**Q. What does ASHA (in Ray Tune) actually save compute on, and what's the risk?**
ASHA (Asynchronous Successive Halving) periodically compares trials at shared checkpoints and kills the worst-performing fraction early, reallocating that compute to promising trials — this can multiply the number of configurations explorable per unit compute versus running every trial to completion. The risk is killing a trial that would have overtaken others later (e.g. one with a different, slower-converging but ultimately better learning-rate schedule) — mitigated by tuning the halving aggressiveness and grace period.

**Q. How does a Ray Actor differ from a stateless Ray Task, and when do you need one for serving?**
A Task is a one-shot function call with no persisted state between invocations; an Actor is a long-lived process holding state across calls. Model serving needs actors specifically so the (expensive to load) model weights are loaded once and reused across many requests, rather than reloaded per call as a fresh task would require.

## Traps
- Using Ray tasks for something that needs persistent state (a loaded model) instead of an actor — forces reloading state on every call, killing performance.
- Not setting resource requirements (`num_gpus`, `num_cpus`) on remote functions/actors — Ray's scheduler can co-locate more tasks on a node than the hardware can actually support if resource requests aren't declared.
- Assuming Ray Serve's autoscaling and batching are zero-config — dynamic batching parameters (max batch size, batch wait timeout) need tuning against your latency SLA, not left at defaults.
- Treating Ray as a drop-in Spark replacement for large-scale structured ETL — it lacks Spark's mature SQL engine and Catalyst-style query optimization for that workload.

## Related
[[distributed-training]], [[hyperparameter-tuning]], [[model-serving-patterns]], [[apache-spark]], [[kubernetes]]
