---
title: Deep Learning — Map of Content
type: map
domain: deep-learning
roles: [data-scientist, ml-engineer, ai-engineer]
updated: 2026-09-13
---

# Deep Learning — Map of Content

## Why this domain is asked
Deep learning is the bridge between classical ML and the LLM-heavy AI Engineer track — every candidate is expected to derive backprop cold and reason about why training diverges, and ML/AI Engineers get a dedicated round on architecture and training mechanics. It typically carries less weight than classical ML for a DS but more than it for an AI Engineer, where it's foundational.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[neural-network-fundamentals]] | Forward pass, universal approximation intuition | core |
| 2 | [[activation-functions]] | Why ReLU won, saturation and dead-neuron failure modes | core |
| 3 | [[backpropagation]] | The single most-derived algorithm in DL interviews | core |
| 4 | [[loss-functions]] | Choosing and deriving the right objective | core |
| 5 | [[weight-initialization]] | Why Xavier/He init prevents vanishing signal at depth | intermediate |
| 6 | [[vanishing-and-exploding-gradients]] | Explains half of "why doesn't my model train" questions | intermediate |
| 7 | [[optimizers-sgd-adam]] | Adam's moment estimates, when SGD still wins | core |
| 8 | [[learning-rate-schedules]] | Warmup/decay reasoning, practical training knob | intermediate |
| 9 | [[batch-normalization-and-layernorm]] | Why transformers use LayerNorm, not BatchNorm | intermediate |
| 10 | [[dropout-and-regularization-dl]] | DL-specific regularization beyond L1/L2 | core |
| 11 | [[training-tricks-and-debugging]] | The practical "my loss is NaN" toolkit | intermediate |
| 12 | [[convolutional-neural-networks]] | Convolution/pooling mechanics, receptive fields | core |
| 13 | [[cnn-architectures]] | ResNet skip connections and why depth alone doesn't help | intermediate |
| 14 | [[recurrent-networks-and-lstm]] | Sequence modeling before attention, still asked | intermediate |
| 15 | [[sequence-modeling-basics]] | Framing sequence problems before choosing an architecture | core |
| 16 | [[attention-mechanism]] | The mechanical core of every modern LLM question | core |
| 17 | [[transformer-architecture]] | Most-asked architecture in AI Engineer interviews | core |
| 18 | [[embeddings]] | Shared substrate for recommenders, search and LLMs | core |
| 19 | [[transfer-learning-and-finetuning]] | Practical default over training from scratch | core |
| 20 | [[mixed-precision-and-memory]] | Memory-budget reasoning for training/serving | intermediate |
| 21 | [[distributed-training]] | Data/model/pipeline parallelism at scale | advanced |
| 22 | [[autoencoders-and-representation-learning]] | Unsupervised representation learning, anomaly detection tie-in | intermediate |
| 23 | [[generative-models-overview]] | GANs/VAEs/diffusion landscape, sets up LLM generative framing | intermediate |

## How it's tested per role
- **AI Engineer**: deepest and most frequent DL grilling — transformer internals, attention scaling, training-memory arithmetic are assumed prerequisites before any LLM-specific question.
- **ML Engineer**: tested moderately — architecture choice and training-debugging skills matter more than deriving backprop from scratch, though it can still be asked.
- **Data Scientist**: lighter bar — enough to reason about when a neural net beats a tree model and to discuss embeddings for a use case, not to derive gradients live.
- **MLOps / Agentic / FDE**: DL is mostly context, not a tested skill, except where training-infra questions (distributed training, mixed precision) intersect with MLOps.

## Question bank
See [[qbank-deep-learning]] for the drilled question set.

## Related domains
- [[moc-maths]]
- [[moc-classical-ml]]
- [[moc-nlp-llm]]
- [[moc-system-design]]
