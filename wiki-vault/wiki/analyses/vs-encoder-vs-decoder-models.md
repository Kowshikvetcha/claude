---
title: Encoder vs Decoder Models
type: analysis
domain: nlp-llm
roles: [ai-engineer, ml-engineer, data-scientist]
difficulty: intermediate
frequency: high
status: drafted
tags: [bert, gpt, transformer, embeddings, architecture]
updated: 2026-09-11
sources: []
---

# Encoder vs Decoder Models

## TL;DR
Encoder models (BERT-style) read the whole input at once with bidirectional attention and are
built for *understanding* — classification, embeddings, retrieval, NER. Decoder models (GPT-style)
attend only to the left context (causal masking) and are built for *generation* — writing text,
chat, code, agentic reasoning. If the output is a label, a score, or a vector, reach for an
encoder. If the output is more text, reach for a decoder.

## The real question being asked
This tests whether you understand *why* the attention mask pattern determines what a model is
good for, not just "BERT is old, GPT is new." A candidate who can't explain that bidirectionality
is precisely what makes an encoder unsuitable for autoregressive generation (and vice versa) is
pattern-matching on model names rather than understanding the architecture. Interviewers also use
this to probe whether you'd reach for an expensive generative LLM call when a cheap encoder
embedding or classifier would do the job more cheaply, more reliably, and with lower latency —
a very common real-world mistake in 2025-26 era LLM-happy teams.

## Side by side

| Dimension | Encoder (BERT-style) | Decoder (GPT-style) |
|---|---|---|
| Attention pattern | Bidirectional — every token sees every other token | Causal/masked — each token sees only itself and earlier tokens |
| Native output | A vector per token / pooled sequence vector | Next-token probability distribution, generated autoregressively |
| Typical use | Classification, NER, embeddings, retrieval, reranking | Chat, generation, summarisation, code, reasoning, agents |
| Training objective | Masked language modelling (predict masked tokens using both directions) | Next-token prediction (predict token $t+1$ from tokens $1..t$) |
| Inference cost per call | One forward pass, cheap, no decoding loop | Multi-step decoding, one forward pass per generated token (mitigated by [[kv-cache-and-inference-optimization]]) |
| Latency | Low, milliseconds, easily batched | Higher, scales with output length |
| Output determinism | Deterministic given weights (no sampling needed) | Stochastic by default (temperature/sampling), needs [[decoding-strategies]] |
| Fine-tuning data needs | Small labelled sets often enough (task head + light finetune) | Instruction-tuning/RLHF needs larger, carefully curated data, see [[instruction-tuning-and-sft]] |
| Model sizes in practice | Often 100M–1B params, cheap to self-host | Ranges from small (SLMs) to hundreds of billions, see [[small-language-models-and-cost]] |
| Explainability of "why this label" | Attention/embedding-based tools are mature | Harder — free-text output, no fixed label space |
| Composability with retrieval | Natural fit — embeddings feed vector search, see [[embedding-models]] | Needs to be told to retrieve, or wrapped in a RAG pipeline |

## When Encoder wins
- You need a fixed-size vector representation of text: semantic search, deduplication, clustering,
  or as input to a downstream tabular/XGBoost model. See [[embedding-models]], [[vector-databases]].
- The task has a known, closed label space: sentiment, intent classification, NER, toxicity
  filtering — cheaper and more reliable than prompting a generative model for a label.
- You need low, predictable latency at high QPS (a real-time content moderation filter, a search
  reranker) — see [[latency-and-throughput-budgets]].
- You want a deterministic, auditable decision boundary rather than a sampled free-text answer.

## When Decoder wins
- The task genuinely requires producing novel text: drafting, summarising, translating,
  conversational agents, chain-of-thought reasoning, code generation.
- The task is open-ended or the label space isn't known ahead of time (few-shot classification of
  a category you didn't anticipate, flexible instruction-following).
- You need the model to call tools, plan multi-step actions, or hold a conversation — see
  [[agent-fundamentals]], [[tool-calling-and-function-schemas]].
- You want one general-purpose model to handle many tasks via prompting instead of training a
  separate head per task.

## The honest hybrid answer
Most production NLP/LLM systems use both, at different stages of the same pipeline: an encoder
(or a bi-encoder/embedding model, often distilled from a larger decoder) does retrieval and
reranking cheaply at scale, and a decoder does the final generation step conditioned on what the
encoder retrieved. This is exactly the architecture of RAG — see [[rag-overview]] and
[[reranking]] — and it's also common to use an encoder-based classifier as a cheap pre-filter
("is this even in scope?", "is this toxic?") before paying for an expensive decoder call. Choosing
"encoder or decoder" per task, rather than routing everything through one large generative model,
is usually the cost-and-latency-conscious answer an interviewer wants to hear.

## Interview angle
**Q. Why can't you use BERT to generate text token by token the way GPT does?**
BERT's attention is bidirectional and it was trained to fill in masked tokens using context from
both directions, including tokens *after* the position being predicted. There is no causal mask,
so at generation time it has no well-defined way to produce token $t+1$ using only $1..t$ — the
training objective and the architecture assume the full sequence is already visible. GPT-style
models use a causal mask specifically so training matches the autoregressive generation process
used at inference.

**Q. You need to classify 10 million support tickets into 40 categories with a tight latency SLA.
Would you use an LLM (decoder) or a fine-tuned encoder?**
Fine-tuned encoder — it's a closed-label classification task, an encoder gives a single cheap
forward pass per ticket with low, predictable latency, and it's straightforward to fine-tune on a
few thousand labelled examples per category. Routing this through a large decoder model via
prompting would be slower, more expensive at that volume, and less deterministic for something
that needs to be auditable.

**Follow-up.** What if the category set changes every quarter?
That pushes toward either a decoder used few-shot/zero-shot (flexible label space, no retraining
per change) or periodically re-fine-tuning the encoder's classification head — the tradeoff is
retraining cost and turnaround time vs per-call inference cost, which is the same RAG-vs-finetune
tradeoff discussed in [[vs-rag-vs-finetuning]].

## Related
[[bert-and-encoder-models]]
[[gpt-and-decoder-models]]
[[transformer-architecture]]
[[embeddings]]
[[embedding-models]]
[[decoding-strategies]]
[[llm-system-design-framework]]
