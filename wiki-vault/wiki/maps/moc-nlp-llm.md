---
title: NLP & LLMs — Map of Content
type: map
domain: nlp-llm
roles: [ai-engineer, ml-engineer, agentic-engineer, fde]
updated: 2026-09-13
---

# NLP & LLMs — Map of Content

## Why this domain is asked
This is the fastest-growing domain in Indian interviews right now — nearly every AI Engineer and a growing share of ML Engineer loops dedicate a full round to LLM internals, fine-tuning tradeoffs and serving economics, because that's what "AI-first startup" hiring is actually building. Expect it to be the single heaviest-weighted domain for AI Engineer roles.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[text-preprocessing]] | Baseline NLP hygiene, still asked for classical pipelines | core |
| 2 | [[tokenization-bpe-and-sentencepiece]] | Explains vocabulary size, cost, and multilingual quirks | core |
| 3 | [[word-embeddings-word2vec-glove]] | Historical grounding before contextual embeddings | core |
| 4 | [[language-modeling-objectives]] | Causal vs masked LM — sets up encoder/decoder framing | core |
| 5 | [[bert-and-encoder-models]] | Still the default for classification/retrieval encoders | core |
| 6 | [[gpt-and-decoder-models]] | The architecture behind every chat-style LLM | core |
| 7 | [[llm-pretraining]] | Data, compute and objective choices at pretraining scale | intermediate |
| 8 | [[llm-scaling-laws]] | Chinchilla-style tradeoffs, sizing arguments interviewers probe | intermediate |
| 9 | [[instruction-tuning-and-sft]] | First alignment step, why base models aren't chat-ready | core |
| 10 | [[rlhf]] | Classic alignment pipeline, still asked conceptually | intermediate |
| 11 | [[dpo-and-preference-optimization]] | Modern, cheaper alternative to RLHF — increasingly asked | advanced |
| 12 | [[parameter-efficient-finetuning-lora]] | The practical fine-tuning method at 5 YOE cost budgets | core |
| 13 | [[quantization]] | Memory/latency arithmetic for serving on real hardware | core |
| 14 | [[knowledge-distillation]] | Small-model production strategy | intermediate |
| 15 | [[decoding-strategies]] | Greedy/beam/sampling tradeoffs affecting output quality | core |
| 16 | [[context-window-and-positional-encoding]] | RoPE/ALiBi reasoning, long-context tradeoffs | intermediate |
| 17 | [[kv-cache-and-inference-optimization]] | Core serving-cost lever, asked in every serving discussion | advanced |
| 18 | [[llm-serving-and-throughput]] | Batching, latency/throughput tradeoffs in production | advanced |
| 19 | [[flash-attention-and-efficient-attention]] | Memory-bound attention optimization, common follow-up | advanced |
| 20 | [[mixture-of-experts]] | Sparse-activation architecture behind frontier-scale models | advanced |
| 21 | [[prompt-engineering]] | Table-stakes skill, but interviewers probe for rigor not tricks | core |
| 22 | [[structured-output-and-function-calling]] | Bridges LLMs to agents and production APIs | core |
| 23 | [[hallucination-and-grounding]] | Central production risk question for any LLM app | core |
| 24 | [[llm-evaluation]] | How you actually know a prompt/model change helped | core |
| 25 | [[llm-safety-and-guardrails]] | Increasingly asked at enterprise/regulated clients | intermediate |
| 26 | [[multimodal-models]] | Vision-language models entering mainstream production use | intermediate |
| 27 | [[small-language-models-and-cost]] | Cost-conscious model selection, an Indian-market recurring theme | intermediate |

## How it's tested per role
- **AI Engineer**: the deepest and broadest bar — pretraining/fine-tuning tradeoffs, serving optimization, evaluation design all get grilled in dedicated rounds.
- **ML Engineer**: tested on the parts that touch production — fine-tuning strategy, quantization, serving cost — less on pretraining/RLHF theory.
- **Agentic Engineer**: needs strong grounding in prompt engineering, structured output and evaluation, since these directly gate agent reliability.
- **FDE**: needs practical fluency — cost/latency tradeoffs and prompt-engineering judgment — to make fast client-facing architecture calls, not internals depth.

## Question bank
See [[qbank-nlp-llm]] for the drilled question set.

## Related domains
- [[moc-deep-learning]]
- [[moc-rag]]
- [[moc-agents]]
- [[moc-system-design]]
