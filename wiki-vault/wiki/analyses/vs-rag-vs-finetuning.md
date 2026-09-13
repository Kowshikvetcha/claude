---
title: RAG vs Fine-Tuning
type: analysis
domain: rag
roles: [ai-engineer, ml-engineer, mlops-engineer]
difficulty: intermediate
frequency: high
status: drafted
tags: [rag, finetuning, lora, llm, system-design]
updated: 2026-09-11
sources: []
---

# RAG vs Fine-Tuning

## TL;DR
Use RAG when the model needs access to facts that change or are too large to memorise; use
fine-tuning (usually LoRA) when the model needs to change *how* it behaves — format, tone, task
skill, domain jargon. Most production systems that look mature use both, because the two problems
(what the model knows vs how the model acts) are usually both present.

## The real question being asked
The interviewer is checking whether you understand that RAG and fine-tuning solve different
failure modes, not competing implementations of the same feature. A candidate who says "RAG is
cheaper so always use RAG" or "fine-tuning gives better answers so fine-tune" hasn't understood
either technique. The real test is: can you map a stated business problem ("answers are outdated",
"answers ignore our internal style guide", "the model doesn't know our product catalogue", "the
model can't do this niche extraction task reliably") to the right tool, and can you say why the
other tool would fail on it.

## Side by side

| Dimension | RAG | Fine-tuning (LoRA) |
|---|---|---|
| Solves | Missing / stale / private knowledge | Missing behaviour, style, task skill |
| Freshness | Update the index, answer changes instantly | Requires retraining to update |
| Cost to stand up | Retrieval infra (chunking, embeddings, vector store, reranker) | GPU time for training runs, data curation |
| Cost per query | Extra retrieval latency + larger prompt (more input tokens) | None extra — same inference cost as base model |
| Data requirement | A document corpus, no labels needed | Labelled (instruction, response) pairs, ideally hundreds to low thousands of good examples |
| Attribution / citations | Natural — you have the retrieved chunks | Impossible — knowledge is baked into weights |
| Hallucination risk | Lower on in-context facts, still possible if retrieval is bad | Model can still hallucinate; behaviour is now "confidently wrong in the new style" |
| Domain jargon / style | Prompting can partially fake it; imperfect | Learns exactly the target register in enough examples |
| Maintenance | Re-index on data change, monitor retrieval quality | Retrain on drift, manage adapter versions |
| Failure mode | Retrieves wrong/irrelevant chunks, context gets crowded | Overfits on small data, forgets general ability (catastrophic forgetting) |
| Auditability | High — can show the source passage | Low — behaviour change is opaque |

## When RAG wins
- Knowledge changes weekly/daily (pricing, policy, ticket history, live inventory).
- You need citations or an audit trail for compliance (BFSI, healthcare in India especially).
- The corpus is too large or too private to ever put in training data (customer PII, internal
  wikis) — see [[security-and-pii-in-ml]].
- You want to swap or remove a document instantly without retraining anything.
- You don't have (and can't afford to build) a labelled instruction dataset.

## When fine-tuning wins
- The failure is behavioural, not factual: the model knows the facts but answers in the wrong
  format, ignores instructions, or doesn't consistently call tools correctly. See
  [[structured-output-and-function-calling]].
- You need consistent structured output (a fixed JSON schema, a specific extraction template) at
  high volume, where paying the "explain the format every time" prompt tax is wasteful.
- Domain style/vocabulary is pervasive rather than a lookup — legal drafting tone, a company's
  support voice, a very specific coding convention.
- Latency and cost per call matter more than freshness — no retrieval hop, no reranker, one
  forward pass.
- You have (or can generate, e.g. via distillation from a bigger model) enough quality examples —
  LoRA needs far less data than full fine-tuning. See [[parameter-efficient-finetuning-lora]] and
  [[knowledge-distillation]].

## The honest hybrid answer
Real systems combine them constantly: fine-tune the model to *reliably follow a retrieval-and-cite
protocol* (e.g. "always answer using only the provided context, cite the chunk id, say 'I don't
know' if the context doesn't cover it") and then run RAG on top of that fine-tuned model. The
fine-tune fixes behaviour (format discipline, tool-calling reliability, refusal calibration); RAG
supplies the facts. This is exactly the pattern behind most enterprise support-bot and agentic-RAG
deployments — see [[case-rag-assistant]] and [[agentic-rag]]. The two costs are additive, not
alternative, which is the point interviewers want you to notice: "RAG vs fine-tuning" is a false
dichotomy in production, useful mainly as a diagnostic question ("is this a knowledge gap or a
behaviour gap?") before you design the system.

## Interview angle
**Q. Our support bot gives outdated answers about refund policy. RAG or fine-tune?**
RAG. Policy changes are a knowledge-freshness problem, not a behaviour problem — index the current
policy docs and retrieve at query time. Fine-tuning would bake in whatever policy existed at
training time and go stale again next quarter.

**Q. Our extraction model returns free text instead of the JSON schema we need, even though we
prompt it every time. What do you do?**
This is a behaviour problem with prompting already failing, so fine-tune (LoRA on schema-following
examples) rather than adding more prompt engineering or RAG — RAG wouldn't touch the format issue
at all, and every call is currently paying the token cost of restating the schema.

**Follow-up.** What if you can't collect enough labelled examples for fine-tuning?
Generate them: use a stronger model to produce (input, correctly-formatted output) pairs from your
real traffic, filter with validation rules or a judge model, and fine-tune on that distilled set —
see [[knowledge-distillation]] and [[llm-evaluation]] for how to check the distilled data is
actually correct before training on it.

## Related
[[rag-overview]]
[[parameter-efficient-finetuning-lora]]
[[instruction-tuning-and-sft]]
[[agentic-rag]]
[[hallucination-and-grounding]]
[[case-rag-assistant]]
[[llm-system-design-framework]]
