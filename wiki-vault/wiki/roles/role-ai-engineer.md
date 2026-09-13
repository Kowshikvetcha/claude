---
title: AI Engineer
type: role
domain: meta
roles: [ai-engineer]
updated: 2026-09-13
---

# AI Engineer

## What this role actually does
"AI Engineer" is the fastest-growing and least legacy-laden title in the India market right now, and at ~5 years experience it means: building products on top of LLMs — RAG pipelines, agentic workflows, prompt/eval systems — rather than training models from scratch. This is a meaningfully different job from "ML Engineer": you are far more likely to be composing APIs (OpenAI/Anthropic/open-weight models via an inference provider), a vector database, and an orchestration layer than fitting a gradient-boosted tree.

In **product companies**, AI Engineers usually sit inside a "GenAI" or "Applied AI" team bolted onto an existing product, shipping features like assistants, summarization, or semantic search — with real pressure on latency, cost per call, and hallucination rate. In **AI-first startups**, this is often the core engineering job, with heavy ownership of the whole stack from prompt to eval to deployment, and much faster iteration cycles. In **GCCs**, AI Engineer roles are newer and often more experimental — proof-of-concept work for a global business unit, less production pressure, but also less budget and slower approval cycles for new tools. **Service firms** increasingly staff "GenAI" pods for client RAG/chatbot projects — breadth across client domains, less depth in any one system, heavy emphasis on demoing quickly.

Classical ML is not the focus here — you're expected to know enough to be conversant (e.g. why a smaller fine-tuned classifier might beat an LLM call for a narrow task) but the day-to-day is prompt design, retrieval quality, evaluation harnesses, and cost/latency tradeoffs across model choices. This is also the role where "5 years experience" is least standardized — many candidates are ML/SWE engineers who pivoted in the last 1–2 years, so interviewers often probe fundamentals more than tenure would suggest.

## Typical interview loop
Typically 4–5 rounds, with unusually high variance because the role is new:
1. **Recruiter screen** — often includes a quick gut-check on hands-on LLM/RAG experience vs. just "used ChatGPT a lot."
2. **Coding round** — usually Python, sometimes lighter on classic DSA than an SWE loop, but a working coding bar (build a small RAG pipeline or parse/transform a dataset) is common.
3. **LLM/RAG/agents deep dive** — chunking strategy tradeoffs, embedding model choice, when RAG vs. fine-tuning, agent loop design, tool-calling schemas, hallucination mitigation.
4. **System design (LLM-flavoured)** — design a RAG assistant or an agentic workflow for a given business problem, with explicit reasoning about latency, cost per query, and evaluation strategy.
5. **Hiring manager / founder round** (common at startups) — product sense, how you'd prioritize between improving retrieval vs. prompt vs. adding an agent step, and behavioral fit.

Some companies add a take-home (build a small RAG or agent demo) in place of round 3 or 4 — increasingly common precisely because the role is new and resistant to standard whiteboard formats.

## Domain weighting
| Domain | Weight | Why |
|---|---|---|
| [[moc-nlp-llm]] | Very high | Core theory — tokenization, decoding, fine-tuning methods, evaluation — is assumed baseline knowledge. |
| [[moc-rag]] | Very high | This is the flagship AI Engineer skill; expect deep questions on chunking, retrieval, reranking, and failure modes. |
| [[moc-agents]] | High | Tool calling, ReAct-style loops, and agent evaluation are increasingly core, not a bonus topic. |
| [[moc-system-design]] | Medium-high | Specifically the LLM-flavoured framework — cost/latency/eval reasoning over classical distributed-systems depth. |
| [[moc-programming]] | Medium | Solid Python and API composition skills; DSA bar usually lighter than a pure SWE/MLE loop. |
| [[moc-mlops]] | Medium | LLMOps concerns (versioning prompts, monitoring hallucination rate) matter, less than for an MLE. |
| [[moc-behavioral]] | Medium | Product sense and prioritization judgment are explicitly interviewed given the role's ambiguity. |
| [[moc-data-engineering]] | Low-medium | Needed for ingestion pipelines feeding RAG systems, not a deep focus. |
| [[moc-classical-ml]] | Low | Awareness expected (when NOT to use an LLM), rarely tested in depth. |
| [[moc-deep-learning]] | Low-medium | Transformer/attention fundamentals matter since they underpin everything else here. |

## JD vocabulary
- **"LLM application development"** → building products around API-based models, not training them; expect zero questions about pretraining infrastructure.
- **"RAG pipeline"** → they want someone who has actually debugged retrieval quality (wrong chunks, stale embeddings), not just called a `retriever.invoke()`.
- **"Agentic workflows" / "autonomous agents"** → tool-calling, multi-step planning, and — critically — failure handling when a tool call goes wrong or loops.
- **"Prompt engineering"** → increasingly means *systematic* prompt engineering (versioning, A/B testing prompts, structured output schemas), not ad hoc tweaking.
- **"Evals"** → a strong signal of a mature team; expect to discuss both automated (LLM-as-judge, exact-match, retrieval metrics) and human evaluation loops.
- **"Cost optimization" / "token efficiency"** → real operational concern in India-based teams with tighter per-query budgets than a well-funded US startup; expect to discuss caching, smaller models for sub-tasks, and prompt compression.
- **"Multi-modal"** → increasingly appears in JDs even for text-first teams; know the concept even if you haven't shipped it.

## Readiness checklist
- [ ] Can explain the tradeoffs between RAG and fine-tuning for a given business problem and argue both sides.
- [ ] Can design a chunking strategy for at least two different document types (long-form prose vs. tabular/structured) and justify chunk size choices.
- [ ] Can explain hybrid search (BM25 + vector) and when pure vector search fails.
- [ ] Can explain reranking and why a retriever's top-k isn't the final answer.
- [ ] Can design an evaluation harness for a RAG system — what metrics, what's automated vs. human-reviewed.
- [ ] Can design a tool-calling schema for an agent and reason about what happens when a tool call fails or returns garbage.
- [ ] Can explain the difference between a ReAct-style loop and a simpler single-shot tool call, and when each is warranted.
- [ ] Can explain quantization and why it matters for serving cost/latency, at a conceptual level.
- [ ] Can explain KV-cache and why it matters for inference latency, without needing to derive it from scratch.
- [ ] Can diagnose a hallucination failure mode and propose at least two concrete mitigations (grounding, structured output, verification step).
- [ ] Can reason about cost per query across model size choices and justify when a smaller/cheaper model suffices.
- [ ] Can walk through one real RAG or agent system you built end to end, including what broke and how you fixed it.
- [ ] Comfortable explaining why a classical ML model might outperform an LLM call for a narrow, high-volume task.

## The night-before-list
- [[moc-rag]]
- [[rag-overview]]
- [[chunking-strategies]]
- [[hybrid-search-bm25-vector]]
- [[reranking]]
- [[rag-evaluation]]
- [[rag-failure-modes]]
- [[advanced-rag-patterns]]
- [[moc-agents]]
- [[tool-calling-and-function-schemas]]
- [[react-and-reasoning-loops]]
- [[agent-evaluation]]
- [[agent-guardrails-and-safety]]
- [[llm-evaluation]]
- [[hallucination-and-grounding]]
- [[prompt-engineering]]
- [[structured-output-and-function-calling]]
- [[llm-system-design-framework]]
- [[quantization]]
- [[kv-cache-and-inference-optimization]]

## Related
- [[moc-nlp-llm]]
- [[moc-rag]]
- [[moc-agents]]
- [[moc-system-design]]
- [[plan-12-week]]
- [[plan-7-day-sprint]]
