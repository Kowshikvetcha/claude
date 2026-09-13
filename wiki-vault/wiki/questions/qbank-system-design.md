---
title: System Design Question Bank
type: qbank
domain: system-design
roles: [ml-engineer, ai-engineer, mlops-engineer, agentic-engineer, fde]
difficulty: core
frequency: high
status: drafted
tags: [question-bank]
updated: 2026-09-13
---

# System Design Question Bank

> How to use: cover the answers, write yours first, then compare. Every answer should be structurable through [[ml-system-design-framework]] (or [[llm-system-design-framework]] for LLM-flavoured questions) — if you can't hang your answer on that skeleton, that's the gap to close.

## Warm-up

### Q1. Walk through the structure you'd use to answer "design a fraud-detection system" in a 45-minute round.
**Answer.** Use [[ml-system-design-framework]]: clarify requirements and metrics first (latency budget, precision/recall tradeoff, cost of a false decline vs a missed fraud), then data → features → model → serving → monitoring → failure modes → tradeoffs, narrating the reasoning out loud at each step rather than jumping straight to an architecture diagram.
**Follow-ups.** What's the single most common way candidates lose points in the first five minutes? → Proposing an architecture before pinning down the metric and latency budget — every later decision (batch vs real-time, model complexity) depends on those numbers.
**Page.** [[ml-system-design-framework]]. See [[case-fraud-detection]] for a worked example.

### Q2. What's the first question you should ask before proposing any architecture, and why do candidates lose points by skipping it?
**Answer.** Pin down the requirements and the metric: what does "good" mean numerically (a business metric, not just accuracy), what's the latency/throughput budget, and what's the cost asymmetry between error types. Skipping this makes every later design choice (model complexity, batch vs real-time, caching) look arbitrary to the interviewer, even if the final architecture is reasonable.
**Follow-ups.** How do you turn a vague ask like "make search better" into a metric? → Anchor on a measurable proxy the business already tracks or could track (CTR, add-to-cart rate, session length) and state the tradeoff it makes against other metrics (e.g. relevance vs diversity).
**Page.** [[requirements-and-metrics-definition]]

### Q3. Batch vs real-time inference — how do you decide, and what does each imply for the rest of the design?
**Answer.** Decide based on how fresh the prediction needs to be relative to how expensive/complex a low-latency serving path is to build and operate. Batch (score everything nightly, store results in a table) is cheap and simple but predictions can be hours stale; real-time (a model server behind an API) gives freshness but adds a serving SLA, needs a feature store for parity, and costs more to run 24/7.
**Follow-ups.** What's a common middle ground? → Near-real-time micro-batching (e.g. every few minutes) or a batch-scored cache with a real-time fallback for cold-start entities.
**Page.** [[batch-vs-realtime-inference]]

### Q4. Why does training-serving skew happen, and name two concrete causes.
**Answer.** It happens when the feature values or computation logic a model saw at training time differ from what it sees at inference time. Two concrete causes: (1) feature computed with a batch pipeline offline (e.g. Spark) but recomputed with different logic in a low-latency online path; (2) a feature using data that wasn't actually available at prediction time in production (time-travel leakage), inflating offline metrics that then don't hold up live.
**Follow-ups.** What's the standard fix? → A feature store that serves the exact same feature computation/values to both training and serving paths.
**Page.** [[training-serving-skew]]

### Q5. What is CAP theorem and why does it matter when choosing a store for an online feature-serving layer?
**Answer.** Under a network partition, a distributed system can guarantee either consistency (every read sees the latest write) or availability (every request gets a response), not both. It matters for feature stores because a partition-tolerant, highly-available online store (e.g. eventually-consistent key-value store) may serve slightly stale features rather than fail the prediction request — usually the right tradeoff for ML serving, where a stale feature is cheaper than a dropped request.
**Follow-ups.** Where would you instead need strong consistency in an ML system? → A model registry/deployment record — you never want two serving replicas disagreeing about which model version is "current."
**Page.** [[cap-theorem-and-consistency]]

## Core

### Q6. Design a recommendation system for an e-commerce homepage.
**Answer.** Requirements: metric is likely CTR or add-to-cart rate, latency budget ~100-200ms for a page render. Data: implicit signals (views, clicks, purchases) plus item metadata. Features: user history embeddings, item embeddings, recency/popularity signals. Model: two-stage — a cheap candidate-generation model (collaborative filtering or embedding nearest-neighbour) narrowing millions of items to hundreds, then a heavier ranking model (gradient-boosted trees or a small neural ranker) scoring the shortlist. Serving: candidate generation can be precomputed/cached per user segment; ranking happens online. Monitoring: track CTR by segment and watch for feedback-loop drift (the model only sees items it already recommended).
**Follow-ups.** How do you handle cold-start users/items? → Fall back to popularity- or content-based features (item metadata, category) until enough interaction data accumulates for the collaborative signal to kick in.
**Page.** [[ml-system-design-framework]]. See [[case-recommendation-system]].

### Q7. How would you design the caching layer for a search-ranking system serving 10k QPS with a p99 latency budget of 100ms?
**Answer.** Identify what's cacheable without hurting relevance: query-independent signals (item embeddings, popularity scores) can be cached aggressively with long TTLs; full query results can be cached for the most frequent/repeated queries with a short TTL to stay fresh. Put a cache in front of the expensive reranking step specifically, since that's usually the latency bottleneck, and size the cache/TTL from the query-frequency distribution (a long-tail of unique queries won't benefit from result caching, only from caching the reusable sub-computations).
**Follow-ups.** What's the risk of caching too aggressively here? → Serving stale rankings after an index update or a ranking-model deploy — need a cache-invalidation hook tied to those events, not just a TTL.
**Page.** [[caching-strategies]], [[latency-and-throughput-budgets]]. See [[case-search-ranking]].

### Q8. Design the API contract between an ML model server and its consumers — what needs to be in it beyond the raw prediction?
**Answer.** Beyond the prediction value: a model/version identifier (so a consumer can log which model produced which decision — critical for debugging and rollback), a confidence/probability score (not just the argmax class), input feature values actually used (for audit and to catch silent schema drift), and a request-level correlation ID for tracing through logs/monitoring. The contract should also define a clear error/fallback response for when the model can't score a request (missing features, timeout) rather than letting the consumer crash.
**Follow-ups.** Why does versioning the response matter so much in a regulated or high-stakes domain? → You need to reconstruct exactly which model version made a given decision months later, for audit or complaint handling — this is a compliance requirement in lending/insurance-adjacent systems.
**Page.** [[api-design-for-ml]]

### Q9. Design a demand-forecasting pipeline on a medallion-architecture lakehouse.
**Answer.** Bronze: raw point-of-sale/inventory events landed as-is from source systems. Silver: cleaned, deduplicated, conformed to a consistent schema (one row per SKU-store-day), with data-quality checks (no negative quantities, no future-dated events). Gold: aggregated time-series features per SKU-store (rolling averages, seasonality flags, promotion calendars) ready for the forecasting model. The forecasting job reads from gold, trains/scores per SKU-store combination (often thousands of models or one global model with SKU as a feature), and writes forecasts back to a serving table for downstream planning systems to consume.
**Follow-ups.** Why forecast per-SKU models instead of one global model, and what's the tradeoff? → Per-SKU models capture idiosyncratic seasonality but don't share signal across similar products and don't scale operationally to tens of thousands of models; a global model with SKU/category as features shares statistical strength (especially for sparse SKUs) at the cost of being less tailored to any one item.
**Page.** [[medallion-architecture]]. See [[case-demand-forecasting]].

### Q10. How do you scale a model-serving layer horizontally, and what state (if any) makes that hard?
**Answer.** Stateless model replicas behind a load balancer scale horizontally trivially — add more pods/instances as QPS grows, since each request is independent. What makes it hard is any state the request path depends on: an in-memory feature cache that isn't shared across replicas (each replica computes/caches independently, wasting memory and risking staleness skew between replicas), or session-affinity requirements for a stateful conversational agent. The fix is pushing that state into a shared external store (a cache or feature store) rather than keeping it replica-local.
**Follow-ups.** What's a load-balancing pitfall specific to ML serving? → Uneven request cost (some inputs — e.g. long documents to an LLM, or large batches — take far longer to score than others), so naive round-robin load balancing can leave some replicas overloaded; needs load-aware or queue-depth-aware routing.
**Page.** [[scalability-patterns]], [[distributed-systems-basics]]

### Q11. Design the monitoring plan for a fraud-detection model in production.
**Answer.** Track three layers: (1) system health — latency, error rate, throughput of the serving path; (2) data/input health — feature-distribution drift versus the training distribution, schema violations, missing-feature rates; (3) model-quality health — proxy metrics available in near-real-time (flagged-transaction rate, analyst override rate) plus delayed ground-truth metrics once chargebacks/confirmed-fraud labels arrive weeks later. Alert on drift and proxy-metric shifts immediately; use the delayed ground truth to validate that the drift actually mattered and to trigger retraining decisions.
**Follow-ups.** Why can't you just monitor accuracy directly for fraud? → Ground-truth fraud labels arrive with a lag (confirmed via chargeback/dispute, often weeks later) and the classes are heavily imbalanced, so you need faster proxy signals to catch degradation before the lagging metric confirms it.
**Page.** [[model-monitoring]], [[data-drift-and-concept-drift]]. See [[case-fraud-detection]].

### Q12. How would you design a feature pipeline that guarantees feature parity between training and serving?
**Answer.** Route both the offline training job and the online serving path through the same feature-store definitions — the transformation logic is written once (e.g. as a PySpark job for batch features, or a shared function for on-demand features) and both paths read from it, rather than reimplementing the logic twice in different languages/systems. The feature store materializes an offline table for training and an online low-latency store (key-value) for serving, both populated from the same source-of-truth definitions, so what the model learned on is what it sees live.
**Follow-ups.** What's the failure mode when teams skip a feature store? → Someone reimplements a "rolling 7-day average" feature slightly differently in the real-time Java service versus the offline Spark job, and the model silently degrades because the online feature distribution no longer matches what it was trained on — classic training-serving skew.
**Page.** [[feature-stores]], [[training-serving-skew]]

## Hard

### Q13. Design an LLM-powered customer support system end-to-end.
**Answer.** Use [[llm-system-design-framework]]: requirements first (accuracy/groundedness bar, latency budget per turn, cost per conversation, escalation-to-human path). Architecture: RAG over the support knowledge base for factual grounding, a smaller/cheaper model for intent classification and routing, the main LLM for response generation with function-calling for actions (refunds, order lookups), and a guardrail layer to catch hallucinated policy claims before they reach the customer. Serving needs streaming responses for perceived latency, and monitoring needs both automated groundedness checks and a human-review sample, since correctness can't be fully automated.
**Follow-ups.** What's the biggest cost lever in a design like this? → Right-sizing the model per sub-task — a cheap small model for intent routing and a larger model only for final generation, rather than routing every turn through the most expensive model.
**Page.** [[llm-system-design-framework]]

### Q14. Design the serving stack for a 70B-parameter LLM at low latency and cost.
**Answer.** Quantize the model (int8/int4) to cut memory bandwidth, the usual inference bottleneck, and fit more of the model/batch on fewer GPUs. Use KV-caching so autoregressive decoding doesn't recompute attention over the whole prefix at every step, and continuous/dynamic batching so the server keeps GPUs busy across requests arriving at different times instead of padding to the slowest request in a static batch. For latency-sensitive traffic, consider a smaller distilled model in front, escalating to the full model only when needed.
**Follow-ups.** Why does batching help throughput but hurt latency, and how do you resolve the tension? → Bigger batches amortize the fixed cost of a forward pass over more tokens (higher throughput) but make each request wait for the batch to fill or for other requests in the batch to finish (higher latency); continuous batching (adding/removing sequences from the batch every step rather than at fixed batch boundaries) captures most of the throughput win with much less added latency.
**Page.** [[kv-cache-and-inference-optimization]], [[llm-serving-and-throughput]]

### Q15. Design a RAG system for internal enterprise document search.
**Answer.** Ingestion: parse heterogeneous documents (PDFs, wikis, tickets), chunk them at a size that balances context completeness against embedding relevance, and embed with a model matched to the domain (consider fine-tuning or a domain-tuned embedding model for jargon-heavy enterprise text). Retrieval: a vector database for semantic search, combined with keyword/BM25 for exact term matches (IDs, product codes) that embeddings blur, and a reranker on the merged candidate set before the LLM sees them. Serving: assemble a context window from top reranked chunks with citations, and evaluate with retrieval-recall metrics offline plus a groundedness/faithfulness check online.
**Follow-ups.** What's the most common way a real enterprise RAG deployment fails that a demo never exposes? → Document permissions — retrieval has to respect the same access controls as the source system, or the system leaks content a user shouldn't see; this is usually bolted on late and causes real production incidents.
**Page.** [[rag-overview]], [[vector-databases]]

### Q16. Design a multi-agent system for automating a multi-step client workflow.
**Answer.** Decompose the workflow into sub-tasks each handled by a specialized agent (e.g. a data-extraction agent, a validation agent, an action-taking agent) coordinated by an orchestrator that owns the overall plan and can intervene when an agent's output looks wrong, rather than letting agents freely hand off to each other unsupervised. Put a human-in-the-loop checkpoint before any irreversible action (sending an email, writing to a client's system of record), cap the number of agent-to-agent steps to prevent runaway loops, and log every intermediate decision for debugging, since failures in multi-agent systems are usually a silently-wrong intermediate step rather than an obvious crash.
**Follow-ups.** When is a multi-agent design overkill versus a single well-prompted agent or a fixed pipeline? → When the sub-tasks are actually deterministic/well-specified — a fixed pipeline is cheaper, faster, and far easier to debug than an agent architecture, which should be reserved for genuinely open-ended planning under uncertainty.
**Page.** [[multi-agent-systems]], [[agent-cost-and-latency-optimization]]

### Q17. You need to cut LLM inference cost by 60% without materially hurting quality. Walk through your levers.
**Answer.** In order of usual impact-to-effort: (1) route by task difficulty — a small/cheap model for easy queries, escalate to the expensive model only when needed (a cascade); (2) quantize the serving model to cut per-token compute/memory cost; (3) cache repeated or near-duplicate requests/prompts (common in support/FAQ-style traffic); (4) shrink the prompt/context — trim retrieved context, use shorter system prompts, avoid re-sending full conversation history every turn; (5) as a last resort, distill a smaller model on the expensive model's outputs for the specific task. Validate each change against the quality metric before rolling out, since cost cuts that silently degrade quality are the real risk, not the cost itself.
**Follow-ups.** Why is model cascading usually the highest-leverage lever? → Most production traffic is easy (simple lookups, common intents) and doesn't need the frontier model at all — routing that majority to a much cheaper model captures most of the savings with minimal quality risk on the hard tail that still gets escalated.
**Page.** [[cost-optimization-for-ml]], [[quantization]]

## Related
See [[moc-system-design]].
