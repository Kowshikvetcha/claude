---
title: System Design — Map of Content
type: map
domain: system-design
roles: [ml-engineer, ai-engineer, mlops-engineer, agentic-engineer, fde]
updated: 2026-09-13
---

# System Design — Map of Content

## Why this domain is asked
At 5 years' experience, nearly every role in this vault gets at least one "design an ML/LLM system for X" round — it's the format interviewers use to see whether a candidate can integrate modeling, data and infra judgment under real constraints, not just answer isolated questions. Expect one of the higher-stakes rounds in the loop, often the tie-breaker between a strong-hire and a hire.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[ml-system-design-framework]] | The structured approach that keeps a 45-minute round on track | core |
| 2 | [[requirements-and-metrics-definition]] | Getting the problem framing right before any architecture talk | core |
| 3 | [[distributed-systems-basics]] | Baseline vocabulary — replication, partitioning, consensus | core |
| 4 | [[cap-theorem-and-consistency]] | Explains real tradeoffs behind storage/serving choices | intermediate |
| 5 | [[latency-and-throughput-budgets]] | Turns vague "make it fast" asks into concrete numbers | intermediate |
| 6 | [[scalability-patterns]] | Horizontal scaling, sharding, load balancing for ML systems | intermediate |
| 7 | [[caching-strategies]] | Cheap latency win, comes up in nearly every design | core |
| 8 | [[api-design-for-ml]] | Contract design between model-serving and consumers | core |
| 9 | [[training-serving-skew]] | The classic "why does prod differ from offline" failure | core |
| 10 | [[llm-system-design-framework]] | LLM-specific variant — cost, context, latency constraints | advanced |

## How it's tested per role
- **ML Engineer / MLOps Engineer**: get the deepest system-design grilling — expect a full end-to-end design (data → training → serving → monitoring) with follow-ups on failure modes.
- **AI Engineer / Agentic Engineer**: tested via the LLM-specific framework — cost per call, context budgets, latency versus quality tradeoffs replace classical throughput/replication questions.
- **Data Scientist**: lighter system-design bar; usually limited to "how would this model be deployed and monitored" rather than a full architecture.
- **FDE**: tested for pragmatic, constraint-aware design under a client's existing infra rather than greenfield architecture idealism.

## Question bank
See [[qbank-system-design]] for the drilled question set.

## Related domains
- [[moc-mlops]]
- [[moc-data-engineering]]
- [[moc-nlp-llm]]
- [[moc-rag]]
- [[moc-agents]]
