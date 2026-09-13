---
title: Agents — Map of Content
type: map
domain: agents
roles: [agentic-engineer, ai-engineer, fde]
updated: 2026-09-13
---

# Agents — Map of Content

## Why this domain is asked
Agentic systems are the newest and fastest-moving part of the Indian AI hiring market — "Agentic Engineer" is now a standalone title, and even generalist AI Engineer loops now expect a round on tool use, planning loops and multi-agent failure modes. Weight here is rising fastest of any domain in this vault.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[agent-fundamentals]] | What actually makes a system "agentic" vs a chained prompt | core |
| 2 | [[tool-calling-and-function-schemas]] | Mechanical foundation every agent framework builds on | core |
| 3 | [[react-and-reasoning-loops]] | The canonical reason-act-observe pattern, asked everywhere | core |
| 4 | [[planning-and-task-decomposition]] | How agents break down multi-step goals | intermediate |
| 5 | [[agent-memory]] | Short vs long-term memory design, a recurring design gap | intermediate |
| 6 | [[model-context-protocol]] | Emerging standard for tool/context interoperability | intermediate |
| 7 | [[agent-frameworks-landscape]] | Practical fluency across LangGraph/etc. tradeoffs | intermediate |
| 8 | [[multi-agent-systems]] | Orchestration patterns and coordination failure modes | advanced |
| 9 | [[agentic-rag]] | Retrieval as an agent-invoked tool rather than a fixed pipeline | advanced |
| 10 | [[computer-use-and-browser-agents]] | Newest frontier capability, increasingly asked conceptually | advanced |
| 11 | [[human-in-the-loop-patterns]] | Production safety valve for high-stakes agent actions | intermediate |
| 12 | [[agent-guardrails-and-safety]] | Preventing runaway loops, unsafe tool calls | intermediate |
| 13 | [[agent-cost-and-latency-optimization]] | Multi-call agent loops get expensive fast — a real design constraint | advanced |
| 14 | [[agent-evaluation]] | Hardest open problem in the space — how do you know it works | advanced |

## How it's tested per role
- **Agentic Engineer**: the primary domain of the interview — expect deep questions on planning loops, memory design, multi-agent coordination and evaluation methodology.
- **AI Engineer**: tested as an extension of LLM knowledge — tool calling and ReAct loops are now assumed baseline, multi-agent depth less so.
- **FDE**: tested for pragmatic judgment — when an agent is overkill versus a simple pipeline, and how to keep a client's agent from misbehaving in production.
- **ML Engineer / MLOps Engineer**: rarely a focus area, though MLOps increasingly intersects here via agent observability and cost monitoring.

## Question bank
See [[qbank-agents]] for the drilled question set.

## Related domains
- [[moc-nlp-llm]]
- [[moc-rag]]
- [[moc-system-design]]
- [[moc-mlops]]
