---
title: RAG — Map of Content
type: map
domain: rag
roles: [ai-engineer, agentic-engineer, fde, ml-engineer]
updated: 2026-09-13
---

# RAG — Map of Content

## Why this domain is asked
Retrieval-augmented generation is the most commonly *shipped* LLM pattern in Indian enterprise and GCC work right now, so interviewers use it to test whether a candidate has actually built something real versus only prompted a chatbot. Expect a dedicated system-design-style round for AI Engineer/FDE roles, often framed as "design a RAG system for X."

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[rag-overview]] | End-to-end mental model before any component deep dive | core |
| 2 | [[document-ingestion-and-parsing]] | Where most real-world RAG systems actually fail first | core |
| 3 | [[chunking-strategies]] | Highest-leverage design decision, heavily probed | core |
| 4 | [[embedding-models]] | Choosing and evaluating the retrieval representation | core |
| 5 | [[vector-databases]] | Index choice and tradeoffs for production retrieval | core |
| 6 | [[ann-algorithms-hnsw-ivf]] | Explains recall/latency tradeoffs behind vector DB choice | advanced |
| 7 | [[hybrid-search-bm25-vector]] | Practical fix for pure-vector-search failure modes | intermediate |
| 8 | [[query-rewriting-and-expansion]] | Handles the gap between user intent and indexed content | intermediate |
| 9 | [[reranking]] | Cheap, high-impact precision fix asked in most designs | intermediate |
| 10 | [[context-assembly-and-compression]] | Fitting retrieved evidence into a finite context budget | intermediate |
| 11 | [[rag-evaluation]] | How you prove retrieval quality, not just demo it | core |
| 12 | [[advanced-rag-patterns]] | Multi-hop, self-querying — separates senior candidates | advanced |
| 13 | [[graph-rag]] | Handles relationship-heavy corpora classic RAG misses | advanced |
| 14 | [[rag-failure-modes]] | The "what breaks in production" question every panel asks | core |

## How it's tested per role
- **AI Engineer**: full end-to-end design round — chunking strategy, embedding choice, evaluation methodology, and how to debug a specific failure mode.
- **Agentic Engineer**: RAG tested as one tool inside a larger agent loop — emphasis on when to retrieve, not just how.
- **FDE**: tested for judgment under client constraints — cost, data-freshness and latency tradeoffs over algorithmic depth.
- **ML Engineer**: tested lightly, mainly on the embedding-model and vector-index engineering pieces.

## Question bank
See [[qbank-rag]] for the drilled question set.

## Related domains
- [[moc-nlp-llm]]
- [[moc-agents]]
- [[moc-system-design]]
- [[moc-data-engineering]]
