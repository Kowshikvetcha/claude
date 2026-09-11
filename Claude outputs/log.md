---
title: Log
type: map
domain: meta
updated: 2026-09-11
---

# Log

Chronological, append-only record of every `ingest`, filed-back `query`, and `lint`.
Entry prefix is fixed so the file stays greppable:

```
## [YYYY-MM-DD] <ingest|query|lint|build> | <title>
```

---

## [2026-09-11] build | Vault bootstrapped

- Architecture adopted: Karpathy LLM-wiki pattern (raw → wiki → schema, with ingest/query/lint).
- Wrote [[CLAUDE]] (schema), [[START-HERE]], [[index]], this log, `raw/README`, `_templates/`.
- Scope set: DS / MLE / AIE / MLOps / Agentic / FDE roles, India market, ~5 years experience.
- Note format fixed: TL;DR → Intuition → Maths → Diagram → Code → In practice → Interview angle
  → Traps → Flashcards → Related.
- Obsidian constraints encoded in schema §6 (MathJax `$…$`, native Mermaid fences, flat wikilinks,
  Windows-safe filenames).
- Generated the domain MOCs, role pages, concept corpus, analyses, question banks and drills.
  See [[index]] for the full catalog.

---

<!-- Newest entries go at the BOTTOM. Append, never rewrite history. -->
