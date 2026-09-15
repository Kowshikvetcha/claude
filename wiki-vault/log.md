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

## [2026-09-13] build | Vault completed — gaps closed, index generated, lint passed

Resumed from `HANDOFF.md` (now deleted — its gaps are closed and nothing should link to a
one-time handoff note). Dispatched parallel writers to close every remaining gap from the
2026-09-11 bootstrap:

- **Question banks (12 written):** [[qbank-agents]], [[qbank-behavioral]], [[qbank-classical-ml]],
  [[qbank-data-engineering]], [[qbank-deep-learning]], [[qbank-mlops]], [[qbank-nlp-llm]],
  [[qbank-programming]], [[qbank-rag]], [[qbank-rapid-fire]], [[qbank-sql]],
  [[qbank-system-design]] — each 15-18 questions across Warm-up/Core/Hard, cross-linked to that
  domain's concept pages. `qbank-rapid-fire` uses the special all-domain lightning-round shape
  (short Q/A per domain, no derivations).
- **Analyses (8 written):** [[case-agentic-support-automation]], [[case-churn-prediction]],
  [[case-document-extraction-pipeline]], [[case-llm-cost-reduction]], [[case-ml-platform-design]],
  [[case-rag-assistant]], [[case-realtime-feature-pipeline]], [[vs-vector-db-options]] — each a
  genuinely distinct architecture (agent orchestration, supervised churn scoring, document-AI
  routing, cost-optimization/cascade, ML platform infra, retrieval-and-grounding RAG, streaming
  feature pipeline, vector-DB comparison), not a template with nouns swapped.
- **Drills/plans (3 written):** [[drill-case-prompts]], [[plan-12-week]], [[plan-7-day-sprint]] —
  written before the qbanks/analyses above landed, then patched afterward to wire in real links
  where they'd left "pending" placeholders.
- **Concept gap closed:** [[kubernetes-for-ml]] — referenced from 5 pages ([[distributed-training]],
  [[infrastructure-as-code]], [[kubernetes]], [[moc-mlops]], [[role-mlops-engineer]]) but had no
  page until this session; written and cross-linked.
- **Orphans fixed:** [[dbt]] (linked from [[medallion-architecture]]'s tooling note), [[fastapi]]
  (linked from [[model-serving-patterns]]'s sidecar-pattern prose), [[role-agentic-engineer]]
  (linked from [[qbank-agents]]'s Related section).
- **[[index]] generated** last, per [[CLAUDE]] §3/§8 — scripted a walk of `wiki/**/*.md`
  frontmatter grouped by category (maps, roles, entities, concepts-by-domain, analyses, questions,
  drills), 303 content pages plus the index itself.
- **Lint pass** (per [[CLAUDE]] §8): zero broken wikilinks, zero orphan pages, zero missing
  frontmatter across all content pages, all `moc-*` curriculum entries have a page behind them.
  Gaps section in [[index]] is empty.
- `HANDOFF.md` deleted per its own closing instruction now that the gaps it tracked are closed.

---

## [2026-09-15] build | wiki/concepts/ reorganized into per-domain folders

The 228 concept pages moved from a flat `wiki/concepts/*.md` layout into one subfolder per domain
(`wiki/concepts/maths/`, `wiki/concepts/classical-ml/`, `wiki/concepts/nlp-llm/`, etc. — the same
13 domains already used everywhere else in the vault), so a topic can be browsed directly in the
file tree instead of only through a MOC or a Dataview query. Every file moved via `git mv` with no
renames, so no wikilink needed to change — Obsidian resolves `[[filename]]` by name, not path.

- Updated [[CLAUDE]] §3 (directory layout no longer says "flat on purpose"; documents the
  folder-per-domain convention and that a page's folder must match its `domain:` field) and §8
  (`ingest`/`query` now say to file new concept pages under `wiki/concepts/<domain>/`).
- Updated [[START-HERE]]'s layout table to describe the folder-per-domain structure.
- Regenerated [[index]] — unaffected in content (grouping was already keyed off the `domain:`
  frontmatter field, not the folder), only the generator script's folder-detection logic changed.
- Full `lint` pass after the move: zero broken wikilinks, zero orphans, zero missing frontmatter —
  confirms the move was link-safe.

---

<!-- Newest entries go at the BOTTOM. Append, never rewrite history. -->
