# CLAUDE.md — Wiki Schema

> This is the **schema** for this knowledge base. It tells an LLM how the wiki is
> structured, what the conventions are, and what workflow to follow when ingesting
> sources, answering questions, or maintaining the wiki.
> Architecture follows Karpathy's LLM-wiki pattern: **raw sources → wiki → schema**,
> with `ingest` / `query` / `lint` as the three operations.

---

## 1. Purpose

A compounding, LLM-maintained knowledge base for **cracking interviews in India at ~5 years
of experience** for these roles:

| Role | Slug |
|---|---|
| Data Scientist | `data-scientist` |
| Machine Learning Engineer | `ml-engineer` |
| AI Engineer (LLM/product) | `ai-engineer` |
| MLOps Engineer | `mlops-engineer` |
| AI / Agentic Engineer | `agentic-engineer` |
| Forward Deployed Engineer | `fde` |

Coverage target: **all basics of every topic these roles require, taken to moderately
advanced depth**, with maths, code, diagrams and flowcharts wherever they earn their place.

Owner context (use it — examples should feel native, not generic): ML Engineer on the
Data+AI team at DataNimbus. Daily stack: **Databricks / PySpark, medallion architecture,
MLflow, XGBoost, Unity Catalog**. Where a concept can be illustrated with that stack
without distorting it, do so — the notes double as talking points for his own experience.

---

## 2. Three layers

```
raw/        immutable sources you curate (papers, JDs, transcripts, blog dumps, screenshots)
wiki/       LLM-generated + LLM-maintained markdown pages (the compounding artifact)
CLAUDE.md   this schema — conventions + workflows. Co-evolve it as the domain teaches you things.
```

Plus two spine files at the vault root:

- `index.md` — content-oriented catalog. Every page is listed, grouped by category. Read it **first** when answering a query.
- `log.md` — chronological record. Append-only. One entry per ingest / query-filed-back / lint.

---

## 3. Directory layout

```
knowledge_claude/                 ← Obsidian vault root
├── CLAUDE.md                     ← this file (the schema)
├── index.md                      ← catalog of every page
├── log.md                        ← chronological operations record
├── START-HERE.md                 ← human entry point: how to study with this vault
├── raw/                          ← immutable sources. Never edited by the LLM.
│   ├── README.md
│   └── assets/                   ← images, PDFs, screenshots referenced by raw notes
├── _templates/                   ← Obsidian Templates plugin
│   ├── concept.md
│   ├── analysis.md
│   ├── question-bank.md
│   └── source-summary.md
└── wiki/
    ├── maps/                     ← one Map of Content per domain. The curriculum spine.
    ├── concepts/                 ← one page per concept, grouped into a folder per domain.
    │   ├── maths/
    │   ├── stats/
    │   ├── programming/
    │   ├── sql/
    │   ├── classical-ml/
    │   ├── deep-learning/
    │   ├── nlp-llm/
    │   ├── rag/
    │   ├── agents/
    │   ├── mlops/
    │   ├── data-engineering/
    │   ├── system-design/
    │   └── behavioral/
    ├── entities/                 ← tools, libraries, platforms, papers, algorithms-as-products
    ├── roles/                    ← one page per target role: scope, rounds, readiness checklist
    ├── analyses/                 ← comparisons, syntheses, ML/LLM system-design case studies
    ├── questions/                ← question banks (per domain, per role) + answered queries filed back
    ├── drills/                   ← flashcard decks, rapid-fire, SQL/coding problem sets
    └── summaries/                ← one page per ingested raw source
```

**`wiki/concepts/` is grouped one folder per domain**, named exactly after the `domain:` value
below — a concept's folder and its frontmatter `domain:` field must always agree. This makes
browsing-by-topic possible directly in the filesystem/Obsidian file tree, not just via MOCs.
The folder is a convenience, not the source of truth: **wikilinks are still by filename only**
(`[[bias-variance-tradeoff]]`, never `[[classical-ml/bias-variance-tradeoff]]`) so moving a page
between domain folders later — because you decide it fits a different domain — never breaks an
inbound link; just `git mv` it into its new folder and update its `domain:` field to match.

---

## 4. Naming conventions

- Filenames: **kebab-case**, lowercase, `.md`. `bias-variance-tradeoff.md`, `lora-and-qlora.md`.
- One concept per file. If a title needs "and" between two unrelated ideas, it is two files.
- Prefer the term an interviewer would say: `gradient-boosting` not `boosting-algorithms-overview`.
- Acronyms stay as spoken: `rag-evaluation.md`, `shap-values.md`, `cap-theorem.md`.
- Maps: `moc-<domain>.md`. Roles: `role-<slug>.md`. Question banks: `qbank-<topic>.md`.
  Case studies: `case-<system>.md`. Comparisons: `vs-<a>-<b>.md`. Source summaries: `src-<slug>.md`.
- Wikilinks are by filename only — `[[bias-variance-tradeoff]]` — never by path. Obsidian resolves it.

---

## 5. Page anatomy

Every page opens with YAML frontmatter. Fields:

```yaml
---
title: Bias–Variance Tradeoff
type: concept            # concept | entity | map | role | analysis | case | qbank | drill | source
domain: classical-ml     # see domain list below
roles: [data-scientist, ml-engineer]   # which target roles are asked this
difficulty: core         # core | intermediate | advanced
frequency: high          # how often it shows up in interviews: high | medium | low
status: seed             # seed | drafted | reviewed | mastered   ← the human moves this along
tags: [generalization, model-selection]
updated: 2026-09-11
sources: []              # wikilinks to wiki/summaries pages, or URLs
---
```

**Domains** (the `domain:` value, and the set of MOCs):
`maths` · `stats` · `programming` · `sql` · `classical-ml` · `deep-learning` · `nlp-llm` ·
`rag` · `agents` · `mlops` · `data-engineering` · `system-design` · `behavioral`

### Concept page sections (in this order, omit what genuinely does not apply)

1. `## TL;DR` — 2–4 lines. The answer you would give if you had 20 seconds.
2. `## Intuition` — the picture in your head before any notation. Analogy allowed here, nowhere else.
3. `## The maths` — LaTeX. Define every symbol. Derive, do not assert, when the derivation is itself the interview question.
4. `## Diagram` — Mermaid (`flowchart`, `sequenceDiagram`, `graph`) when structure or flow matters.
5. `## Code` — runnable Python. Small, self-contained, no hand-waving. PySpark / MLflow / Databricks flavour where natural.
6. `## In practice` — when to reach for it, defaults that actually work, what breaks at scale, cost/latency.
7. `## Interview angle` — the questions actually asked, each with a crisp model answer. Include at least one follow-up chain.
8. `## Traps` — the things that get candidates rejected. Common wrong answers, stated as wrong.
9. `## Flashcards` — spaced-repetition lines (see §7).
10. `## Related` — wikilinks out. Minimum 3. This is what makes the vault a graph rather than a folder.

### Other page types

- **map** (`wiki/maps/moc-*.md`): frontmatter + a short "what this domain is and why it's asked"
  + an ordered curriculum table (`| # | Page | Why it matters | Difficulty |`) + "how this domain is
  tested per role" + links to the relevant question banks.
- **role** (`wiki/roles/role-*.md`): what the role actually does day to day in India, typical
  interview loop round by round, the JD vocabulary, which domains carry the most weight, a
  readiness checklist, and the 20 pages to revise the night before.
- **analysis / case**: for system design — `Requirements → Metrics → Data → Features → Model →
  Serving → Monitoring → Failure modes → Tradeoffs discussed out loud`.
- **qbank**: `### Q<n>. <question>` then `**Answer.**` then `**Follow-ups.**` Grouped by sub-topic, easy → hard.
- **source** (`wiki/summaries/src-*.md`): what the source is, key takeaways, which wiki pages it changed,
  and what it contradicts.

---

## 6. Obsidian compatibility rules

These are hard requirements — the vault is read in Obsidian.

- **Maths**: inline `$\sigma(z)$`, block `$$ ... $$`. No `\[ ... \]`, no `\begin{align}` outside `$$`.
  Escape underscores inside text, never inside `$`. Prefer `\mathbb{E}`, `\hat{y}`, `\mathcal{L}`.
- **Diagrams**: fenced ` ```mermaid ` blocks. Obsidian renders them natively — do not add a library.
  Keep node labels short; quote labels containing `(`, `)` or `:`  → `A["Train (offline)"]`.
- **Links**: `[[filename]]` or `[[filename|display text]]`. Never `[[folder/filename]]`.
  Never link to a page that does not exist yet unless you also add it to §"Gaps" in `index.md`.
- **Callouts**: `> [!tip]`, `> [!warning]`, `> [!example]`, `> [!question]` — used sparingly.
- **Tags**: in frontmatter only, not inline `#tags`, so the graph stays clean.
- **Code fences**: always language-tagged (` ```python `, ` ```sql `, ` ```bash `).
- No HTML. No `<br>`. Tables must be pure markdown.
- Filenames contain no spaces, no `:`, `/`, `?`, `*`, `|`, `"` — Windows-safe.

---

## 7. Flashcard format

Spaced Repetition plugin syntax. Inline cards use `::`, reversed cards `:::`.

```
What does high variance look like on a learning curve?::Train error low, val error high, gap does not close with more data.
```

Rules: 4–10 cards per concept page. One fact per card. No card longer than two lines.
Never put LaTeX-heavy derivations on a card — put the *trigger* on the card and link the page.

**Critical:** the plugin only scans notes carrying the `flashcards` tag (its `flashcardTags`
setting is `#flashcards`, and frontmatter `tags:` entries count as vault tags). Any page with a
`## Flashcards` section **must** include `flashcards` in its frontmatter `tags:` list, or its
cards are invisible to review — silently, with no error.

---

## 8. Operations

### `ingest` — a new source arrives

1. Human drops a file into `raw/` (or gives a URL). Sources are **immutable**; never edit them.
2. Read it. Discuss the key takeaways with the human before writing.
3. Write `wiki/summaries/src-<slug>.md`.
4. Update **every** wiki page the source touches — a real source usually touches 10–15 pages.
   Prefer updating an existing page over creating a near-duplicate.
5. Create pages for concepts the source introduces that have no page yet, filed under
   `wiki/concepts/<domain>/` matching the new page's `domain:` field.
6. Update `index.md` (add new pages, move anything out of the Gaps list).
7. Append to `log.md`: `## [YYYY-MM-DD] ingest | <source title>` + one line per page touched.

### `query` — answer a question against the wiki

1. Read `index.md` first to find candidate pages. Then read those pages, not the whole vault.
2. Synthesise an answer **with citations** — cite as wikilinks to the pages used.
3. If the answer is durable and reusable, file it back: new page in `wiki/concepts/<domain>/`,
   `wiki/analyses/` or `wiki/questions/`, then update `index.md` and append to `log.md`
   (`## [YYYY-MM-DD] query | <question>`).
4. If the wiki could not answer it, that is a **gap** — record it in `index.md` § Gaps.

### `lint` — periodic health check

Run every ~20 ingests, or on request. Check for:

- contradictions between pages (same claim, two answers)
- stale claims superseded by a newer source (model names, library APIs, pricing, benchmark numbers)
- orphan pages — no inbound wikilinks
- broken wikilinks — link target file does not exist
- concepts mentioned repeatedly across pages but having no page of their own
- missing cross-references (`## Related` with fewer than 3 links)
- frontmatter drift: missing fields, invalid `domain`, `status` never advanced
- pages absent from `index.md`
- coverage gaps per role: a `role-*.md` checklist item with no page behind it

Write findings to `log.md` under `## [YYYY-MM-DD] lint` and fix what is mechanical.

---

## 9. Writing standards

- Write for someone with 5 years of experience. Skip the undergraduate throat-clearing.
- **Show the derivation** when the derivation is the question (backprop, bias–variance decomposition,
  the normal equation, why cross-entropy pairs with softmax, why attention scales by $1/\sqrt{d_k}$).
- Every code block must be one a candidate could type on a whiteboard or in a shared editor, and
  must run. No pseudo-APIs, no invented library functions.
- Numbers age. Prefer "a 7B model at fp16 needs ~14 GB for weights" (a rule you can re-derive)
  over "model X scores 82.4 on benchmark Y".
- State tradeoffs as tradeoffs. An interview answer that has no downside named is a weak answer.
- Indian-market framing where it changes the answer: round structures, service vs product vs GCC
  expectations, take-home norms, what "5 years" is assumed to mean.
- Never pad. A page that says everything in 120 lines beats one that says it in 400.

---

## 10. Conventions the human maintains

- `status:` is the human's field. The LLM sets `seed`/`drafted`; the human moves pages to
  `reviewed` and `mastered`. Study progress is a frontmatter query, not a separate tracker.
- Anything in `raw/` is sacred. Anything in `wiki/` is disposable and regenerable.
- The vault is a git repository — version history comes free, so the LLM may rewrite boldly.
