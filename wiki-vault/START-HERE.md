---
title: Start Here
type: map
domain: meta
updated: 2026-09-13
---

# Start Here

## What this vault is

An **interview-preparation knowledge base** for Data Scientist / ML Engineer / AI Engineer /
MLOps Engineer / Agentic Engineer / Forward Deployed Engineer roles in India, pitched at ~5 years
of experience. Owner context: an ML Engineer on a Databricks/PySpark/medallion-architecture/
MLflow/XGBoost/Unity Catalog stack — examples across the vault lean on that stack where it fits
naturally.

It's built on the [[CLAUDE|LLM-wiki pattern]]: a small set of immutable raw sources in `raw/`, a
large body of LLM-generated-and-maintained wiki pages in `wiki/`, and a schema ([[CLAUDE]]) that
tells the LLM how to keep the two in sync and how every page should look. You bring sources and
questions; the LLM does the writing and bookkeeping — you never hand-edit `wiki/` pages to keep
them "neat," you tell Claude to `ingest`, `query`, or `lint` and it rewrites what's needed.

**Current size**: 228 concept pages across 13 domains, 13 domain maps, 6 role pages, 18 tool/
platform entity pages, 17 system-design analyses and comparisons, 14 question banks, and 7 drills/
study plans — 303 pages total, all cataloged in [[index]].

## How it's organized

| Folder | What's in it | Count |
|---|---|---|
| `wiki/maps/` | One Map of Content per domain — **the curriculum spine**. Each has an ordered "read these in this order" table. | 13 |
| `wiki/roles/` | One page per target role: what the job actually does, the interview loop round by round, domain weighting, a readiness checklist. | 6 |
| `wiki/concepts/` | The bulk of the vault. Flat, one idea per page: TL;DR → Intuition → Maths → Diagram → Code → In practice → Interview angle → Traps → Flashcards → Related. | 228 |
| `wiki/entities/` | Tools, libraries, platforms as products (XGBoost, Spark, Kubernetes, MLflow, LangChain, etc.) — "what it is / when to use it vs alternatives," not interview theory. | 18 |
| `wiki/analyses/` | System-design case studies (`case-*`) and head-to-head comparisons (`vs-*`). | 17 |
| `wiki/questions/` | Question banks (`qbank-*`) — one per domain, plus a cross-domain rapid-fire bank. | 14 |
| `wiki/drills/` | Hands-on problem sets (SQL, Python, PySpark, ML-from-scratch), a rapid-fire case-prompt list, and two study schedules (12-week, 7-day sprint). | 7 |
| `wiki/summaries/` | One page per raw source you `ingest` — starts empty, fills over time. | grows |
| `raw/` | Immutable source material you drop in (JDs, papers, transcripts). Never edited by the LLM. | you control |
| `index.md` | The catalog — every page, grouped, with a one-line description. **Read this first when hunting for something.** | 1 |
| `log.md` | Chronological record of every ingest / query / lint, append-only. | 1 |
| `CLAUDE.md` | The schema itself — page anatomy, naming rules, Obsidian constraints, the three operations below. Edit it when you learn something about how *you* want to study; the whole vault follows it. | 1 |

## The four things you actually do here

| You want to… | Say this to Claude |
|---|---|
| **Study** | Nothing to say — just open a page. Start from a role in `wiki/roles/`, or a domain map in `wiki/maps/`. |
| **Add a new source** | Drop the file in `raw/`, then: **"ingest raw/\<file>"** — it gets summarized, and every existing page it touches gets updated (never duplicated). |
| **Ask something** | **"query: \<your question>"** — Claude reads [[index]] first, answers with citations to the pages used, and files a durable answer back into the vault if it's reusable. |
| **Clean up** | **"lint the wiki"** — checks for contradictions, stale claims, broken wikilinks, orphan pages, and missing cross-references; fixes what's mechanical, logs the rest. |

## How to actually study

1. **Pick your target role** in `wiki/roles/`. Read it once — it sets which domains carry the most
   weight for you and gives you the interview-loop shape to expect.
2. **Pick a track based on your runway:**
   - Weeks of runway → [[plan-12-week]], a full curriculum sequenced across all 13 domain maps.
   - An interview next week → [[plan-7-day-sprint]], a triage plan that names what to explicitly skip.
3. **Work a domain map (`moc-*`) top to bottom.** For each concept page: read it → close it →
   re-derive the maths on paper → run the code → *then* mark `status: reviewed` in its frontmatter.
4. **Drill every session, don't just re-read.** Run the `## Flashcards` section (Spaced Repetition
   plugin) and, once a domain is `reviewed`, do its `wiki/questions/qbank-*.md` cold — write your
   answer before revealing theirs. Anything you fumble goes back to `status: drafted`.
5. **Weekly, run one system design out loud.** Pick a prompt from [[drill-case-prompts]], give
   yourself 45 minutes, then check it against the matching `wiki/analyses/case-*.md`.
6. **Ask Claude to `lint` monthly** so gaps surface before an interviewer finds them.

> [!tip]
> The single highest-return habit: after reading a page, write its TL;DR from memory in your own
> words. If you can't, you haven't learned it — you've recognised it.

## Obsidian setup

| Plugin | Why |
|---|---|
| **Spaced Repetition** | The `## Flashcards` section on every concept page uses its `::` syntax. |
| **Dataview** | Turns frontmatter into a live progress tracker — queries below. |
| **Templater** or core **Templates** | `_templates/` holds the page skeletons for adding new pages by hand. |
| **Excalidraw** *(optional)* | For redrawing a page's diagram yourself — often the fastest way to actually learn it. |

Core plugins to switch on: Graph view, Backlinks, Outline, Search, Tags.
Settings → Appearance → **MathJax** rendering (on by default) and leave **Mermaid** enabled — every
diagram in the vault is a native ` ```mermaid ` fence, no extra plugin needed.

### Track progress with Dataview

What's left, ranked by what to do first (high-frequency, unreviewed, easiest first):

````markdown
```dataview
TABLE difficulty, frequency, status
FROM "wiki/concepts"
WHERE status != "mastered" AND frequency = "high"
SORT difficulty ASC
```
````

A per-domain burndown, so you can see which domain still needs the most work:

````markdown
```dataview
TABLE length(rows) AS pages
FROM "wiki/concepts"
GROUP BY domain
```
````

## Related
[[CLAUDE]] · [[index]] · [[log]]
