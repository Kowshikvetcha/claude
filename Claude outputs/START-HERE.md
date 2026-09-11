---
title: Start Here
type: map
domain: meta
updated: 2026-09-11
---

# Start Here

This vault is an **interview-preparation wiki** for Data Science / ML Engineer / AI Engineer /
MLOps Engineer / Agentic Engineer / Forward Deployed Engineer roles in India, pitched at
~5 years of experience.

It is built on the [[CLAUDE|LLM-wiki pattern]]: a small set of immutable raw sources, a large
body of generated-and-maintained wiki pages, and a schema that tells the LLM how to keep the
two in sync. You do the sourcing and the asking; the LLM does the bookkeeping.

## The three things you actually touch

| You want to… | Do this |
|---|---|
| Study | Open a Map of Content in `wiki/maps/` and work down the curriculum table |
| Add material | Drop the file in `raw/`, then tell Claude **"ingest raw/<file>"** |
| Ask something | Tell Claude **"query: <your question>"** — it reads [[index]] first, answers with citations, and files good answers back |
| Clean up | Tell Claude **"lint the wiki"** |

## Layout

- [[index]] — the catalog. Every page, grouped. Start here when hunting.
- [[log]] — what has been ingested, asked and fixed, in date order.
- [[CLAUDE]] — the schema. Conventions, page anatomy, workflows. Edit this when you learn
  something about how *you* study; the whole vault follows it.
- `wiki/maps/` — one MOC per domain. **This is the curriculum.**
- `wiki/roles/` — one page per target role: the interview loop, the weighting, the readiness checklist.
- `wiki/concepts/` — the bulk. Flat, one idea per page.
- `wiki/analyses/` — system design case studies and head-to-head comparisons.
- `wiki/questions/` — question banks.
- `wiki/drills/` — flashcards, rapid-fire, SQL and coding sets.

## Recommended Obsidian plugins

| Plugin | Why |
|---|---|
| **Spaced Repetition** | The `## Flashcards` section on every concept page uses its `::` syntax |
| **Dataview** | Turns frontmatter into progress tracking (see below) |
| **Templater** or core **Templates** | `_templates/` holds the page skeletons |
| **Excalidraw** *(optional)* | For drawing your own versions of the diagrams — the best way to learn them |

Core plugins to switch on: Graph view, Backlinks, Outline, Search, Tags.
Settings → Appearance → enable **MathJax**-rendered LaTeX (on by default) and leave
**Mermaid** enabled.

## Track progress with Dataview

Paste this into a note to see what is left:

````markdown
```dataview
TABLE difficulty, frequency, status
FROM "wiki/concepts"
WHERE status != "mastered" AND frequency = "high"
SORT difficulty ASC
```
````

And a per-domain burndown:

````markdown
```dataview
TABLE length(rows) AS pages
FROM "wiki/concepts"
GROUP BY domain
```
````

## How to actually study

1. Pick a role from `wiki/roles/`. Read its page. That sets the weighting.
2. Work its highest-weight domain MOC top to bottom. For each page: read → close it → rederive the
   maths on paper → run the code → then mark `status: reviewed`.
3. Every session, drill the flashcards. That is where retention comes from, not rereading.
4. Once a domain is `reviewed`, do the matching `wiki/questions/qbank-*.md` cold — write answers
   before revealing. Anything you fumble goes back to `status: drafted`.
5. Weekly, do one case from `wiki/analyses/case-*.md` out loud, on a whiteboard, in 45 minutes.
6. Ask Claude to `lint` monthly so the gaps surface before an interviewer finds them.

> [!tip]
> The single highest-return habit: after reading a page, write its TL;DR from memory in your own
> words. If you can't, you haven't learned it — you've recognised it.
