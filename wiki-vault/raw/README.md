---
title: raw/ — Source Collection
type: map
domain: meta
updated: 2026-09-11
---

# `raw/` — the immutable source collection

Everything here is **input**. Nothing here is ever edited, reformatted or deleted by the LLM.
If a source is wrong, you add a newer source; you do not rewrite the old one.

## What belongs here

- Job descriptions you are actually targeting (the single highest-value source type — it tells the
  wiki what to weight)
- Interview transcripts, recruiter screening notes, your own post-mortems after a round
- Papers and blog posts (LoRA, Attention Is All You Need, RAG survey, DPO, …)
- Course notes, book chapters, screenshots of slides → `raw/assets/`
- Take-home assignments and the feedback you got on them
- Your own project write-ups — TP Logic, the RAG pipeline, the DLT pipelines — so the wiki can
  turn them into STAR stories and system-design answers

## Naming

`YYYY-MM-DD-short-slug.<ext>` — e.g. `2026-09-14-jd-flipkart-mle-3.pdf`,
`2026-09-20-round2-feedback-phonepe.md`.

Images and PDFs referenced from a note go in `raw/assets/` with the same slug.

## How to ingest

Tell Claude:

```
ingest raw/2026-09-14-jd-flipkart-mle-3.pdf
```

It will read the source, talk through the takeaways with you, write
`wiki/summaries/src-<slug>.md`, update every affected wiki page, update [[index]], and append to
[[log]]. Ingest **one source at a time** and stay in the loop — read the summary, check the
diffs, tell it what to emphasise.

> [!warning]
> Do not dump twenty sources at once. The value of this pattern comes from you reading the
> summaries and steering. Batch-ingesting produces a wiki nobody has read.
