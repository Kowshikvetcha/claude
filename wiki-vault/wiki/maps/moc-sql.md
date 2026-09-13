---
title: SQL — Map of Content
type: map
domain: sql
roles: [data-scientist, ml-engineer, mlops-engineer, fde]
updated: 2026-09-13
---

# SQL — Map of Content

## Why this domain is asked
SQL is the one universal filter across Indian product companies, GCCs and service firms alike — nearly every DS/analytics-adjacent role has a dedicated SQL round, and it recurs inside data-engineering and system-design conversations too. It carries real weight: a shaky window-function answer at 5 YOE is a bigger red flag than a shaky deep-learning answer.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[sql-fundamentals]] | Baseline SELECT/WHERE/GROUP BY fluency, execution order | core |
| 2 | [[joins-deep-dive]] | Most common source of subtly wrong query results | core |
| 3 | [[aggregations-and-grouping]] | HAVING vs WHERE, grouping sets, common metric rollups | core |
| 4 | [[subqueries-and-ctes]] | Readable, composable query structure for complex asks | core |
| 5 | [[window-functions]] | The single highest-signal SQL topic in interviews | intermediate |
| 6 | [[sql-analytics-patterns]] | Funnels, retention, cohort — recurring "business" SQL asks | intermediate |
| 7 | [[sql-query-optimization]] | Indexes, execution plans — separates senior from mid-level | advanced |

## How it's tested per role
- **Data Scientist**: heaviest SQL bar — live query-writing round plus analytics-pattern case questions (funnels, cohorts, retention).
- **ML Engineer / MLOps Engineer**: expected to be fluent for feature pipelines and debugging data issues, but rarely tested with an isolated SQL-only round; optimization questions matter more here.
- **FDE**: SQL is a daily tool for client data exploration — expect fast, correct query-writing under time pressure rather than deep optimization theory.
- **AI Engineer / Agentic Engineer**: lightest bar; SQL shows up mainly as a tool an agent calls, not a skill tested directly.

## Question bank
See [[qbank-sql]] for the drilled question set.

## Related domains
- [[moc-data-engineering]]
- [[moc-programming]]
- [[moc-system-design]]
