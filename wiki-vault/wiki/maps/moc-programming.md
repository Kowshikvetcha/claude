---
title: Programming — Map of Content
type: map
domain: programming
roles: [data-scientist, ml-engineer, ai-engineer, mlops-engineer, agentic-engineer, fde]
updated: 2026-09-13
---

# Programming — Map of Content

## Why this domain is asked
Every one of these roles has at least one live-coding round, and at 5 years' experience the bar isn't "can you code" but "do you write production-grade Python and reason about complexity without prompting." Expect 1-2 rounds of DSA-flavoured coding plus a separate "write clean, testable code" evaluation baked into take-homes.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[python-data-model-and-idioms]] | Fluency here separates "uses Python" from "knows Python" | core |
| 2 | [[numpy-essentials]] | Vectorized thinking underlies every ML codebase | core |
| 3 | [[pandas-essentials]] | Daily-driver tool for DS/MLE data wrangling questions | core |
| 4 | [[big-o-and-complexity]] | Required vocabulary before any coding round | core |
| 5 | [[hashing-and-dictionaries]] | Most common building block in coding-interview solutions | core |
| 6 | [[arrays-and-strings-patterns]] | Bread-and-butter pattern set for easy/medium problems | core |
| 7 | [[trees-and-graphs-basics]] | Needed for medium-hard problems and system-design whiteboards | intermediate |
| 8 | [[dynamic-programming-patterns]] | The pattern most candidates freeze on — worth deliberate practice | advanced |
| 9 | [[python-performance-and-memory]] | Explains why a pandas pipeline is slow and how to fix it | intermediate |
| 10 | [[oop-and-design-patterns-for-ml]] | Signals production maturity in system-design and take-homes | intermediate |
| 11 | [[testing-python-code]] | Increasingly checked explicitly in senior ML hiring | intermediate |
| 12 | [[coding-interview-strategy]] | How to run the round itself, not just solve the problem | core |

## How it's tested per role
- **ML Engineer / MLOps Engineer**: hardest coding bar — expect medium LeetCode-style problems plus a "productionize this function" exercise (typing, tests, error handling).
- **Data Scientist**: coding is usually pandas/SQL-flavoured rather than pure DSA; less emphasis on trees/DP, more on correct vectorized data manipulation.
- **AI Engineer / Agentic Engineer**: coding rounds increasingly focus on API/orchestration code (calling LLMs, parsing structured output) over classic algorithms, but a DSA screen is still common at product companies.
- **FDE**: coding bar is broad but shallow — must ship end-to-end scripts fast and correctly, not necessarily solve the hardest DP problem.

## Question bank
See [[qbank-programming]] for the drilled question set.

## Related domains
- [[moc-sql]]
- [[moc-data-engineering]]
- [[moc-system-design]]
- [[moc-classical-ml]]
