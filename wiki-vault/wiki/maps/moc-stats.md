---
title: Statistics — Map of Content
type: map
domain: stats
roles: [data-scientist, ml-engineer, ai-engineer]
updated: 2026-09-13
---

# Statistics — Map of Content

## Why this domain is asked
Statistics is where Data Scientist interviews in India differentiate signal from noise: A/B testing design, p-value interpretation and causal-inference traps are asked in nearly every product-DS loop, because they map directly to day-to-day experimentation work. Expect a dedicated 30-45 minute round, plus stats reasoning folded into case studies and take-homes.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[descriptive-statistics]] | Baseline vocabulary — mean/median/skew reasoning | core |
| 2 | [[sampling-and-sampling-distributions]] | Foundation for every inference argument that follows | core |
| 3 | [[central-limit-theorem]] | Justifies why normal-approximation tricks work at all | core |
| 4 | [[confidence-intervals]] | Communicating uncertainty to stakeholders | core |
| 5 | [[maximum-likelihood-estimation]] | Connects probability theory to how models are actually fit | core |
| 6 | [[hypothesis-testing]] | The mechanical backbone of every "is this real" question | core |
| 7 | [[p-values-and-significance]] | Most misunderstood term in interviews — must be airtight | core |
| 8 | [[type-i-and-type-ii-errors]] | Framing tradeoffs in decision-making under uncertainty | core |
| 9 | [[statistical-power-and-sample-size]] | "How long should we run this experiment" questions | intermediate |
| 10 | [[common-statistical-tests]] | Picking the right test — t-test vs chi-square vs Mann-Whitney | intermediate |
| 11 | [[multiple-testing-correction]] | Guardrail-metric and multi-variant experiment traps | intermediate |
| 12 | [[resampling-bootstrap-and-permutation]] | Modern, assumption-light alternative to classical tests | intermediate |
| 13 | [[ab-testing-design]] | The single most-asked DS stats topic end to end | core |
| 14 | [[ab-testing-pitfalls]] | Separates senior candidates — novelty effects, SRM, peeking | advanced |
| 15 | [[causal-inference-basics]] | "Correlation isn't causation" pushed into a real answer | advanced |
| 16 | [[bayesian-inference-basics]] | Alternative inferential paradigm, priors vs frequentist framing | advanced |

## How it's tested per role
- **Data Scientist**: the hardest and most frequent stats grilling of any role — full A/B test design end to end, causal inference edge cases, defending a p-value in front of a skeptical panel.
- **ML Engineer**: lighter touch — mostly needs to reason about evaluation metric stability and whether an observed model-quality change is statistically meaningful.
- **AI Engineer**: mainly needed for offline/online LLM evaluation — is a prompt change actually better, or noise.
- **MLOps / Agentic / FDE**: rarely tested directly beyond "how would you know if this metric change is real."

## Question bank
See [[qbank-stats]] for the drilled question set.

## Related domains
- [[moc-maths]]
- [[moc-classical-ml]]
- [[moc-system-design]]
- [[moc-behavioral]]
