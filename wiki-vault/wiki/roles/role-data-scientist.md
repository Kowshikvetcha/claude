---
title: Data Scientist
type: role
domain: meta
roles: [data-scientist]
updated: 2026-09-13
---

# Data Scientist

## What this role actually does
At ~5 years experience in India, "Data Scientist" spans a wide range depending on the company type. In **product companies** (e-commerce, fintech, food-tech, edtech), the job is closer to applied statistics plus analytics engineering: designing and reading A/B tests, building churn/pricing/fraud models that are mostly gradient-boosted trees, and translating metric movements into business narratives for a PM or leadership review. Deep learning shows up occasionally (ranking, embeddings) but is rarely the daily tool. In **GCCs** (Global Capability Centres of banks, retailers, manufacturers), the role often skews more analytical and less experimental — you inherit a well-defined KPI, build a model against it, and spend real time on data wrangling, SQL, and stakeholder reporting for a business unit that sits outside India. In **AI-first startups**, "Data Scientist" and "ML Engineer" job titles blur completely; expect to own the whole pipeline from data pull to deployment. In **top service firms**, DS work is project-based and client-driven — breadth over depth, frequent context switching between domains, and less ownership of production systems.

By year 5, you are expected to scope your own problems (not just execute a spec), defend a metric choice in front of a sceptical stakeholder, and know when a model is not the answer — sometimes a rule, a dashboard, or a well-designed experiment does the job cheaper. The single biggest differentiator from a 2-year DS is **statistical rigour under ambiguity**: correct handling of experiment design, confounders, and multiple comparisons rather than just calling `.fit()`.

## Typical interview loop
A common structure across product companies and GCCs is 4–6 rounds:
1. **Recruiter / HR screen** — background, comp expectations, notice period.
2. **SQL + coding screen** — window functions, joins, sometimes a pandas or Python DS-adjacent problem. Lighter on DSA than an SWE loop, but expect at least one array/hashmap-style question.
3. **Statistics and ML fundamentals** — hypothesis testing, A/B testing pitfalls, bias-variance, when to use logistic regression vs. trees, metric selection under class imbalance.
4. **Case study / business problem round** — an open-ended prompt ("how would you measure the impact of feature X", "design an experiment for Y", "detect fraud in Z"). Evaluated on structuring ambiguity, not on a right answer.
5. **ML system design (lighter than an MLE loop)** — mostly about data and metric pipeline, less about serving infra.
6. **Hiring manager + team fit** — past projects, STAR-style behavioural questions, why this role/company.

Service firms often compress this to 2–3 rounds with a heavier case-study weight; AI-first startups often add a take-home.

## Domain weighting
| Domain | Weight | Why |
|---|---|---|
| [[moc-stats]] | Very high | A/B testing, causal inference, and hypothesis testing are the signature DS skill — this is what separates a DS interview from an MLE interview. |
| [[moc-sql]] | High | Most DS work in India is still SQL-first; a broken join or window function query is an instant filter. |
| [[moc-classical-ml]] | High | Trees/boosting and classic supervised learning cover the large majority of production DS models. |
| [[moc-system-design]] | Medium | Expected to reason about a metric pipeline and an experiment platform, less about serving latency. |
| [[moc-behavioral]] | Medium | Business framing and stakeholder narrative are explicitly interviewed, not incidental. |
| [[moc-programming]] | Medium | Pandas/Python fluency and light coding, not deep CS fundamentals. |
| [[moc-data-engineering]] | Low-medium | Enough to be dangerous with pipelines feeding your models, not expected to own Spark infra. |
| [[moc-deep-learning]] | Low | Occasional exposure (embeddings, ranking), rarely core. |
| [[moc-nlp-llm]] / [[moc-rag]] / [[moc-agents]] | Low | Increasingly asked about at a conceptual level as companies bolt on LLM features, but not core DS turf yet. |
| [[moc-mlops]] | Low | Aware of the lifecycle, not usually the owner of it. |

## JD vocabulary
- **"Experimentation / A/B testing platform"** → they want someone who can design tests correctly (power, randomization unit, novelty effects), not just read a `p < 0.05` off a dashboard.
- **"Causal inference"** → increasingly a real ask, not decorative; expect at minimum a definition of confounding and one method (diff-in-diff, propensity matching, or instrumental variables) at a conceptual level.
- **"Full-stack data scientist"** → expect to write your own SQL/ETL, build the model, and ship a dashboard or a simple API — no dedicated data engineer to hand off to.
- **"Business acumen" / "stakeholder management"** → you will be asked to defend a modelling choice to a non-technical VP, not just to another DS.
- **"0 to 1 problem"** at a startup → no existing pipeline, no labelled data; expect questions on how you'd bootstrap a metric or a dataset from scratch.
- **"GenAI exposure" / "LLM use cases"** → increasingly appears even in classic DS JDs; usually means using an LLM API for a feature (e.g. summarization, tagging), not training one.

## Readiness checklist
- [ ] Can design an A/B test end-to-end: hypothesis, metric (with guardrails), sample size/power calculation, randomization unit, and a stopping rule.
- [ ] Can name and explain at least 3 common pitfalls in A/B testing (novelty effect, peeking, SUTVA violation, sample ratio mismatch).
- [ ] Can explain when and why to use a non-parametric test instead of a t-test.
- [ ] Can explain multiple testing correction (Bonferroni vs. FDR) and when each applies.
- [ ] Can write a window-function SQL query (running total, rank within group, period-over-period) without hesitation.
- [ ] Can explain the bias-variance tradeoff with a concrete example from a project you shipped.
- [ ] Can justify a metric choice for an imbalanced classification problem (why not accuracy) and defend it against a stakeholder pushback.
- [ ] Can explain one causal inference method (diff-in-diff or propensity score matching) well enough to sketch it on a whiteboard.
- [ ] Can walk through one real project end-to-end in STAR format: problem framing, data, model choice, metric, business impact, what you'd do differently.
- [ ] Can reason through an open-ended case ("how would you detect X" / "design a metric for Y") out loud, structuring before jumping to a solution.
- [ ] Can explain XGBoost/LightGBM well enough to answer "why trees over logistic regression here" and vice versa.
- [ ] Can explain what data leakage looks like in a time-series or user-level dataset and how you'd catch it.
- [ ] Comfortable explaining a model's prediction to a non-technical stakeholder using SHAP or feature importance, without jargon.

## The night-before-list
- [[moc-stats]]
- [[ab-testing-design]]
- [[ab-testing-pitfalls]]
- [[hypothesis-testing]]
- [[p-values-and-significance]]
- [[type-i-and-type-ii-errors]]
- [[statistical-power-and-sample-size]]
- [[multiple-testing-correction]]
- [[causal-inference-basics]]
- [[common-statistical-tests]]
- [[window-functions]]
- [[joins-deep-dive]]
- [[bias-variance-tradeoff]]
- [[classification-metrics]]
- [[imbalanced-classification]]
- [[xgboost-deep-dive]]
- [[data-leakage]]
- [[star-method]]
- [[project-narrative-construction]]
- [[questions-to-ask-interviewers]]

## Related
- [[moc-stats]]
- [[moc-sql]]
- [[moc-classical-ml]]
- [[moc-behavioral]]
- [[plan-12-week]]
- [[plan-7-day-sprint]]
