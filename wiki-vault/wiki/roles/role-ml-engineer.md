---
title: ML Engineer
type: role
domain: meta
roles: [ml-engineer]
updated: 2026-09-13
---

# ML Engineer

## What this role actually does
At ~5 years experience, an ML Engineer in India is expected to own a model **all the way to production** — not just get good offline metrics. In practice this means: solid feature engineering on top of whatever data platform the company runs (Spark/Databricks in larger setups, pandas/SQL in smaller ones), model training that is overwhelmingly gradient-boosted trees (XGBoost/LightGBM) with occasional deep learning for specific problems (ranking, recommendations, fraud), and real ownership of the training-to-serving handoff: versioning, monitoring, retraining triggers.

The mix of "ML" vs. "engineering" work shifts hard by company type. In **product companies**, expect roughly half your time in engineering — pipelines, CI/CD for models, serving infra, on-call for model regressions — and half in modelling. In **GCCs**, the role often has more engineering ownership per person (smaller ML teams supporting large legacy systems) but less freedom to pick architectures — you inherit constraints from a global team. In **AI-first startups**, ML Engineer often *is* the MLOps person too; you set up the registry, the pipeline, and the monitoring yourself with no dedicated platform team. In **service firms**, the role is closer to "build what the client's SOW says," with less say in modelling choices and heavier emphasis on delivery timelines and documentation.

A 5-year ML Engineer is expected to reason about the full lifecycle: why a model is retrained weekly vs. daily, what happens when a feature pipeline breaks silently, how a schema change in Unity Catalog or a similar governance layer breaks reproducibility. Interviewers increasingly probe for these system-level failure modes, not just modelling theory.

## Typical interview loop
Typically 5–6 rounds, more engineering-heavy than a DS loop:
1. **Recruiter screen.**
2. **Coding round** — DSA (arrays, hashmaps, trees/graphs, sometimes DP), often on a live-coding platform. Expect this to be a real filter, not a formality.
3. **ML fundamentals / applied ML deep dive** — heavy on trees/boosting internals, feature engineering, handling of categorical/missing data, metric selection, and often a "debug this model" style question (why is train AUC 0.95 and test AUC 0.60).
4. **ML system design** — design a recommendation system, a fraud pipeline, or a feature store; expected to reason about training-serving skew, latency budgets, and retraining cadence, not just draw boxes.
5. **MLOps / production round** — CI/CD for models, model registry, monitoring, rollback strategy; sometimes merged into the system design round at smaller companies.
6. **Hiring manager + behavioral** — ownership stories, incident handling, cross-team collaboration.

Startups often collapse rounds 3–5 into one long technical conversation plus a take-home; GCCs tend to add an extra process/compliance-oriented round.

## Domain weighting
| Domain | Weight | Why |
|---|---|---|
| [[moc-classical-ml]] | Very high | Trees/boosting depth is the single most tested area — expect deep questions on XGBoost internals, feature engineering, and metric tradeoffs. |
| [[moc-mlops]] | Very high | This is what separates an ML Engineer from a DS — registry, monitoring, retraining, CI/CD are core, not bonus. |
| [[moc-system-design]] | High | ML system design rounds are near-universal at this level. |
| [[moc-programming]] | High | Real DSA coding bar, plus strong Python/pandas fluency. |
| [[moc-data-engineering]] | High | Spark/pipeline literacy expected, especially where Databricks-style stacks are in play. |
| [[moc-deep-learning]] | Medium | Enough to reason about when DL beats trees and vice versa; rarely deep theory unless the role is DL-specific. |
| [[moc-sql]] | Medium | Needed for feature debugging and data validation, less central than for a DS. |
| [[moc-behavioral]] | Medium | Ownership and incident-response narratives are explicitly probed. |
| [[moc-stats]] | Low-medium | Enough for metric validity and basic experiment reading, not the centre of the loop. |
| [[moc-nlp-llm]] / [[moc-rag]] / [[moc-agents]] | Low | Rising in JDs but usually not core unless the team ships LLM features. |

## JD vocabulary
- **"End-to-end ownership"** → you will build, deploy, and be paged for the model, not hand it off after a notebook.
- **"Feature store"** → expect questions on point-in-time correctness and training-serving skew, not just the tool name.
- **"Model in production" / "productionize"** → they've been burned by DS teams that only ship notebooks; be ready to talk packaging, serving, and monitoring concretely.
- **"MLOps maturity"** → ask (or expect to be asked) where the company actually sits: notebook-only, some CI, or full registry+monitoring+retraining automation. Answers vary wildly across India-based teams.
- **"Scale"** — in Indian product companies this usually means throughput/cost efficiency at moderate data volumes (not FAANG-scale); in GCCs it can mean supporting a global system with strict SLAs.
- **"Databricks / Unity Catalog / Delta Lake"** → strong signal the company runs a lakehouse stack; know medallion architecture and governance concepts even if you haven't used that exact tool.

## Readiness checklist
- [ ] Can explain gradient boosting from first principles (residual fitting, learning rate, regularization) without notes.
- [ ] Can explain XGBoost-specific internals: second-order gradients, tree pruning, handling of missing values.
- [ ] Can design a feature engineering pipeline that avoids training-serving skew and explain point-in-time correctness.
- [ ] Can debug a model with high train performance and poor test performance across at least 3 plausible causes (leakage, overfitting, distribution shift).
- [ ] Can design an end-to-end ML system (e.g. recommendation, fraud detection) covering data, features, training, serving, and monitoring.
- [ ] Can explain the difference between batch and real-time inference and when each is appropriate.
- [ ] Can explain what a model registry does and why it matters for rollback and audit.
- [ ] Can explain data drift vs. concept drift and how you'd detect each in production.
- [ ] Can describe a CI/CD pipeline for ML (not just software) — what gets tested before a model ships.
- [ ] Can solve a medium-difficulty DSA problem (arrays/hashmaps/trees) cleanly under time pressure.
- [ ] Can explain categorical encoding tradeoffs (target encoding leakage risk, one-hot cardinality blow-up) with a fix for each.
- [ ] Can explain a retraining strategy: what triggers it, how you validate before promoting a new model.
- [ ] Can tell one real incident story (model broke, pipeline failed) in STAR format, focused on diagnosis and fix.

## The night-before-list
- [[moc-classical-ml]]
- [[gradient-boosting]]
- [[xgboost-deep-dive]]
- [[lightgbm-and-catboost]]
- [[bagging-vs-boosting]]
- [[feature-engineering]]
- [[categorical-encoding]]
- [[data-leakage]]
- [[hyperparameter-tuning]]
- [[classification-metrics]]
- [[ml-system-design-framework]]
- [[training-serving-skew]]
- [[model-monitoring]]
- [[data-drift-and-concept-drift]]
- [[model-registry-and-versioning]]
- [[ci-cd-for-ml]]
- [[batch-vs-realtime-inference]]
- [[feature-stores]]
- [[medallion-architecture]]
- [[star-method]]

## Related
- [[moc-classical-ml]]
- [[moc-mlops]]
- [[moc-system-design]]
- [[moc-data-engineering]]
- [[plan-12-week]]
- [[plan-7-day-sprint]]
