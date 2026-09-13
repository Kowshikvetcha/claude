---
title: MLOps — Map of Content
type: map
domain: mlops
roles: [mlops-engineer, ml-engineer, data-scientist, agentic-engineer]
updated: 2026-09-13
---

# MLOps — Map of Content

## Why this domain is asked
MLOps is the domain that decides whether an MLOps Engineer offer happens at all — it's tested almost exclusively here, as a series of "how would you operate this in production" scenarios rather than modeling questions. For ML Engineers it's the second-heaviest domain after classical ML/DL, since shipping a model that degrades silently is the most common real-world failure interviewers probe for.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[ml-lifecycle]] | The map every other MLOps topic slots into | core |
| 2 | [[experiment-tracking-mlflow]] | Table-stakes tool fluency, especially given the Databricks stack | core |
| 3 | [[model-registry-and-versioning]] | Governs safe promotion from experiment to production | core |
| 4 | [[data-versioning]] | Reproducibility question interviewers love to probe | core |
| 5 | [[feature-stores]] | Solves training/serving skew at the data layer | intermediate |
| 6 | [[training-pipelines]] | Orchestrating repeatable, auditable model training | core |
| 7 | [[ci-cd-for-ml]] | Extends software CI/CD to data+model artifacts | intermediate |
| 8 | [[model-packaging-and-containers]] | Practical packaging question, ties to Docker/Kubernetes | core |
| 9 | [[model-serving-patterns]] | Core "how do you expose this model" design question | core |
| 10 | [[batch-vs-realtime-inference]] | Framing decision behind most serving-architecture answers | core |
| 11 | [[model-monitoring]] | The single most-asked MLOps topic in interviews | core |
| 12 | [[data-drift-and-concept-drift]] | Explains *why* monitoring is needed, not just how | core |
| 13 | [[model-retraining-strategies]] | Closes the loop from monitoring to action | intermediate |
| 14 | [[shadow-and-canary-deployment]] | Safe rollout patterns, common system-design follow-up | intermediate |
| 15 | [[kubernetes-for-ml]] | Standard deployment substrate at most companies | intermediate |
| 16 | [[infrastructure-as-code]] | Reproducible infra, increasingly expected baseline | intermediate |
| 17 | [[orchestration-and-workflows]] | Airflow/DLT-style pipeline scheduling and dependency management | intermediate |
| 18 | [[cost-optimization-for-ml]] | Real budget constraint every Indian team is asked about | intermediate |
| 19 | [[ml-testing-strategy]] | Data/model tests beyond ordinary software unit tests | intermediate |
| 20 | [[reproducibility]] | Seeds, environments, determinism — a recurring trap question | intermediate |
| 21 | [[unity-catalog-and-governance]] | Directly maps to the Databricks day-job stack | intermediate |
| 22 | [[llmops]] | MLOps practices adapted for prompt/LLM-based systems | advanced |
| 23 | [[observability-and-logging]] | Debugging production model behaviour after the fact | intermediate |
| 24 | [[security-and-pii-in-ml]] | Compliance-heavy question at Indian enterprise/GCC clients | intermediate |

## How it's tested per role
- **MLOps Engineer**: the entire interview is essentially this table — expect deep, scenario-based questions across monitoring, serving, CI/CD and infra.
- **ML Engineer**: tested on the modeling-serving boundary specifically — packaging, serving patterns, monitoring — less on infra-as-code depth.
- **Data Scientist**: tested lightly, mostly experiment tracking and knowing when a model needs retraining, not deployment mechanics.
- **Agentic Engineer**: increasingly tested on LLMOps-flavoured monitoring — cost, latency and behaviour drift of agent pipelines in production.

## Question bank
See [[qbank-mlops]] for the drilled question set.

## Related domains
- [[moc-classical-ml]]
- [[moc-data-engineering]]
- [[moc-system-design]]
- [[moc-nlp-llm]]
