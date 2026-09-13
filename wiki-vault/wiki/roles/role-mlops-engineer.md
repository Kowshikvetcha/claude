---
title: MLOps Engineer
type: role
domain: meta
roles: [mlops-engineer]
updated: 2026-09-13
---

# MLOps Engineer

## What this role actually does
At ~5 years experience, an MLOps Engineer in India is primarily an infrastructure and platform engineer whose customers are data scientists and ML engineers, not end users directly. The job is building and operating the plumbing: pipelines, registries, serving infra, monitoring, and governance — so that DS/MLE teams can ship models without each reinventing deployment. This role leans more "platform/DevOps for ML" than "modelling," and is judged heavily on reliability, reproducibility, and cost efficiency rather than model accuracy.

In **product companies**, MLOps Engineers typically sit on a central ML platform team supporting many model teams; expect strong emphasis on self-service tooling, CI/CD standardization, and cost control across dozens of models. In **GCCs**, the role is often tied to a specific global platform (a bank's or retailer's internal ML platform) with heavier governance/compliance requirements — audit trails, model risk management, data residency — and less freedom to pick tools. In **AI-first startups**, MLOps is frequently one person wearing many hats (infra, DevOps, and some modelling support), with a much scrappier stack (managed services over self-hosted Kubernetes) and correspondingly less depth expected in any one area. **Service firms** staff MLOps roles for client platform builds — expect to implement a client-specified stack rather than choose your own, and heavy documentation/handoff expectations.

By year 5, you're expected to reason about *tradeoffs* in the platform itself: batch vs. real-time serving cost, when to introduce a feature store vs. when it's overkill, how much observability is enough before it becomes noise. Deep modelling theory is explicitly not the bar here — knowing how gradient boosting works matters less than knowing how to detect that a deployed model's input distribution has silently shifted.

## Typical interview loop
Typically 4–6 rounds, weighted toward systems and infra:
1. **Recruiter screen.**
2. **Coding/scripting round** — Python and often some infra-as-code or bash; less DSA-heavy than an SWE loop, more "can you write a robust pipeline script."
3. **MLOps deep dive** — experiment tracking, model registry, CI/CD for ML, containerization, orchestration (Airflow/Databricks Workflows/similar), and how these fit together.
4. **System design (platform-flavoured)** — design an ML platform component: a feature store, a monitoring system, a model-serving layer at a given scale/latency budget.
5. **Data engineering / infra round** — Spark fundamentals, data pipeline design, IaC (Terraform or similar), Kubernetes basics.
6. **Hiring manager + behavioral** — incident response stories, cross-team collaboration with DS/MLE, how you've handled a production outage.

GCCs often add a compliance/governance-focused round; startups often compress rounds 3–5 into a single long systems conversation.

## Domain weighting
| Domain | Weight | Why |
|---|---|---|
| [[moc-mlops]] | Very high | This is the entire job — registry, monitoring, retraining, CI/CD, governance are the core competency being hired for. |
| [[moc-data-engineering]] | Very high | Pipelines, Spark, orchestration, and data quality are daily tools, not adjacent skills. |
| [[moc-system-design]] | High | Platform-level design (serving, caching, scalability) is tested explicitly and in depth. |
| [[moc-programming]] | Medium-high | Strong scripting and automation ability expected; DSA bar usually lighter than a pure SWE loop. |
| [[moc-behavioral]] | Medium | Incident handling and cross-team collaboration stories are explicitly probed. |
| [[moc-classical-ml]] | Low-medium | Enough to understand what you're serving/monitoring, not expected to design models yourself. |
| [[moc-sql]] | Low-medium | Useful for data validation and pipeline debugging. |
| [[moc-deep-learning]] | Low | Only as much as needed to understand serving requirements (e.g. GPU memory for a model). |
| [[moc-nlp-llm]] | Low | Rising fast as LLMOps becomes its own sub-track, but not core for a classical MLOps JD. |
| [[moc-stats]] | Low | Rarely tested directly. |

## JD vocabulary
- **"ML platform"** → building reusable infra for multiple model teams, not a single model's pipeline; expect questions about multi-tenancy and self-service.
- **"Model lifecycle management"** → registry, versioning, promotion gates (staging → production), and rollback — expect concrete process questions, not just tool names.
- **"Observability" for ML** → beyond standard app metrics: input/output drift, prediction distribution monitoring, data quality checks on incoming features.
- **"Infrastructure as Code" (Terraform, similar)** → expect to be asked to reason about reproducible environments, not just write YAML.
- **"Governance" / "Unity Catalog" / "model risk management"** → especially strong in GCC and BFSI-adjacent JDs; expect questions on access control, lineage, and audit trails.
- **"Cost optimization"** → a very real, frequently asked concern in India-based teams — spot instances, autoscaling, right-sizing GPU/CPU for inference.
- **"LLMOps"** → increasingly appended to MLOps JDs; expect at least a conceptual question on prompt versioning or LLM-specific monitoring even in a classical MLOps interview.

## Readiness checklist
- [ ] Can design an end-to-end CI/CD pipeline for ML, including what automated checks gate a model's promotion to production.
- [ ] Can explain the difference between a model registry and a feature store, and why both exist.
- [ ] Can design a model monitoring system: what to log, what triggers an alert, how you'd detect drift vs. a genuine performance drop.
- [ ] Can explain data drift vs. concept drift with a concrete detection method for each.
- [ ] Can design a retraining strategy with clear triggers (schedule, drift threshold, performance floor) and a safe rollout plan (shadow/canary).
- [ ] Can explain containerization for ML serving (why package a model this way, what goes wrong without it).
- [ ] Can reason about batch vs. real-time inference tradeoffs for a given latency/cost budget.
- [ ] Can explain an orchestration tool's role (DAGs, retries, dependency management) using a concrete pipeline example.
- [ ] Can explain medallion architecture and why data quality gates matter at each layer.
- [ ] Can describe how you'd secure and govern access to a shared feature/model platform (roles, lineage, audit).
- [ ] Comfortable reasoning about infrastructure-as-code for reproducible ML environments.
- [ ] Can tell one real production incident story (model or pipeline failure) focused on detection, diagnosis, and prevention.
- [ ] Can explain cost optimization levers for ML infra (autoscaling, spot instances, batching, model size choice) concretely.

## The night-before-list
- [[moc-mlops]]
- [[ml-lifecycle]]
- [[experiment-tracking-mlflow]]
- [[model-registry-and-versioning]]
- [[ci-cd-for-ml]]
- [[model-serving-patterns]]
- [[batch-vs-realtime-inference]]
- [[model-monitoring]]
- [[data-drift-and-concept-drift]]
- [[model-retraining-strategies]]
- [[shadow-and-canary-deployment]]
- [[orchestration-and-workflows]]
- [[infrastructure-as-code]]
- [[kubernetes-for-ml]]
- [[cost-optimization-for-ml]]
- [[unity-catalog-and-governance]]
- [[medallion-architecture]]
- [[spark-architecture]]
- [[data-quality-and-validation]]
- [[scalability-patterns]]

## Related
- [[moc-mlops]]
- [[moc-data-engineering]]
- [[moc-system-design]]
- [[plan-12-week]]
- [[plan-7-day-sprint]]
