---
title: 7-Day Sprint Plan
type: drill
domain: system-design
roles: [data-scientist, ml-engineer, ai-engineer, mlops-engineer, agentic-engineer, fde]
difficulty: core
frequency: high
status: drafted
tags: [drill, study-plan, 7-day-sprint, cross-domain]
updated: 2026-09-13
---

# 7-Day Sprint Plan

This is triage, not curriculum. You have an interview in a week — the goal is to be dangerous on
the highest-frequency 60% of what gets asked, not thorough on all of it. If you have the luxury of
12 weeks instead, use [[plan-12-week]] and ignore this page. Every day below names what to skip —
read the skip list before you start, or you'll burn hours on low-yield depth.

Pick your target role's `role-*.md` domain-weighting table first (e.g. [[role-ml-engineer]],
[[role-data-scientist]]) and bias extra time toward whatever it marks "very high" — this plan
assumes a generalist loop and needs that one adjustment per person.

## Day 1 — Maths/stats warm-up

**Cover:** [[bias-variance-tradeoff]], [[regularization-l1-l2]] (why L1 gives sparsity — say it out
loud once), [[gradient-descent-variants]] intuition, [[bayes-theorem-and-conditional-probability]],
[[hypothesis-testing]], [[p-values-and-significance]], [[type-i-and-type-ii-errors]],
[[ab-testing-design]], [[ab-testing-pitfalls]], [[classification-metrics]].
Drill: [[qbank-maths]], [[qbank-stats]] — timed, no notes.

**Skip:** [[lagrange-multipliers-and-constraints]], full [[eigen-decomposition-and-svd]] derivations,
[[causal-inference-basics]] depth, [[bayesian-inference-basics]] depth. Know the one-sentence
definition of each and move on — these are asked directly only for a Data Scientist loop, and even
then rarely as the deciding question.

## Day 2 — Classical ML core

**Cover:** [[decision-trees]], [[bagging-vs-boosting]], [[gradient-boosting]],
[[xgboost-deep-dive]] (second-order gradients, pruning, missing-value handling — this is the single
most-asked derivation for ML Engineer/Data Scientist), [[feature-engineering]],
[[categorical-encoding]], [[data-leakage]], [[imbalanced-classification]], [[roc-auc-and-pr-curves]].
Spot-check: [[drill-ml-from-scratch]] on trees/boosting problems only, and skim [[qbank-classical-ml]]'s Core section.

**Skip:** [[ensemble-stacking-and-blending]], [[tsne-and-umap]],
[[hierarchical-and-density-clustering]], [[curse-of-dimensionality]] depth,
[[time-series-features-and-validation]] unless your target JD explicitly mentions forecasting.
These show up as follow-ups, not openers — a one-line awareness answer is enough under time
pressure.

## Day 3 — Deep learning + NLP/LLM

**Cover:** [[backpropagation]] (intuition and the chain-rule shape, not a full from-scratch
derivation unless you're an AI/ML Engineer), [[attention-mechanism]], [[transformer-architecture]],
[[embeddings]], [[parameter-efficient-finetuning-lora]], [[quantization]],
[[decoding-strategies]], [[hallucination-and-grounding]], [[llm-evaluation]].
Spot-check: [[qbank-deep-learning]] and [[qbank-nlp-llm]] Warm-up + Core sections only.

**Skip:** [[distributed-training]] internals, [[mixture-of-experts]], full
[[flash-attention-and-efficient-attention]] mechanics, the maths of [[rlhf]]/[[dpo-and-preference-optimization]].
Have one paragraph ready for each ("RLHF trains a reward model then does PPO against it; DPO skips
the reward model and optimizes preference pairs directly") and do not go deeper unless pushed —
these are asked as follow-ups almost exclusively in AI Engineer loops.

## Day 4 — RAG + agents

**Cover:** [[rag-overview]], [[chunking-strategies]], [[hybrid-search-bm25-vector]], [[reranking]],
[[rag-failure-modes]], [[agent-fundamentals]], [[tool-calling-and-function-schemas]],
[[react-and-reasoning-loops]], [[agent-guardrails-and-safety]].
Spot-check: [[qbank-agents]] and [[qbank-rag]] Warm-up + Core sections only.

**Skip:** [[graph-rag]], [[advanced-rag-patterns]], deep [[multi-agent-systems]] coordination theory,
[[agent-cost-and-latency-optimization]] arithmetic, [[computer-use-and-browser-agents]]. If asked,
name them as "the next thing I'd reach for if the simple version isn't enough" — that answer
signals judgment without needing depth you didn't have time to build this week.

## Day 5 — MLOps + data engineering + SQL

**Cover:** [[model-monitoring]], [[data-drift-and-concept-drift]], [[ci-cd-for-ml]],
[[model-registry-and-versioning]], [[batch-vs-realtime-inference]], [[medallion-architecture]],
[[spark-architecture]], [[pyspark-essentials]], [[window-functions]], [[joins-deep-dive]],
[[sql-analytics-patterns]].
Drill: [[drill-sql-problems]] timed end to end, [[drill-pyspark]] spot-check, and skim
[[qbank-mlops]] plus [[qbank-data-engineering]] Warm-up sections.

**Skip:** [[infrastructure-as-code]] tool specifics, deep [[kubernetes-for-ml]] mechanics (know the
one-sentence version: "containers give reproducible, scalable serving; autoscale on queue depth or
GPU util, not CPU"), [[dlt-declarative-pipelines]], [[kafka-and-event-streaming]] depth,
[[slowly-changing-dimensions]]. These matter far more for an MLOps Engineer loop specifically — if
that's your target role, swap this skip list for the read list instead.

## Day 6 — System design + case studies

**Cover:** [[ml-system-design-framework]] or [[llm-system-design-framework]] (pick whichever matches
your target role) as your structural checklist. Run 4-5 prompts from [[drill-case-prompts]] under
[[role-ml-engineer|your role's section]], timed at 45 minutes each, and drill [[qbank-system-design]]'s
Warm-up + Core sections. Skim — don't deep-read — the two case studies closest to your target
role's domain (check [[case-recommendation-system]], [[case-search-ranking]],
[[case-fraud-detection]], [[case-demand-forecasting]], [[case-churn-prediction]],
[[case-rag-assistant]], [[case-agentic-support-automation]], [[case-document-extraction-pipeline]],
[[case-llm-cost-reduction]], [[case-ml-platform-design]], [[case-realtime-feature-pipeline]] and
pick 2, not all of them).

**Skip:** the other two case studies entirely, and deep [[distributed-systems-basics]] /
[[cap-theorem-and-consistency]] theory beyond the one-paragraph tradeoff each concept page opens
with. A system-design round rewards structure and tradeoff-naming under time pressure far more
than distributed-systems trivia.

## Day 7 — Behavioral + rapid-fire review + rest

**Cover:** [[star-method]], [[handling-failure-questions]], [[stakeholder-and-conflict-questions]],
[[questions-to-ask-interviewers]], [[salary-negotiation-india]]. Prepare 3 STAR stories (a failure,
a conflict, an ownership win) and rehearse them out loud once, not on paper.
Drill: [[qbank-behavioral]]'s Warm-up section, out loud.
Rapid-fire review only: skim [[qbank-maths]], [[qbank-stats]], and [[qbank-rapid-fire]] across
every domain, and re-read the answers (not the questions cold) on anything you got wrong earlier
in the week.

**Skip:** everything else. Do not open a new concept page today. Stop technical prep by early
afternoon — the marginal value of one more page is lower than being rested. Read
[[interview-day-playbook]] once in the evening and go to sleep on time.

## What this plan deliberately does not cover

No maths derivations beyond bias-variance and L1 sparsity, no causal inference, no distributed
training, no multi-agent coordination theory, no Kubernetes/Terraform mechanics, no time-series
leakage traps. If any of those is explicitly named in your JD, swap one skip-list item for it —
but do not add without removing something else; a 7-day sprint that tries to cover everything
covers nothing well.

## Related

[[plan-12-week]] · [[drill-case-prompts]] · [[drill-sql-problems]] · [[drill-python-coding]] ·
[[drill-pyspark]] · [[drill-ml-from-scratch]] · [[moc-behavioral]]
