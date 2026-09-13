---
title: 12-Week Study Plan
type: drill
domain: system-design
roles: [data-scientist, ml-engineer, ai-engineer, mlops-engineer, agentic-engineer, fde]
difficulty: core
frequency: high
status: drafted
tags: [drill, study-plan, 12-week, cross-domain]
updated: 2026-09-13
---

# 12-Week Study Plan

Cross-domain schedule — pulls from all 13 [[moc-maths|MOCs]] in the order interviewers actually
build on them: foundations before modeling, modeling before serving, serving before whiteboard
system design. Built for someone revising with 5 years of practitioner experience already in
hand — this is resequencing and depth, not first-time learning. Adjust week count per role using
each role's `role-*.md` domain-weighting table (e.g. an MLOps Engineer can compress weeks 3-8 and
add a week to MLOps/system-design; an AI Engineer can compress weeks 3-5).

A qbank or drill noted as **(pending)** is on the curriculum but has no page in the vault yet —
check `wiki/questions/` and `wiki/drills/` again before that week; do not treat its absence as a
reason to skip the underlying concept pages.

## Week 1 — Maths + core stats

| Read | Why |
|---|---|
| [[linear-algebra-essentials]], [[vector-norms-and-distances]] | vocabulary every model is written in |
| [[eigen-decomposition-and-svd]], [[matrix-calculus-and-gradients]] | basis for PCA and backprop derivations |
| [[convexity-and-optimization-basics]], [[gradient-descent-variants]], [[lagrange-multipliers-and-constraints]] | why some losses have one minimum, SVM duals |
| [[probability-fundamentals]], [[common-probability-distributions]], [[expectation-variance-covariance]] | base layer for bias-variance and stats work |
| [[bayes-theorem-and-conditional-probability]], [[information-theory-entropy-kl]] | Naive Bayes, cross-entropy, calibration |
| [[descriptive-statistics]], [[sampling-and-sampling-distributions]], [[central-limit-theorem]], [[confidence-intervals]], [[maximum-likelihood-estimation]] | inference vocabulary the rest of stats builds on |

Drill: [[qbank-maths]].
**Checkpoint:** derive the bias-variance decomposition and the normal equation from scratch, no notes. Explain why KL divergence isn't symmetric.

## Week 2 — Stats depth, programming, SQL

| Read | Why |
|---|---|
| [[hypothesis-testing]], [[p-values-and-significance]], [[type-i-and-type-ii-errors]], [[statistical-power-and-sample-size]] | the mechanical backbone of "is this real" questions |
| [[common-statistical-tests]], [[multiple-testing-correction]], [[resampling-bootstrap-and-permutation]] | picking the right test, guardrail-metric traps |
| [[ab-testing-design]], [[ab-testing-pitfalls]], [[causal-inference-basics]], [[bayesian-inference-basics]] | the single most-asked DS stats topic, senior-differentiator traps |
| [[python-data-model-and-idioms]], [[numpy-essentials]], [[pandas-essentials]] | daily-driver fluency |
| [[big-o-and-complexity]], [[hashing-and-dictionaries]], [[arrays-and-strings-patterns]], [[coding-interview-strategy]] | coding-round bar |
| [[sql-fundamentals]], [[joins-deep-dive]], [[aggregations-and-grouping]], [[subqueries-and-ctes]], [[window-functions]], [[sql-analytics-patterns]], [[sql-query-optimization]] | universal filter across every loop in this vault |

Drill: [[qbank-stats]], [[qbank-programming]], [[qbank-sql]], [[drill-python-coding]], [[drill-sql-problems]].
**Checkpoint:** design an A/B test end to end (hypothesis, power, randomization unit, stopping rule). Clear all 15 problems in [[drill-sql-problems]] without hints.

## Week 3 — Classical ML I: framing and baseline models

| Read | Why |
|---|---|
| [[ml-problem-framing]], [[supervised-vs-unsupervised]] | first filter on any case study |
| [[train-test-validation-split]], [[cross-validation]] | gets the "did you leak data" question out of the way |
| [[bias-variance-tradeoff]], [[overfitting-and-underfitting]], [[regularization-l1-l2]] | the most-derived concept set in ML interviews |
| [[linear-regression]], [[logistic-regression]], [[generalized-linear-models]] | baseline models, interview staples |
| [[naive-bayes]], [[k-nearest-neighbours]], [[support-vector-machines]] | cheap baselines, curse-of-dimensionality and margin-theory anchors |

**Checkpoint:** derive why L1 gives sparsity and L2 doesn't, on a whiteboard.

## Week 4 — Classical ML II: trees and ensembles

| Read | Why |
|---|---|
| [[decision-trees]], [[random-forest]], [[bagging-vs-boosting]] | foundation for every ensemble question |
| [[gradient-boosting]], [[xgboost-deep-dive]], [[lightgbm-and-catboost]] | the default production model in Indian DS/MLE shops |
| [[ensemble-stacking-and-blending]], [[hyperparameter-tuning]] | search strategy, Kaggle-flavoured extras |

Drill: [[drill-ml-from-scratch]].
**Checkpoint:** derive gradient boosting as gradient descent in function space; explain XGBoost's second-order approximation and pruning.

## Week 5 — Classical ML III: features, metrics, unsupervised

| Read | Why |
|---|---|
| [[feature-engineering]], [[feature-selection]], [[categorical-encoding]], [[feature-scaling-and-transforms]], [[missing-data-handling]], [[outlier-detection]] | where real-world model quality is actually won |
| [[imbalanced-classification]], [[classification-metrics]], [[regression-metrics]], [[roc-auc-and-pr-curves]], [[probability-calibration]], [[threshold-selection]] | metric tradeoffs asked in nearly every round |
| [[clustering-kmeans]], [[hierarchical-and-density-clustering]], [[dimensionality-reduction-pca]], [[tsne-and-umap]], [[anomaly-detection]] | the unsupervised half of the domain |
| [[recommender-systems-basics]], [[time-series-forecasting]], [[time-series-features-and-validation]] | common case-study domains |
| [[model-interpretability-shap-lime]], [[data-leakage]], [[curse-of-dimensionality]], [[learning-curves-and-diagnostics]] | the senior-candidate red flags to avoid |

Drill: [[qbank-classical-ml]]. For applied review, also read [[case-fraud-detection]], [[case-churn-prediction]], and [[case-recommendation-system]] end to end.
**Checkpoint:** debug "train AUC 0.95, test AUC 0.60" across at least 3 plausible causes out loud.

## Week 6 — Deep learning core

| Read | Why |
|---|---|
| [[neural-network-fundamentals]], [[activation-functions]], [[backpropagation]], [[loss-functions]] | the single most-derived algorithm in DL interviews |
| [[weight-initialization]], [[vanishing-and-exploding-gradients]] | why half of "my model won't train" questions exist |
| [[optimizers-sgd-adam]], [[learning-rate-schedules]], [[batch-normalization-and-layernorm]], [[dropout-and-regularization-dl]] | practical training knobs |
| [[training-tricks-and-debugging]] | the "my loss is NaN" toolkit |

**Checkpoint:** derive backprop for a 2-layer network from scratch, no notes.

## Week 7 — Deep learning advanced + NLP/LLM foundations

| Read | Why |
|---|---|
| [[convolutional-neural-networks]], [[cnn-architectures]], [[recurrent-networks-and-lstm]], [[sequence-modeling-basics]] | sequence modeling before attention |
| [[attention-mechanism]], [[transformer-architecture]], [[embeddings]] | the mechanical core of every modern LLM question |
| [[transfer-learning-and-finetuning]], [[mixed-precision-and-memory]], [[distributed-training]] | practical training-at-scale reasoning |
| [[autoencoders-and-representation-learning]], [[generative-models-overview]] | sets up the generative framing LLMs inherit |
| [[text-preprocessing]], [[tokenization-bpe-and-sentencepiece]], [[word-embeddings-word2vec-glove]] | baseline NLP hygiene |
| [[language-modeling-objectives]], [[bert-and-encoder-models]], [[gpt-and-decoder-models]] | causal vs masked LM, encoder/decoder framing |

**Checkpoint:** derive why attention scores are scaled by $1/\sqrt{d_k}$. Read [[vs-encoder-vs-decoder-models]] and argue both sides of encoder-only vs decoder-only for a given task.

## Week 8 — LLM training, serving, evaluation, safety

| Read | Why |
|---|---|
| [[llm-pretraining]], [[llm-scaling-laws]] | data/compute/objective tradeoffs at scale |
| [[instruction-tuning-and-sft]], [[rlhf]], [[dpo-and-preference-optimization]] | the alignment pipeline, classic and modern |
| [[parameter-efficient-finetuning-lora]], [[quantization]], [[knowledge-distillation]] | the practical fine-tuning and serving toolkit at real cost budgets |
| [[decoding-strategies]], [[context-window-and-positional-encoding]], [[kv-cache-and-inference-optimization]], [[llm-serving-and-throughput]], [[flash-attention-and-efficient-attention]], [[mixture-of-experts]] | the serving-economics layer every AI Engineer round probes |
| [[prompt-engineering]], [[structured-output-and-function-calling]], [[hallucination-and-grounding]], [[llm-evaluation]], [[llm-safety-and-guardrails]], [[multimodal-models]], [[small-language-models-and-cost]] | production risk and cost-conscious model choice |

Drill: [[qbank-deep-learning]], [[qbank-nlp-llm]].
**Checkpoint:** argue RAG vs. fine-tuning for a stated business problem — see [[vs-rag-vs-finetuning]]. Compute the rough fp16 memory footprint of a 7B model from first principles.

## Week 9 — RAG + data engineering

| Read | Why |
|---|---|
| [[rag-overview]], [[document-ingestion-and-parsing]], [[chunking-strategies]] | where most real RAG systems actually fail first |
| [[embedding-models]], [[vector-databases]], [[ann-algorithms-hnsw-ivf]], [[hybrid-search-bm25-vector]] | retrieval representation and index tradeoffs |
| [[query-rewriting-and-expansion]], [[reranking]], [[context-assembly-and-compression]] | the precision fixes every design round expects |
| [[rag-evaluation]], [[advanced-rag-patterns]], [[graph-rag]], [[rag-failure-modes]] | proving quality, not just demoing it |
| [[data-pipeline-fundamentals]], [[batch-vs-streaming]], [[file-formats-parquet-avro]], [[warehouse-vs-lake-vs-lakehouse]] | ETL/ELT vocabulary and storage framing |
| [[spark-architecture]], [[pyspark-essentials]], [[partitioning-and-shuffling]], [[spark-performance-tuning]] | the vault owner's daily-driver stack |
| [[delta-lake]], [[medallion-architecture]], [[dlt-declarative-pipelines]], [[databricks-platform]] | the dominant Indian lakehouse pattern |
| [[data-modeling-star-schema]], [[slowly-changing-dimensions]], [[kafka-and-event-streaming]], [[data-quality-and-validation]] | warehouse modeling and streaming ingestion |

Drill: [[drill-pyspark]], [[qbank-rag]], [[qbank-data-engineering]].
**Checkpoint:** design a RAG system end to end for a real document corpus. Explain medallion bronze/silver/gold with a data-quality gate at each layer — see [[vs-spark-vs-pandas]].

## Week 10 — Agents + MLOps

| Read | Why |
|---|---|
| [[agent-fundamentals]], [[tool-calling-and-function-schemas]], [[react-and-reasoning-loops]] | what makes a system "agentic" vs a chained prompt |
| [[planning-and-task-decomposition]], [[agent-memory]], [[model-context-protocol]], [[agent-frameworks-landscape]] | design gaps interviewers probe |
| [[multi-agent-systems]], [[agentic-rag]], [[computer-use-and-browser-agents]] | coordination patterns and the newest frontier capability |
| [[human-in-the-loop-patterns]], [[agent-guardrails-and-safety]], [[agent-cost-and-latency-optimization]], [[agent-evaluation]] | the production safety and cost layer |
| [[ml-lifecycle]], [[experiment-tracking-mlflow]], [[model-registry-and-versioning]], [[data-versioning]], [[feature-stores]] | the map every other MLOps topic slots into |
| [[training-pipelines]], [[ci-cd-for-ml]], [[model-packaging-and-containers]], [[model-serving-patterns]], [[batch-vs-realtime-inference]] | training-to-serving handoff |
| [[model-monitoring]], [[data-drift-and-concept-drift]], [[model-retraining-strategies]], [[shadow-and-canary-deployment]] | the single most-asked MLOps topic in interviews |
| [[kubernetes-for-ml]], [[infrastructure-as-code]], [[orchestration-and-workflows]], [[cost-optimization-for-ml]], [[ml-testing-strategy]], [[reproducibility]] | reliability and reproducibility guardrails |
| [[unity-catalog-and-governance]], [[llmops]], [[observability-and-logging]], [[security-and-pii-in-ml]] | governance directly mapped to the Databricks stack |

Drill: [[qbank-agents]], [[qbank-mlops]].
**Checkpoint:** design a model monitoring system end to end (what's logged, what alerts, drift vs. genuine regression) and a matching agent-guardrail design.

## Week 11 — System design

| Read | Why |
|---|---|
| [[ml-system-design-framework]], [[requirements-and-metrics-definition]] | the structured approach that keeps a 45-minute round on track |
| [[distributed-systems-basics]], [[cap-theorem-and-consistency]], [[latency-and-throughput-budgets]], [[scalability-patterns]], [[caching-strategies]], [[api-design-for-ml]] | baseline vocabulary for any architecture discussion |
| [[training-serving-skew]] | the classic "why does prod differ from offline" failure |
| [[llm-system-design-framework]] | the LLM-specific variant — cost, context, latency |

Drill: [[drill-case-prompts]] (work every prompt for your target 1-2 roles), [[qbank-system-design]]. Review case studies: [[case-recommendation-system]], [[case-search-ranking]], [[case-fraud-detection]], [[case-demand-forecasting]], [[case-churn-prediction]], [[case-rag-assistant]], [[case-agentic-support-automation]], [[case-document-extraction-pipeline]], [[case-llm-cost-reduction]], [[case-ml-platform-design]], [[case-realtime-feature-pipeline]], and comparisons [[vs-rag-vs-finetuning]], [[vs-spark-vs-pandas]], [[vs-batch-vs-streaming-architecture]], [[vs-encoder-vs-decoder-models]], [[vs-xgboost-vs-neural-networks]], [[vs-vector-db-options]].
**Checkpoint:** run a full 45-minute mock design round on one prompt from [[drill-case-prompts]] with a friend or out loud against a timer.

## Week 12 — Behavioral + full mocks + review

| Read | Why |
|---|---|
| [[star-method]], [[project-narrative-construction]], [[resume-and-jd-mapping]] | the structural skeleton every behavioral answer needs |
| [[handling-failure-questions]], [[stakeholder-and-conflict-questions]] | universally asked, often botched |
| [[questions-to-ask-interviewers]], [[salary-negotiation-india]], [[interview-day-playbook]] | closing moves and negotiation norms |

Drill: [[qbank-behavioral]], and the cross-domain [[qbank-rapid-fire]] as a final warm-up. Spend the rest of the week on full mock loops (recruiter screen → technical → system design → behavioral, back to back) and re-drilling weak spots flagged across weeks 1-11: revisit [[qbank-maths]], [[qbank-stats]], [[drill-sql-problems]], [[drill-python-coding]], [[drill-pyspark]], [[drill-ml-from-scratch]], [[drill-case-prompts]].
**Checkpoint:** one complete mock loop, timed, for your actual target role — see the matching `role-*.md` night-before list one final time before the real thing.

## Related

[[plan-7-day-sprint]] · [[moc-classical-ml]] · [[moc-mlops]] · [[moc-system-design]] · [[moc-behavioral]] ·
[[role-ml-engineer]] · [[role-data-scientist]] · [[role-ai-engineer]]
