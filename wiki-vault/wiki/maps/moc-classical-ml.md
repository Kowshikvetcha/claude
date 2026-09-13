---
title: Classical ML — Map of Content
type: map
domain: classical-ml
roles: [data-scientist, ml-engineer, mlops-engineer, fde]
updated: 2026-09-13
---

# Classical ML — Map of Content

## Why this domain is asked
This is still the largest single domain in Indian DS/MLE interviews — tabular data, gradient-boosted trees and careful evaluation remain the dominant production workload even in the LLM era. Expect it to carry the biggest single share of a DS/MLE loop (often 2-3 full rounds): modeling choice, metrics, feature engineering and diagnosing a broken pipeline.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[ml-problem-framing]] | First filter — can you turn a business ask into an ML problem | core |
| 2 | [[supervised-vs-unsupervised]] | Basic taxonomy interviewers assume you have cold | core |
| 3 | [[train-test-validation-split]] | Gets the "did you leak data" question out of the way early | core |
| 4 | [[cross-validation]] | Standard model-selection scaffolding | core |
| 5 | [[bias-variance-tradeoff]] | The single most-derived concept in ML interviews | core |
| 6 | [[overfitting-and-underfitting]] | Vocabulary for every diagnostic conversation that follows | core |
| 7 | [[regularization-l1-l2]] | Why L1 gives sparsity — a classic derivation question | core |
| 8 | [[linear-regression]] | Baseline model, normal equation, assumptions | core |
| 9 | [[logistic-regression]] | Most-used classifier in industry; interview staple | core |
| 10 | [[generalized-linear-models]] | Generalizes regression/logistic under one framework | intermediate |
| 11 | [[naive-bayes]] | Cheap baseline, tests conditional-independence reasoning | core |
| 12 | [[k-nearest-neighbours]] | Simple, but curse-of-dimensionality discussion anchor | core |
| 13 | [[support-vector-machines]] | Margin/kernel theory, less used but still asked | intermediate |
| 14 | [[decision-trees]] | Foundation for every ensemble method that follows | core |
| 15 | [[random-forest]] | Bagging in practice, feature-importance discussion | core |
| 16 | [[bagging-vs-boosting]] | Core conceptual contrast asked almost everywhere | core |
| 17 | [[gradient-boosting]] | Derivation of boosting as gradient descent in function space | intermediate |
| 18 | [[xgboost-deep-dive]] | The default production model in Indian DS/MLE shops | core |
| 19 | [[lightgbm-and-catboost]] | Practical alternatives, categorical handling tradeoffs | intermediate |
| 20 | [[ensemble-stacking-and-blending]] | Kaggle-flavoured but occasionally asked in practice | intermediate |
| 21 | [[feature-engineering]] | Where real-world model quality is actually won | core |
| 22 | [[feature-selection]] | Dimensionality and interpretability tradeoffs | intermediate |
| 23 | [[categorical-encoding]] | High-cardinality handling, target leakage traps | core |
| 24 | [[feature-scaling-and-transforms]] | Which models need it and why | core |
| 25 | [[missing-data-handling]] | Almost every take-home has dirty data | core |
| 26 | [[outlier-detection]] | Robustness discussion, downstream metric distortion | intermediate |
| 27 | [[imbalanced-classification]] | Fraud/churn-style problems — extremely common case study | core |
| 28 | [[classification-metrics]] | Precision/recall tradeoffs asked in nearly every round | core |
| 29 | [[regression-metrics]] | MAE vs RMSE vs MAPE — business framing matters | core |
| 30 | [[roc-auc-and-pr-curves]] | Metric-curve interpretation under imbalance | core |
| 31 | [[probability-calibration]] | Needed when model output feeds a downstream decision | intermediate |
| 32 | [[threshold-selection]] | Connects metrics to actual business decisions | intermediate |
| 33 | [[hyperparameter-tuning]] | Grid/random/Bayesian search tradeoffs | intermediate |
| 34 | [[clustering-kmeans]] | Most common unsupervised interview topic | core |
| 35 | [[hierarchical-and-density-clustering]] | Alternatives when k-means assumptions fail | intermediate |
| 36 | [[dimensionality-reduction-pca]] | SVD applied — recurs in interviews and real pipelines | core |
| 37 | [[tsne-and-umap]] | Visualization tooling, common follow-up to PCA | intermediate |
| 38 | [[anomaly-detection]] | Adjacent to imbalanced classification, own techniques | intermediate |
| 39 | [[recommender-systems-basics]] | Common case-study domain at product companies | intermediate |
| 40 | [[time-series-forecasting]] | Distinct evaluation and modeling assumptions | intermediate |
| 41 | [[time-series-features-and-validation]] | Leakage traps specific to temporal data | advanced |
| 42 | [[model-interpretability-shap-lime]] | Increasingly required in regulated/enterprise settings | intermediate |
| 43 | [[data-leakage]] | The most commonly cited "senior candidate" red flag to avoid | core |
| 44 | [[curse-of-dimensionality]] | Explains why several earlier techniques exist at all | intermediate |
| 45 | [[learning-curves-and-diagnostics]] | Practical debugging skill, ties the whole domain together | intermediate |

## How it's tested per role
- **Data Scientist**: broadest and deepest coverage — expect a full modeling round plus a case study requiring feature engineering, metric choice and leakage awareness.
- **ML Engineer**: modeling depth expected is similar, but questions pivot toward "how would you serve/monitor this" faster than a DS round would.
- **MLOps Engineer**: needs working knowledge (metrics, drift-relevant concepts) rather than derivation depth — the bar is "can you tell if a model degraded," not "can you derive gradient boosting."
- **FDE**: needs breadth to speak credibly to a client about model choice, but rarely implements from scratch — the bar is judgment, not derivation.

## Question bank
See [[qbank-classical-ml]] for the drilled question set.

## Related domains
- [[moc-maths]]
- [[moc-stats]]
- [[moc-deep-learning]]
- [[moc-mlops]]
- [[moc-system-design]]
