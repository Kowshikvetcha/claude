---
title: Maths — Map of Content
type: map
domain: maths
roles: [data-scientist, ml-engineer, ai-engineer, mlops-engineer, agentic-engineer, fde]
updated: 2026-09-13
---

# Maths — Map of Content

## Why this domain is asked
Maths rarely gets its own interview round at 5 years' experience, but it is the substrate every other round is graded against — a shaky derivation of gradient descent or the normal equation, or a hand-wavy answer on why KL divergence isn't symmetric, quietly caps how "senior" you're perceived to be. Expect it folded into ML-design and coding rounds rather than asked standalone: 10-20% of a typical DS/MLE loop touches this material directly.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[linear-algebra-essentials]] | Vectors, matrices, rank — the language every model is written in | core |
| 2 | [[vector-norms-and-distances]] | Underpins regularization, kNN, embeddings similarity | core |
| 3 | [[eigen-decomposition-and-svd]] | Basis of PCA, low-rank approximation, spectral methods | intermediate |
| 4 | [[matrix-calculus-and-gradients]] | Required to actually derive backprop and the normal equation | intermediate |
| 5 | [[convexity-and-optimization-basics]] | Explains why some losses have one minimum and others don't | intermediate |
| 6 | [[gradient-descent-variants]] | The mechanics behind every optimizer question | core |
| 7 | [[lagrange-multipliers-and-constraints]] | SVM duals and constrained optimization questions | advanced |
| 8 | [[probability-fundamentals]] | Base layer for every stats and ML-inference question | core |
| 9 | [[common-probability-distributions]] | Needed to reason about noise, priors, generative models | core |
| 10 | [[expectation-variance-covariance]] | Bias-variance, portfolio-style reasoning, feature correlation | core |
| 11 | [[bayes-theorem-and-conditional-probability]] | Naive Bayes, A/B test interpretation, calibration | core |
| 12 | [[information-theory-entropy-kl]] | Cross-entropy loss, decision tree splits, RLHF/DPO objectives | advanced |

## How it's tested per role
- **Data Scientist**: gets the deepest probability/statistics-adjacent maths grilling — expect a whiteboard derivation (e.g. bias-variance decomposition, MLE for a simple distribution).
- **ML Engineer / AI Engineer**: tested lightly and applied — "why divide attention scores by $\sqrt{d_k}$", "why does L1 give sparsity" — the maths is a means to explain an engineering choice, not an end.
- **MLOps Engineer**: maths depth expected is the lowest of the group; mostly needs enough linear algebra/statistics to reason about drift metrics and resource sizing.
- **Agentic Engineer / FDE**: almost never asked directly; occasionally shows up as "explain KL divergence" when discussing preference optimization or evaluation metrics.

## Question bank
See [[qbank-maths]] for the drilled question set.

## Related domains
- [[moc-stats]]
- [[moc-classical-ml]]
- [[moc-deep-learning]]
- [[moc-programming]]
