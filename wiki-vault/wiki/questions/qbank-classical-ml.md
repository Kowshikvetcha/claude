---
title: Classical ML Question Bank
type: qbank
domain: classical-ml
roles: [data-scientist, ml-engineer, mlops-engineer, fde]
difficulty: core
frequency: high
status: drafted
tags: [question-bank]
updated: 2026-09-13
---

# Classical ML Question Bank

> How to use: cover the answers, write yours first, then compare. Anything you fumble → the linked concept page's `status` should go back to `drafted`.

## Warm-up

### Q1. What does a learning curve showing high train accuracy but a large, non-closing gap to validation accuracy tell you, and what's your first fix?
**Answer.** That's the classic high-variance signature — the model has enough capacity to memorize the training set but hasn't generalized. First-order fixes: add regularization (L1/L2, tree depth/leaf constraints), get more training data (variance shrinks with data, bias doesn't), or reduce model complexity. Contrast with high-bias: both curves plateau at a mediocre score and converge to each other — there, adding data won't help; you need a more expressive model or better features.
**Follow-ups.** How would the curve look different if the problem were data leakage instead of high variance? → Leakage typically shows unrealistically high validation/test performance too (sometimes near-perfect), not a gap — because the leaked signal is available in both splits, unlike a genuine variance problem where validation performance is capped by generalization error.
**Page.** [[bias-variance-tradeoff]]

### Q2. You're building a churn model and one feature is "number of support tickets in the last 30 days," computed at prediction time. Why is this dangerous, and how do you catch it before it ships?
**Answer.** If "last 30 days" is computed relative to the labeling date rather than a fixed feature-cutoff date that respects prediction timing, you can leak information from after the point where you'd actually have to make the prediction in production (e.g. a support ticket filed because the customer already churned). Catch it by enforcing a strict point-in-time feature cutoff (as-of joins), checking feature importance for a suspiciously dominant single feature, and validating that offline feature-computation logic matches the online/serving path exactly.
**Follow-ups.** What's the difference between this "temporal" leakage and target leakage from a feature that's a proxy for the label itself? → Temporal leakage is about *when* data was available; target leakage is about a feature causally downstream of the label regardless of timing (e.g. "was refunded" predicting "returned item"). Both inflate offline metrics unrealistically, but the fixes differ — one needs an as-of cutoff, the other needs removing/re-deriving the feature.
**Page.** [[data-leakage]]

### Q3. Why do you need a three-way train/validation/test split instead of just train/test, and when would you replace the validation set with cross-validation instead?
**Answer.** If you tune hyperparameters or pick a model using test-set performance, you've implicitly fit to the test set — its reported score is no longer an unbiased estimate of generalization. The validation set absorbs that tuning; the test set is touched exactly once, at the end. Replace the fixed validation split with k-fold cross-validation when data is scarce, since holding out a full validation slice wastes precious training data and a single small split makes the tuning decision noisy.
**Follow-ups.** Why is k-fold CV itself not a safe substitute for a held-out test set? → You still search over folds/hyperparameters using CV score, so the *final* reported CV number is still optimistic for the exact configuration you selected — you still need one untouched test set for the final unbiased estimate.
**Page.** [[train-test-validation-split]]

### Q4. Your fraud classifier has 99.5% accuracy on a dataset that's 99% legitimate transactions. Why is that number meaningless, and what would you report instead?
**Answer.** A model that predicts "legitimate" for everything already scores 99% accuracy without learning anything — accuracy is dominated by the majority class under imbalance. Report precision and recall for the fraud class specifically (or F-beta weighting recall higher if missed fraud is costlier), and look at the full confusion matrix; for ranking quality independent of a threshold, use PR-AUC rather than ROC-AUC, since ROC-AUC can look deceptively good under heavy imbalance.
**Follow-ups.** Your manager wants "one number" for a dashboard — what do you pick and why? → F-beta with beta tuned to the business cost ratio of false negatives to false positives, since it's a single number but still encodes the asymmetric cost rather than being blind to it like accuracy.
**Page.** [[classification-metrics]]

### Q5. Naive Bayes assumes features are conditionally independent given the class — an assumption that's almost always false. Why does it still perform well as a baseline, and where does the independence assumption actually break it?
**Answer.** NB only needs to get the *argmax* class right, not calibrated probabilities — even with correlated features, the errors from violating independence often push the mis-estimated posterior in the same direction for all classes, so the ranking survives even though the magnitudes are wrong. It breaks down when correlated features are asymmetrically informative per class — e.g. adding the same feature twice literally squares its log-odds contribution, effectively double-counting a signal.
**Follow-ups.** Why does Naive Bayes remain a strong baseline for text classification specifically? → High-dimensional, sparse, roughly-independent word-presence features (after basic preprocessing) approximately fit the assumption, and it needs very little data per parameter compared to fitting a full joint distribution.
**Page.** [[naive-bayes]]

## Core

### Q6. Derive why gradient boosting is "gradient descent in function space," and explain what a base learner is actually fitting at each stage.
**Answer.** At stage $m$, given current ensemble $F_{m-1}$, we want the function update $h_m$ that most decreases loss $L(y, F_{m-1}(x)+h_m(x))$. Treating $F(x_i)$ as free parameters, the steepest-descent direction is $-\partial L/\partial F(x_i)$ evaluated at $F_{m-1}$ — for squared error this is exactly the residual $y_i - F_{m-1}(x_i)$, and for general losses it's the "pseudo-residual." Each base learner (a shallow tree) is fit by regression to these pseudo-residuals, then added with a shrinkage-scaled step: $F_m = F_{m-1} + \nu \cdot h_m$.
**Follow-ups.** Why does shrinkage ($\nu < 1$) usually improve test performance despite needing more trees? → It's a regularization mechanism trading a slower, smoother approach to the minimum for lower variance — analogous to a smaller learning rate giving a less noisy optimization path at the cost of more iterations.
**Page.** [[gradient-boosting]]

### Q7. What does XGBoost add on top of vanilla gradient boosting that makes it both faster and less prone to overfitting?
**Answer.** XGBoost uses a second-order (Newton) approximation of the loss — expanding to include both gradient $g_i$ and Hessian $h_i$ per sample — so each tree's optimal leaf weight has a closed form $w_j^* = -\frac{\sum_{i\in j} g_i}{\sum_{i \in j} h_i + \lambda}$, giving more accurate steps than gradient-only boosting. It also folds explicit regularization ($\gamma$ per leaf for tree complexity, $\lambda$ for L2 on leaf weights) directly into the split-gain formula, penalizing tree growth itself rather than relying only on post-hoc shrinkage/early stopping. Column/row subsampling plus histogram-based split-finding give the speed.
**Follow-ups.** Why does the closed-form leaf weight break down when the sum of Hessians in a leaf is near zero? → Near-zero Hessian means the loss is nearly flat there, so the Newton step becomes numerically unstable — this is why `min_child_weight` (a threshold on the Hessian sum) exists to prevent tiny, unstable leaves.
**Page.** [[xgboost-deep-dive]]

### Q8. Contrast bagging and boosting in terms of what each does to bias and variance, and explain why more trees in a random forest doesn't cause overfitting the way more boosting rounds can.
**Answer.** Bagging trains independent high-variance, low-bias base learners (deep trees) on bootstrap resamples and averages them — averaging i.i.d.-ish estimators reduces variance ($\text{Var}(\bar X) = \sigma^2/n$ under independence) while bias stays roughly that of a single deep tree. Boosting trains learners sequentially, each correcting the previous ensemble's errors, primarily reducing bias while variance can creep up with more rounds since later trees increasingly fit noise. Because bagging's trees are trained independently and combined by averaging, adding more of them can only reduce variance further (diminishing returns) — it structurally cannot start overfitting the way boosting can.
**Follow-ups.** So why does a random forest still overfit at all? → Individual trees can overfit locally, and averaging doesn't fully remove correlated errors across trees that share strong dominant features — controlled via `max_depth`, `min_samples_leaf`, and `max_features` (decorrelating trees by limiting the feature subset per split).
**Page.** [[bagging-vs-boosting]]

### Q9. You're building a feature for a Databricks medallion pipeline: "average order value in the trailing 90 days," computed in a silver-to-gold PySpark job. What are the main feature-engineering failure modes here?
**Answer.** Window boundary leakage: if the join isn't strictly `< as_of_date`, same-day orders (possibly including the order you're predicting) leak in. Definition drift: if silver-table records get corrected/backfilled later, recomputing the feature retroactively can produce a different value than what was logged at training time (training-serving skew). Aggregation choice matters too — mean is sensitive to large orders; consider median or a log-transform for linear models (trees don't need it). The standard fix is a feature store with point-in-time correctness and identical feature-computation logic for training and serving.
**Follow-ups.** Why is "just recompute the feature the same way for both training and serving" not sufficient on its own? → The two paths often see different data freshness (offline batch data lagging real-time serving state), so identical code can still yield different values unless the *data snapshot semantics* (as-of timestamps) are also aligned.
**Page.** [[feature-engineering]]

### Q10. You have a categorical feature "merchant_id" with 200,000 distinct values. Why is one-hot encoding a bad idea here, and what would you use instead?
**Answer.** One-hot blows up dimensionality to 200K sparse columns — expensive for trees (inefficient splits over a huge sparse space) and creates rank-deficiency for linear models. Target encoding (replacing each category with a smoothed target statistic, shrunk toward the global mean for low-count categories) compresses this to one dense column, but must be fit only on training folds via out-of-fold encoding or it silently leaks the label. Alternatives: frequency encoding, the hashing trick (fixed-dimension buckets, accepting collisions), or letting LightGBM/CatBoost bin categories internally with leakage-safe logic.
**Follow-ups.** Why does target encoding fit on the full training set (then applied to that same set) cause overfitting specifically? → Each row's own label directly informs its own encoded feature value, so the model effectively sees a smuggled copy of the label during training — out-of-fold encoding breaks this row-to-its-own-label link.
**Page.** [[categorical-encoding]]

### Q11. Name three distinct strategies for handling severe class imbalance and explain what each actually changes about the model or objective.
**Answer.** (1) Resampling (SMOTE/oversampling minority, undersampling majority) changes the *empirical data distribution* the model sees — cheap, but SMOTE's synthetic interpolation can create unrealistic points in high-dimensional/categorical space. (2) Class weighting (`scale_pos_weight`, `class_weight`) changes the *loss function itself* by reweighting the minority class's contribution, without touching the data. (3) Threshold moving leaves model and training untouched and instead recalibrates the *decision boundary* post-hoc to hit a target precision/recall tradeoff — cheapest and most directly business-aligned, but only works if the model's probability ranking is already reasonable.
**Follow-ups.** Which would you reach for first in production and why? → Threshold moving — no retraining, fastest to iterate, directly encodes business cost tradeoff; resampling/weighting are reserved for when the underlying ranking itself is poor.
**Page.** [[imbalanced-classification]]

### Q12. When would you prefer a PR curve over an ROC curve, and why can two models look identical on ROC-AUC but very different on PR-AUC?
**Answer.** ROC-AUC uses false positive rate ($FP/(FP+TN)$), normalized by the huge negative-class count under imbalance — a large absolute number of false positives barely moves FPR, making ROC-AUC look deceptively stable. Precision ($TP/(TP+FP)$) isn't normalized this way, so it's directly sensitive to the absolute count of false positives relative to true positives, exposing degradation ROC-AUC hides. Prefer PR curves whenever positives are rare and precision is the operational concern — fraud, rare-disease screening, etc.
**Follow-ups.** Can you have a high ROC-AUC and a near-useless PR-AUC on the same model? → Yes — with 0.1% positive prevalence, even 1% FPR translates into ~10x more false positives than true positives at reasonable recall, tanking precision, while ROC-AUC (dominated by the huge true-negative count) still looks close to 1.
**Page.** [[roc-auc-and-pr-curves]]

### Q13. Why does a well-tuned XGBoost model often need probability calibration even though its AUC is excellent, and how do Platt scaling and isotonic regression differ?
**Answer.** Tree ensembles optimize ranking-friendly losses and average many trees' leaf outputs, pushing predicted probabilities toward the extremes (overconfident near 0/1) even though the *ordering* of predictions (what AUC measures) is fine. Platt scaling fits a logistic regression on the raw scores ($\sigma(aS+b)$) — a simple, low-variance, monotonic correction that works with limited calibration data but assumes sigmoid-shaped miscalibration. Isotonic regression fits an arbitrary monotonic step function instead — more flexible when miscalibration isn't sigmoid-shaped, but needs more held-out data to avoid overfitting the calibration map itself.
**Follow-ups.** Why must calibration be fit on a held-out split, never the training data? → The model's own training predictions are systematically overconfident on that exact data, so calibrating on it would fit the recalibration map to a training artifact rather than genuine miscalibration on new data.
**Page.** [[probability-calibration]]

### Q14. Derive PCA as an eigendecomposition of the covariance matrix, and explain a case where using it before modeling actively hurts.
**Answer.** PCA seeks the direction $w$ (unit norm) maximizing variance of the projection $w^Tx$: $\max_w w^T\Sigma w$ s.t. $\|w\|=1$. Lagrangian $L = w^T\Sigma w - \lambda(w^Tw-1)$; setting $\partial L/\partial w = 0$ gives $\Sigma w = \lambda w$ — the optimal directions are eigenvectors of $\Sigma$, with variance captured along each equal to its eigenvalue. It hurts when the *class-discriminative* direction isn't the direction of maximum unsupervised variance — two classes separated along a low-variance axis get compressed away by top-$k$ PCA components chosen purely by variance.
**Follow-ups.** Why do tree-based models rarely benefit from PCA preprocessing the way linear/distance-based models do? → Trees split on individual feature thresholds and are invariant to monotonic transforms and correlated/high-dimensional inputs the way distance metrics or matrix-inversion-based linear methods aren't — PCA mainly trades away interpretability without fixing a problem trees actually have.
**Page.** [[dimensionality-reduction-pca]]

## Hard

### Q15. K-means is sensitive to initialization and assumes roughly spherical, similarly-sized clusters. Explain both failure modes and how k-means++ addresses one of them.
**Answer.** Random initialization can place two initial centroids in the same true cluster and none in another, converging to a locally-optimal but globally bad partition (k-means only guarantees convergence to *a* local minimum since the objective is non-convex in the assignment+centroid coupling). Separately, because k-means minimizes Euclidean distance to a centroid, it implicitly assumes convex, isotropic clusters — it fails on elongated, non-convex, or very differently-sized/density clusters. K-means++ addresses only the initialization problem: it seeds centroids sequentially with probability proportional to squared distance from the nearest already-chosen centroid, giving an $O(\log k)$ approximation guarantee — it does nothing for the shape assumption.
**Follow-ups.** What would you reach for when clusters are non-convex or density-varying? → DBSCAN/HDBSCAN (density-based, no shape assumption, handles noise as non-members) or spectral clustering (works in a similarity-graph embedding where non-convex clusters become linearly separable).
**Page.** [[clustering-kmeans]]

### Q16. What makes cross-validation for time-series forecasting fundamentally different from k-fold CV on i.i.d. tabular data, and what leakage can a naive random k-fold split cause?
**Answer.** Standard k-fold CV assumes exchangeability — any row could be in any fold — which breaks for autocorrelated temporal data where the goal is generalizing *forward in time*. A random k-fold split lets future data end up in the training fold for a validation point drawn from the past, letting the model implicitly learn from information that wouldn't exist yet in production (e.g. lag features computed with future values, or global normalization stats computed over the whole series). The correct scheme is walk-forward/expanding-window validation: train on $[0,t]$, validate on $(t,t+h]$, then roll $t$ forward.
**Follow-ups.** Why can even walk-forward validation still leak if you're not careful with feature engineering? → If rolling means, target encodings, or scaler statistics are computed once over the *entire* dataset before splitting, each fold's "past" secretly includes information from its own future — every stateful transform must be refit only on data up to the current training cutoff.
**Page.** [[time-series-features-and-validation]]

### Q17. LightGBM grows trees leaf-wise instead of level-wise like XGBoost's default, and both offer native categorical handling different from one-hot. Explain the mechanics and tradeoffs.
**Answer.** Leaf-wise growth picks the single leaf with the highest split gain across the *entire* current tree and splits only that one, versus level-wise which splits every leaf at the current depth first — leaf-wise reaches lower loss for a given leaf budget (fewer trees needed) but can grow deeper, unbalanced trees that overfit on smaller datasets, requiring tighter `max_depth`/`num_leaves` control. For categoricals, both frameworks can find an optimal split by sorting categories by a per-class statistic and evaluating gain along that order — CatBoost goes further with ordered target statistics (computed only from "prior" rows in a random permutation) specifically to prevent a category's own label from leaking into its encoding.
**Follow-ups.** Why does CatBoost's "ordered boosting" address a subtler leakage problem than just its categorical encoding? → Even the residuals/gradients used to fit each tree can leak if a row's own outcome influenced the model state used to compute its own gradient; ordered boosting computes each row's gradient using a model trained only on rows preceding it in a random permutation, avoiding target leakage in the boosting procedure itself.
**Page.** [[lightgbm-and-catboost]]

### Q18. Explain how SHAP values assign credit to each feature for a single prediction, and why they're more theoretically grounded than a generic feature-importance score.
**Answer.** SHAP is built on Shapley values from cooperative game theory: treat each feature as a "player" and the model's output as the payoff; a feature's SHAP value is its average marginal contribution across all possible orderings/subsets of features being "added in" — $\phi_i = \sum_{S \subseteq F\setminus\{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S\cup\{i\}) - f(S)]$. This is the *unique* attribution satisfying efficiency (contributions sum to prediction minus baseline), symmetry, and additivity — properties a generic split-count or gain-based importance doesn't guarantee, and why SHAP explains per-*prediction*, not just per-*model*.
**Follow-ups.** Why is exact SHAP computation intractable in general, and how does TreeSHAP get around it? → Exact Shapley values require summing over exponentially many subsets; TreeSHAP exploits tree structure to compute exact SHAP values in polynomial time by tracking which feature subsets are consistent with reaching a given leaf, avoiding brute-force enumeration.
**Page.** [[model-interpretability-shap-lime]]

## Related
See [[moc-classical-ml]].
