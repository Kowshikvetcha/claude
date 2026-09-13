---
title: XGBoost
type: entity
domain: classical-ml
roles: [data-scientist, ml-engineer, mlops-engineer]
difficulty: core
frequency: high
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# XGBoost

## What it is
The most widely deployed gradient-boosted tree library in industry tabular ML — fast, regularized, and battle-tested at scale. For the algorithm itself (second-order Taylor expansion, split-finding, regularization terms) see [[xgboost-deep-dive]]; this page is the "how you actually use the library" companion.

## Core concepts
- **Two APIs**: the low-level `xgb.train(params, DMatrix, ...)` and the sklearn-compatible `XGBClassifier`/`XGBRegressor`. Use the sklearn API inside pipelines and grid search; use the native API when you need custom objectives, `DMatrix`-level control, or distributed training.
- **`DMatrix`**: XGBoost's internal data structure — pre-bins features into histograms and supports missing values natively (`np.nan` is handled without imputation).
- **Early stopping**: pass an eval set and `early_stopping_rounds`; training halts when validation metric stops improving, and `best_iteration` tells you the effective tree count — critical to avoid overfitting via too many boosting rounds.
- **Key hyperparameters**: `max_depth` (tree complexity), `learning_rate`/`eta` (shrinkage), `n_estimators` (rounds), `subsample`/`colsample_bytree` (row/column sampling — the "bagging inside boosting" regularizer), `reg_alpha`/`reg_lambda` (L1/L2 on leaf weights), `min_child_weight` (minimum Hessian sum in a leaf — controls overfitting on sparse/imbalanced data).
- **`scale_pos_weight`**: the standard lever for class imbalance instead of resampling — set to `neg_count/pos_count`.
- **Feature importance flavors**: `gain` (average improvement per split — usually the right one), `weight` (split count — biased toward high-cardinality features), `cover` (samples affected). Confusing these is a common analysis mistake.
- **GPU support**: `tree_method="hist"` (CPU, default and fast) vs `device="cuda"` for GPU histogram building on large datasets.

## Code
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, min_child_weight=5,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="auc", early_stopping_rounds=50,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(model.best_iteration, roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
```

## When to use it vs alternatives
- **vs LightGBM**: LightGBM's leaf-wise growth and native categorical support are usually faster on very large or high-cardinality datasets; XGBoost's level-wise growth is a touch more conservative against overfitting on smaller data. In practice both are tried and the better validation score wins.
- **vs CatBoost**: CatBoost's ordered target encoding is the strongest option for heavy categorical features without manual encoding; XGBoost needs encoding done upstream.
- **vs neural networks on tabular data**: see [[vs-xgboost-vs-neural-networks]] — XGBoost remains the stronger default on small-to-medium structured data with heterogeneous feature types; NNs win when there's exploitable structure (text, images, very large row counts) or you need shared embedding representations across tasks.

## Interview angle
**Q. Your XGBoost model has near-perfect train AUC and mediocre validation AUC. Fix it.**
Reduce `max_depth`, increase `min_child_weight` and `reg_lambda`, lower `subsample`/`colsample_bytree`, and use early stopping on a validation set rather than a fixed large `n_estimators`. Also check for leakage before touching hyperparameters — overfitting this extreme is often a leaked feature, not model capacity.

**Q. How do you handle class imbalance in XGBoost — resampling or `scale_pos_weight`?**
Prefer `scale_pos_weight` first: it reweights the gradient/Hessian without duplicating or discarding rows, so it doesn't distort the feature distribution or blow up training time. Resampling (SMOTE etc.) is a fallback when the imbalance is extreme and reweighting alone doesn't move recall enough.

**Q. Why does XGBoost handle missing values without imputation, and is that always desirable?**
At each split, XGBoost learns a default direction for missing values as part of split-finding, effectively treating "missing" as informative. That's usually fine and saves imputation work — but it means missingness patterns silently become part of the model, which can bake in leakage if the missingness itself correlates with the label for a reason (e.g. a field only populated after the label is known).

## Traps
- Reporting `weight`-based feature importance and calling it "predictive power" — it just counts splits and favours high-cardinality features.
- Setting `n_estimators` very high with no early stopping and no validation set — trains a model no one checked for overfitting.
- Using `scale_pos_weight` for **multi-class** imbalance — it only applies to binary objectives; use `sample_weight` instead.
- Forgetting that GPU (`hist` + `cuda`) and CPU `hist` can give slightly different results due to floating-point non-determinism — don't expect bit-identical reruns across devices.

## Related
[[xgboost-deep-dive]], [[gradient-boosting]], [[bagging-vs-boosting]], [[hyperparameter-tuning]], [[imbalanced-classification]], [[lightgbm-and-catboost]]
