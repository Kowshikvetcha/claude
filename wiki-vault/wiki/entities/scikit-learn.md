---
title: scikit-learn
type: entity
domain: classical-ml
roles: [data-scientist, ml-engineer]
difficulty: core
frequency: high
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# scikit-learn

## What it is
The default Python library for classical ML — linear models, trees, SVMs, clustering, preprocessing, model selection, all behind one consistent API. It is not built for deep learning or GPU compute; it is built for the 80% of tabular-data work that never needs either. Almost every classical-ML interview assumes you know its API cold.

## Core concepts
- **Estimator API**: everything is an object with `.fit(X, y)`, `.predict(X)` (or `.transform(X)`), and `.score(X, y)`. Transformers and models share this shape, which is why they compose.
- **Pipeline**: chains transformers and a final estimator into one object so preprocessing is fit only on training folds — the standard defence against leakage.
- **ColumnTransformer**: applies different preprocessing to different column subsets (e.g. scale numeric, one-hot categorical) and outputs one combined matrix.
- **Cross-validation utilities**: `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV` wrap a pipeline and a CV splitter (`KFold`, `StratifiedKFold`, `TimeSeriesSplit`).
- **Stateless vs stateful**: `fit` learns parameters (means, coefficients); `transform`/`predict` applies them. Calling `transform` before `fit` raises `NotFittedError` — this statefulness is what pipelines exploit to prevent leakage.
- **`fit_transform` shortcut**: equivalent to `fit(X).transform(X)`, but on a pipeline this must never be called on the full dataset before splitting.
- **Sparse and dense interop**: many transformers (OneHotEncoder, TF-IDF) return sparse matrices; downstream estimators must support them or you densify explicitly.

## Code
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

num_cols, cat_cols = ["age", "income"], ["city"]

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                       ("scale", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pipe = Pipeline([("prep", preprocess),
                  ("clf", RandomForestClassifier(random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

grid = GridSearchCV(pipe, {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 10]},
                     cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.score(X_test, y_test))
```

## When to use it vs alternatives
- **vs XGBoost/LightGBM**: scikit-learn's own tree ensembles (RandomForest, GradientBoosting) are fine baselines but slower and weaker on large tabular data than XGBoost/LightGBM — use scikit-learn for the pipeline scaffolding and swap in a boosted-tree library (both expose an sklearn-compatible API) for the actual model.
- **vs PyTorch/TensorFlow**: scikit-learn has no autograd, no GPU tensors — wrong tool for deep learning or anything needing custom loss/gradient work.
- **vs Spark MLlib**: scikit-learn assumes data fits in one machine's memory; MLlib exists for the distributed case. On a single-node subsample of Spark data, scikit-learn is usually simpler and faster to iterate with.

## Interview angle
**Q. Why must preprocessing go inside a Pipeline instead of being applied to the whole dataset before splitting?**
Because fitting a scaler, imputer, or encoder on the full dataset lets statistics from the test fold (mean, category cardinality) leak into training, inflating validation scores. A Pipeline re-fits preprocessing on each training fold inside cross-validation, so the held-out fold never influences it.

**Q. How does `GridSearchCV` pick the final model?**
It refits the estimator with the best-scoring hyperparameter combination on the *entire* training set (`refit=True` by default) after selecting via CV; the returned `best_estimator_` is that refit, not one of the CV folds.

**Q. When would you not use `n_jobs=-1`?**
On a shared cluster node or inside another parallel context (e.g. already inside joblib or Spark), oversubscribing CPU cores this way causes contention and can be slower than sequential.

## Traps
- Calling `fit_transform` on the full dataset, then splitting — classic leakage that inflates offline metrics silently.
- Using `LabelEncoder` on input features (ordinal features that aren't ordinal) instead of `OneHotEncoder` — it invents a false ordering.
- Forgetting `handle_unknown="ignore"` on `OneHotEncoder`, which then crashes in production the first time an unseen category appears.
- Treating `.score()` as always accuracy — it's whatever the estimator's default scorer is, which for regressors is R², not RMSE.

## Related
[[cross-validation]], [[feature-engineering]], [[data-leakage]], [[xgboost]], [[hyperparameter-tuning]]
