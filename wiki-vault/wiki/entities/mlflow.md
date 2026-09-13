---
title: MLflow
type: entity
domain: mlops
roles: [ml-engineer, mlops-engineer, data-scientist]
difficulty: core
frequency: high
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# MLflow

## What it is
The most common open-source tool for experiment tracking, model packaging, and model registry in the ML lifecycle — it answers "which run produced this model, with what params and metrics, and is it approved for production?" It's a first-class citizen inside Databricks but runs standalone too.

## Core concepts
- **Tracking**: `mlflow.log_param`, `log_metric`, `log_artifact` inside a `run` record everything needed to reproduce and compare experiments; runs are grouped under `experiments`.
- **Autologging**: `mlflow.autolog()` (or framework-specific `mlflow.sklearn.autolog()`, `mlflow.xgboost.autolog()`) captures params, metrics, and the model artifact with no manual logging calls — the fast path for standard training loops.
- **Model format (`MLmodel`)**: a flavor-agnostic packaging spec — a logged model carries enough metadata (conda/pip env, input signature, flavor) that `mlflow.pyfunc.load_model()` can load and serve it without knowing it was XGBoost, sklearn, or PyTorch underneath.
- **Model Registry**: a versioned, stage-aware store (`None → Staging → Production → Archived`) on top of logged models — the mechanism for promoting a specific run's model into production deliberately, with an audit trail of who transitioned what and when.
- **Model signature & input example**: schema (column names/types) attached to a logged model so serving infrastructure can validate inputs before they hit the model — catches schema drift at the door instead of inside `predict()`.
- **Tracking server backend**: metadata goes to a backend store (SQL database) and artifacts to an artifact store (S3/ADLS/DBFS/local) — separating "what happened" from "the actual files" is what lets multiple training jobs and UI viewers share one tracking server.

## Code
```python
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

mlflow.set_experiment("/Shared/fraud-model")

with mlflow.start_run(run_name="xgb-baseline") as run:
    params = dict(n_estimators=500, max_depth=6, learning_rate=0.05)
    mlflow.log_params(params)

    model = XGBClassifier(**params).fit(X_train, y_train)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    mlflow.log_metric("val_auc", auc)

    mlflow.xgboost.log_model(
        model, artifact_path="model",
        signature=mlflow.models.infer_signature(X_train, model.predict(X_train)),
        registered_model_name="fraud-detector",
    )

# promote a version after review
client = mlflow.MlflowClient()
client.transition_model_version_stage("fraud-detector", version=3, stage="Production")
```

## When to use it vs alternatives
- **vs Weights & Biases**: W&B has richer experiment visualization/collaboration UX and is popular in research teams; MLflow's edge is the built-in model registry and open, framework-agnostic packaging format, which matters more once you're deploying, not just experimenting.
- **vs DVC**: DVC focuses on data/pipeline versioning with git-like semantics; MLflow focuses on run tracking and model lifecycle — many teams use both.
- **vs building your own metadata store**: reasonable only at a scale/customization need MLflow genuinely can't meet; otherwise it's reinventing a registry, permissions model, and UI for no benefit.

## Interview angle
**Q. What's the difference between an MLflow "run" and a registered "model version"?**
A run is one training execution's record (params, metrics, artifacts); a registered model version is a specific model artifact promoted into the registry, which points back to the run that produced it but is a separate, stage-managed lifecycle object. Multiple runs can log models; only some get registered, and only some registered versions get promoted to Production.

**Q. How do you prevent a schema-mismatched input from silently corrupting predictions in production?**
Log the model with a signature (`infer_signature`) and enable input validation at serving time — MLflow model serving rejects or warns on inputs that don't match the logged schema, catching upstream feature-pipeline drift before it reaches the model rather than after.

**Q. Your team wants to roll back a bad production model quickly. How does the registry make that fast?**
Because stage transitions are just metadata pointers to already-logged, immutable model versions, rollback is transitioning stage back to the previous version — no retraining, no artifact rebuild — which is the entire point of separating "logging" from "promotion."

## Traps
- Treating autologging as a substitute for logging business-specific metrics — it captures what the framework knows, not domain KPIs you still need to log manually.
- Registering every experimental run instead of only reviewed candidates — turns the registry into noise and defeats its purpose as an audit trail.
- Assuming the artifact store is backed up just because the tracking server is running — artifact storage (S3/DBFS) needs its own retention/backup policy.
- Loading a model via its framework-specific flavor (`mlflow.xgboost.load_model`) in serving code that should be framework-agnostic — use `mlflow.pyfunc.load_model` so the serving path doesn't care what produced the model.

## Related
[[experiment-tracking-mlflow]], [[model-registry-and-versioning]], [[reproducibility]], [[databricks]], [[ci-cd-for-ml]]
