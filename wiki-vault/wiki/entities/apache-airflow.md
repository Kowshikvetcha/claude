---
title: Apache Airflow
type: entity
domain: data-engineering
roles: [ml-engineer, mlops-engineer]
difficulty: core
frequency: medium
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# Apache Airflow

## What it is
The most widely used workflow orchestrator for scheduling and monitoring pipelines expressed as directed acyclic graphs (DAGs) of tasks — ETL jobs, ML training pipelines, cross-system dependencies. It answers "run this after that, retry on failure, alert someone, and show me the history," not the data processing itself.

## Core concepts
- **DAG**: a Python file defining tasks and their dependencies (`task_a >> task_b`) — the DAG is *defined* in Python but each task typically *executes* elsewhere (a Spark job, a container, a SQL query) via an operator.
- **Operators**: the unit of work — `PythonOperator`, `BashOperator`, provider-specific operators (`DatabricksSubmitRunOperator`, `KubernetesPodOperator`, `S3ToRedshiftOperator`). Modern Airflow favors the **TaskFlow API** (`@task` decorator) for a more Pythonic DAG-authoring style over instantiating operators directly.
- **Scheduler & executor**: the scheduler parses DAGs and decides what's due to run; the executor (Celery, Kubernetes, Local) determines *where* tasks actually run — this split is why Airflow scales from a single laptop to a multi-node production cluster without changing DAG code.
- **XComs**: small cross-task data passing (task A's output referenced by task B) — deliberately size-limited; not a substitute for passing large datasets, which should go through external storage with only a reference passed via XCom.
- **Idempotency & `execution_date`/logical date**: DAG runs are keyed by a logical date, not wall-clock run time, so backfills and reruns for a given date produce (should produce) the same result — tasks must be written idempotently (safe to rerun) for retries and backfills to be meaningful.
- **Sensors**: a special operator type that waits for a condition (a file to land, a partition to appear) before downstream tasks proceed — the standard way to make a DAG event-driven within an otherwise schedule-driven system, though they can occupy worker slots if not configured with deferrable/reschedule mode.
- **SLAs, retries, alerting**: per-task `retries`, `retry_delay`, and `on_failure_callback` (Slack/PagerDuty hooks) are what make Airflow a production-grade scheduler rather than just cron.

## Code
```python
from airflow.decorators import dag, task
from datetime import datetime, timedelta

default_args = {"retries": 2, "retry_delay": timedelta(minutes=5)}

@dag(schedule="@daily", start_date=datetime(2026, 1, 1), catchup=False, default_args=default_args)
def daily_feature_pipeline():

    @task
    def extract():
        return "/path/to/raw/2026-09-13"

    @task
    def transform(raw_path: str):
        # e.g. submit a Spark/Databricks job, wait for completion
        return raw_path.replace("raw", "features")

    @task
    def load(features_path: str):
        print(f"loading {features_path} into feature store")

    load(transform(extract()))

daily_feature_pipeline()
```

## When to use it vs alternatives
- **vs Databricks Workflows**: Databricks Jobs are simpler and tightly integrated when the entire pipeline lives inside Databricks; Airflow wins once the pipeline spans multiple systems (S3, Snowflake, Databricks, an API call, a Kubernetes job) with dependency logic across all of them.
- **vs Dagster/Prefect**: Dagster's asset-centric model (tracking data assets, not just task runs) gives stronger data lineage and testing ergonomics out of the box; Prefect has a lighter-weight, more Pythonic dynamic-DAG story. Airflow's advantage is ecosystem maturity and the largest set of pre-built provider operators.
- **vs plain cron**: cron has no dependency graph, no retry/backfill semantics, no UI, and no cross-task data passing — fine for a single isolated job, wrong for anything with dependencies or failure-handling needs.

## Interview angle
**Q. What's the difference between `execution_date` (logical date) and the actual run time, and why does it matter?**
A DAG scheduled `@daily` for Jan 2 actually runs on Jan 3 (after the interval closes) but is tagged with logical date Jan 2 — the date the data *represents*, not when it ran. This lets backfills for arbitrary historical dates produce results consistent with what would have run "as of" that date, and is why tasks must pull data scoped to the logical date rather than "now."

**Q. Why can XComs be a design smell if you rely on them heavily?**
XComs are meant for small metadata (a file path, a row count), stored in Airflow's own metadata database — passing large DataFrames through them either fails outright or bloats the metadata DB. The correct pattern is writing data to external storage in one task and passing only the storage reference via XCom to the next.

**Q. A DAG task keeps failing on retry with the same error. What's your triage order?**
Check task logs first for the actual exception; verify the task is idempotent (a partial failure on retry #1 might have left state that breaks retry #2, e.g. a non-idempotent `INSERT` instead of `MERGE`); check upstream data/dependency sensors aren't the real blocker; only then consider resource limits (executor slot starvation, memory) as the cause.

## Traps
- Writing non-idempotent tasks (blind `INSERT` instead of upsert/overwrite-by-partition) — retries and backfills then corrupt data instead of safely reproducing it.
- Doing heavy computation inside the DAG file's top-level Python (outside a task) — this code re-runs on every scheduler parse cycle, not just at execution time, and can slow down or crash the scheduler.
- Using Sensors in default `poke` mode at scale without `reschedule`/deferrable mode — they hold a worker slot the whole time they wait, starving other tasks.
- Confusing "the DAG defines dependencies" with "the DAG does the work" — the actual computation should happen in the target system (Spark cluster, container), not inside the Airflow worker process itself.

## Related
[[orchestration-and-workflows]], [[training-pipelines]], [[data-pipeline-fundamentals]], [[databricks]], [[ci-cd-for-ml]]
