---
title: Databricks
type: entity
domain: data-engineering
roles: [data-scientist, ml-engineer, mlops-engineer]
difficulty: core
frequency: high
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# Databricks

## What it is
A managed lakehouse platform built around Apache Spark — notebooks, job scheduling, cluster management, Delta Lake storage, MLflow, and Unity Catalog governance in one workspace. For architecture and design-pattern depth see [[databricks-platform]], [[medallion-architecture]], [[unity-catalog-and-governance]]; this page covers the day-to-day workspace/notebook/cluster/jobs mechanics.

## Core concepts
- **Workspace & notebooks**: multi-language notebooks (Python/SQL/Scala/R cells in one notebook) attached to a running cluster; `%sql`, `%python` magics switch cell language, and `dbutils.widgets` parameterizes notebooks for reuse across jobs.
- **Clusters**: all-purpose clusters (interactive, shared, billed while idle) vs job clusters (spun up per job run, torn down after — the cost-correct choice for scheduled production pipelines). Autoscaling adds/removes workers within a configured range based on load.
- **DBFS / Unity Catalog volumes**: the filesystem abstraction over cloud object storage (S3/ADLS/GCS); Unity Catalog additionally layers governed, three-level namespacing (`catalog.schema.table`) and fine-grained access control on top of raw storage paths.
- **Delta Lake as the default table format**: every managed table is Delta by default — ACID transactions, time travel (`VERSION AS OF`), schema enforcement/evolution, and `MERGE INTO` for upserts, which is what makes medallion pipelines reliable on top of object storage.
- **Jobs & workflows**: a Job is one or more Tasks (notebook, JAR, Python script, dbt, SQL) with dependencies between them, a schedule/trigger, and retry policy — Databricks' native orchestrator, distinct from but overlapping with Airflow for cross-system pipelines.
- **`dbutils`**: the notebook utility API — `dbutils.fs` (filesystem ops), `dbutils.secrets` (secret scopes, never hardcode credentials), `dbutils.widgets` (parameters), `dbutils.notebook.run` (notebook chaining).
- **Repos / Databricks Asset Bundles**: git-integrated version control for notebooks and, increasingly, DABs as the infra-as-code way to define and deploy jobs/clusters across environments (dev/staging/prod) reproducibly.

## Code
```python
# Reading a Unity Catalog table and writing a Delta upsert (Silver layer pattern)
df = spark.table("bronze.events.raw_clicks")

cleaned = (df
    .dropDuplicates(["event_id"])
    .filter("event_ts IS NOT NULL")
)

cleaned.createOrReplaceTempView("staged")

spark.sql("""
MERGE INTO silver.events.clicks AS target
USING staged AS source
ON target.event_id = source.event_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

# dbutils examples
secret = dbutils.secrets.get(scope="prod-scope", key="api-token")
dbutils.widgets.text("run_date", "2026-09-13")
run_date = dbutils.widgets.get("run_date")
```

## When to use it vs alternatives
- **vs raw open-source Spark on EMR/self-managed clusters**: Databricks trades some cost premium for managed cluster lifecycle, Delta/Unity Catalog integration, collaborative notebooks, and native MLflow — worth it when the team values less infra ops; self-managed Spark wins on cost control and avoiding vendor lock-in at large scale.
- **vs Snowflake**: Snowflake is a stronger pure-SQL warehouse experience with less notebook/ML tooling built in; Databricks is the better fit when ML/data-science workloads (not just BI/SQL) dominate.
- **vs plain Airflow + Spark clusters**: Databricks Jobs cover Spark-centric orchestration natively; Airflow is still preferred for cross-system orchestration (triggering non-Databricks systems, complex DAG dependency logic) — many shops run both, Airflow calling into Databricks jobs.

## Interview angle
**Q. All-purpose cluster vs job cluster — when do you use which, and why does it matter for cost?**
All-purpose clusters are for interactive development, stay up (and billed) between uses, and are typically shared by a team. Job clusters are created fresh for a scheduled job run and terminated immediately after — the correct default for production pipelines, since you're not paying for idle time and each run gets a clean environment.

**Q. How does Databricks avoid the classic "small files" problem on cloud object storage?**
Delta Lake's `OPTIMIZE` command compacts small files into larger ones, and `Z-ORDER` co-locates related data for faster file skipping on filtered queries; without periodic `OPTIMIZE`, high-frequency streaming writes fragment into thousands of tiny files that tank read performance.

**Q. Why would you use Unity Catalog instead of just DBFS paths for table access?**
Unity Catalog centralizes governance (row/column-level security, lineage, audit logs) across workspaces at the catalog/schema/table level instead of per-path ACLs on cloud storage, and provides a queryable data lineage graph — important once multiple teams and compliance requirements are in play, not just single-user notebooks.

## Traps
- Leaving all-purpose clusters running (auto-termination misconfigured) — the most common source of surprise Databricks bills.
- Writing directly to cloud storage paths instead of through Unity Catalog-governed tables — bypasses access control and lineage tracking.
- Treating notebook execution order as guaranteed without checking — cells can be re-run out of order interactively, producing state that doesn't match a fresh top-to-bottom run (a common cause of "works in my notebook, fails as a job").
- Ignoring `MERGE INTO` in favor of full-table overwrites for incremental updates — correct only for genuinely small tables; at scale it's wasteful and loses time-travel granularity.

## Related
[[databricks-platform]], [[medallion-architecture]], [[unity-catalog-and-governance]], [[delta-lake]], [[apache-spark]], [[mlflow]]
