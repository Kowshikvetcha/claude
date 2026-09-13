---
title: dbt
type: entity
domain: data-engineering
roles: [data-scientist, ml-engineer, mlops-engineer]
difficulty: core
frequency: medium
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# dbt

## What it is
A SQL-based transformation framework that brings software-engineering practice — version control, testing, documentation, dependency management — to the "T" in ELT. You write transformations as SQL `SELECT` statements; dbt compiles them, figures out execution order from the references between them, and materializes them as tables/views in the warehouse or lakehouse.

## Core concepts
- **Models**: a `.sql` file containing a `SELECT` statement is a "model" — dbt handles the DDL/DML (`CREATE TABLE AS`, `MERGE`, etc.) needed to materialize it; you never write `CREATE TABLE` yourself.
- **`ref()` and the DAG**: instead of hardcoding a table name, a model references another model via `{{ ref('other_model') }}` — dbt parses these references to build a dependency DAG and runs models in the correct order automatically, and `ref()` also makes environment promotion (dev → staging → prod schema) transparent since it resolves to the right underlying table per environment.
- **Materializations**: `view` (default, cheap, recomputed on every query), `table` (materialized fully on each run), `incremental` (only processes new/changed rows since the last run — the standard choice for large fact tables where a full rebuild is wasteful), `ephemeral` (inlined as a CTE into downstream models, never materialized itself).
- **Tests**: declarative data-quality checks in YAML (`not_null`, `unique`, `relationships`, `accepted_values`) plus custom SQL tests — `dbt test` fails the build if data violates these, making data-quality checks a first-class, version-controlled part of the pipeline rather than an afterthought. See [[data-quality-and-validation]].
- **Sources & freshness**: `sources` declare raw upstream tables dbt doesn't create, letting you test and document them and check freshness (has this source been updated recently) without owning their ingestion.
- **Jinja + macros**: models are SQL with Jinja templating on top — macros let you DRY up repeated SQL logic (e.g. a standard way to deduplicate) across many models.
- **Lineage/docs**: `dbt docs generate` produces a browsable DAG and column-level documentation automatically derived from the `ref()` graph and YAML metadata — this is what makes dbt's "who depends on this table" question answerable without spelunking through pipeline code.

## Code
```sql
-- models/staging/stg_orders.sql
select
    order_id,
    customer_id,
    order_ts,
    status
from {{ source('raw', 'orders') }}
where order_id is not null

-- models/marts/fct_daily_orders.sql
{{ config(materialized='incremental', unique_key='order_date_customer') }}

select
    date_trunc('day', order_ts) as order_date,
    customer_id,
    order_date || '-' || customer_id as order_date_customer,
    count(*) as order_count
from {{ ref('stg_orders') }}
{% if is_incremental() %}
where order_ts > (select max(order_date) from {{ this }})
{% endif %}
group by 1, 2
```
```yaml
# models/staging/schema.yml
models:
  - name: stg_orders
    columns:
      - name: order_id
        tests: [not_null, unique]
      - name: status
        tests:
          - accepted_values:
              values: ['pending', 'shipped', 'cancelled']
```

## When to use it vs alternatives
- **vs writing raw SQL scripts / stored procedures**: raw SQL scripts have no dependency graph, no built-in testing, and no lineage documentation — dbt is worth adopting the moment more than a handful of interdependent transformation steps exist, which is almost always.
- **vs Spark/PySpark for transformation logic**: dbt is SQL-only and warehouse/lakehouse-native (runs the compiled SQL where your data already lives — Snowflake, BigQuery, Databricks SQL); Spark/PySpark is the right tool when transformations need non-SQL logic (custom Python, ML feature engineering, complex UDFs) or need to scale beyond what the SQL engine handles well. Many medallion-architecture stacks use PySpark for Bronze→Silver and dbt for Silver→Gold, or dbt end-to-end when the warehouse is SQL-native.
- **vs Airflow for orchestration**: dbt orchestrates *within* its own DAG of SQL models; it does not replace Airflow for cross-system scheduling (dbt is commonly one task inside a larger Airflow DAG that also handles ingestion and downstream triggers).

## Interview angle
**Q. Why prefer `incremental` materialization over rebuilding a table from scratch every run?**
For a large fact table, a full rebuild reprocesses the entire history on every run — costly in compute/warehouse credits and slow. Incremental models filter to only new/changed rows (typically via a watermark on a timestamp column) and append/merge just that delta, keeping run time roughly proportional to new data volume rather than total history.

**Q. How does dbt's `ref()` function support promoting the same codebase from dev to production?**
`ref()` doesn't hardcode a fully-qualified table name — dbt resolves it at compile time based on the active target/profile (which schema/database the current environment points to), so the identical model code runs against a dev schema in development and the production schema in a scheduled production run, without any find-and-replace.

**Q. A `not_null` test starts failing in production after a schema change upstream. What's your process?**
Treat it as the pipeline correctly catching a real data-quality regression, not a nuisance to bypass — trace which source or upstream model introduced the nulls (dbt's lineage graph narrows this quickly), decide whether the fix belongs upstream (source system) or in a staging model (defensive filtering/coalescing), and only relax the test itself if nulls are now a legitimate, expected state that downstream models must explicitly handle.

## Traps
- Using `table` materialization by default for large fact tables that get reprocessed daily — wastes compute that `incremental` would avoid; but reaching for `incremental` everywhere adds complexity where a cheap `view`/`table` would do for small dimension data.
- Writing business logic without corresponding tests — a model with no `not_null`/`unique`/`relationships` tests gives no early warning when upstream data quality degrades.
- Referencing a raw source table directly in a downstream model instead of through a staging model — couples every consumer to the raw schema and loses the single place to handle upstream schema drift.
- Forgetting that `ephemeral` models are inlined as CTEs on every downstream query — overusing them for expensive logic recomputes that logic repeatedly instead of materializing it once.

## Related
[[medallion-architecture]], [[data-modeling-star-schema]], [[data-quality-and-validation]], [[data-pipeline-fundamentals]], [[apache-airflow]]
