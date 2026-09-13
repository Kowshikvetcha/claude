---
title: Data Engineering Question Bank
type: qbank
domain: data-engineering
roles: [ml-engineer, mlops-engineer, data-scientist, fde]
difficulty: core
frequency: high
status: drafted
tags: [question-bank]
updated: 2026-09-13
---

# Data Engineering Question Bank

> How to use: cover the answers, write yours first, then compare. Anything you fumble → the linked concept page's `status` should go back to `drafted`.

## Warm-up

### Q1. What's the actual difference between ETL and ELT, and why did the industry move toward ELT?
**Answer.** ETL transforms data in a separate processing tier before it ever lands in the warehouse — necessary when the warehouse itself was too weak or expensive to run heavy transforms. ELT lands raw data first (cheap object storage) and transforms it in-place using the warehouse/lakehouse's own compute (Spark, warehouse SQL engine). The shift happened because storage got cheap, compute got elastic and separable from storage, and keeping a raw copy around lets you re-derive any downstream transform when business logic changes — you're not locked into whatever the ETL job decided to keep.
**Follow-ups.** What do you lose by going full ELT? → Raw, untransformed data sitting in your lake is now inside your compliance/PII boundary even before anyone decided it should be — you need governance (masking, access control) applied earlier than in a classic ETL world.
**Page.** [[data-pipeline-fundamentals]]

### Q2. When would you choose streaming over batch for a pipeline, and what does that choice cost you operationally?
**Answer.** Choose streaming when the business action tied to the data has a latency requirement batch can't hit — fraud blocking, real-time personalization, alerting. Batch is the default otherwise: simpler to reason about, easier to backfill, cheaper per unit of data processed, and idempotent reruns are trivial. Streaming costs you: always-on infrastructure (no scale-to-zero), harder exactly-once semantics, and debugging is fundamentally different (you can't just re-run yesterday's job against a fixed input — the input is a moving window).
**Follow-ups.** What's a middle ground many teams land on? → Micro-batch / structured streaming with a trigger interval (e.g. Spark Structured Streaming's `trigger(processingTime=...)` or `availableNow`) — streaming semantics and checkpointing, batch-like debuggability and cost profile.
**Page.** [[batch-vs-streaming]]

### Q3. Why does Parquet dominate lakehouse storage over row-oriented formats like Avro for analytical workloads?
**Answer.** Parquet is columnar: a query that touches 3 of 50 columns only reads those 3 columns' bytes off disk, and columnar layout compresses far better (similar values adjacent) than row-major layout. Avro is row-oriented and schema-embedded per file, which makes it the right choice for write-heavy, full-row-read workloads (Kafka message serialization, CDC event streams) but wrong for scan-heavy analytics where you're aggregating over a handful of columns across billions of rows.
**Follow-ups.** Where does Avro actually win over Parquet in a Databricks pipeline? → As the wire format for Kafka topics feeding into a bronze layer — schema evolution is well-supported, and you're consuming whole records anyway, so columnar pruning buys nothing at that stage.
**Page.** [[file-formats-parquet-avro]]

### Q4. Draw the line between a data warehouse, a data lake, and a lakehouse.
**Answer.** A warehouse stores structured data in a proprietary, optimized format with strong schema enforcement and fast SQL — but it's expensive per TB and bad at unstructured data. A lake stores anything (files, any format) cheaply in object storage with no enforced schema — flexible, but you get "data swamp" reliability problems (no ACID, no schema guarantees, no time travel). A lakehouse (Delta Lake, Iceberg, Hudi on top of a lake) adds a transactional metadata layer over lake storage, giving you warehouse-grade ACID, schema enforcement and BI-tool performance while keeping lake-grade cost and format flexibility.
**Follow-ups.** What's the one warehouse feature a lakehouse still typically lags on? → Sub-second point-lookup latency for highly concurrent OLTP-adjacent serving — lakehouses are still optimized for scan-heavy analytical access patterns, not row-level transactional serving.
**Page.** [[warehouse-vs-lake-vs-lakehouse]]

### Q5. Walk through what happens, physically, when you submit a Spark job — driver, executors, DAG.
**Answer.** The driver runs your `main()`, builds a logical plan from your transformations, and the Catalyst optimizer turns it into a physical plan of stages split at shuffle boundaries. The cluster manager (YARN/Kubernetes/Databricks' own) allocates executors, each with its own JVM and a fixed number of cores/slots. The driver schedules tasks (one per partition) onto executor cores; each stage's tasks must all finish before the next stage (which depends on shuffled output) can start. The driver is a single point of coordination — if it OOMs (e.g. from a `.collect()` on a huge dataset) the whole job dies regardless of executor health.
**Follow-ups.** Why is a stage boundary always a shuffle boundary? → Because only an operation that needs data from *other* partitions (`groupBy`, `join`, `repartition`) needs to redistribute data across the cluster; anything that can be computed partition-locally (`map`, `filter`) stays in the same stage.
**Page.** [[spark-architecture]]

## Core

### Q6. Why does `df.groupBy("key").agg(...)` trigger a shuffle but `df.filter(...)` doesn't, and what determines how much data actually moves?
**Answer.** `filter` is a narrow transformation — each output partition depends only on the corresponding input partition, so it executes in place with no network movement. `groupBy` is wide — all rows sharing a key must end up co-located on one executor to be aggregated together, and since keys are scattered arbitrarily across input partitions, Spark must write shuffle files and pull matching keys across the network (a hash partition exchange). The volume moved is roughly the full dataset size minus whatever a map-side partial aggregation (Spark does this automatically for associative aggregates like `sum`/`count`) manages to combine before the shuffle.
**Follow-ups.** How does a broadcast join avoid this shuffle entirely? → If one side is small enough to fit in executor memory, Spark ships a full copy of it to every executor instead of shuffling both sides — turning an expensive shuffle join into a local, shuffle-free hash-join per partition.
**Page.** [[partitioning-and-shuffling]]

### Q7. Explain lazy evaluation in PySpark and why calling `.count()` mid-pipeline for debugging can be a performance trap.
**Answer.** Transformations (`select`, `filter`, `withColumn`) just build up a logical plan (a DAG of lineage) — nothing executes until an action (`.collect()`, `.count()`, `.write()`) forces evaluation. This lets Catalyst optimize the *whole* chain (predicate pushdown, column pruning) rather than each step in isolation. The trap: every `.count()` you sprinkle in for debugging is a separate action that re-executes the DAG from the last cached point, or from scratch if nothing's cached — so five debug `.count()` calls can silently mean five full re-reads of your bronze table.
**Follow-ups.** What's the fix if you genuinely need to inspect intermediate state without paying that cost repeatedly? → `.cache()` or `.persist()` the intermediate DataFrame before the debug actions, and unpersist it once you're done — trading executor memory for avoiding repeated recomputation.
**Page.** [[pyspark-essentials]]

### Q8. You have a join between a 500M-row fact table and a 400M-row dimension table that's badly skewed on the join key — one key has 40% of all rows. What do you do?
**Answer.** A standard shuffle hash join sends every row for a given key to one executor task — the skewed key creates one task doing 40% of the work while others idle, so wall-clock time is dominated by that one straggler. Fixes: (1) salting — append a random suffix (0..N) to the skewed key on both sides and explode the dimension side N-ways, spreading the hot key's rows across N tasks; (2) Adaptive Query Execution's skew join optimization (enabled by default on Databricks Runtime), which detects oversized partitions at runtime and automatically splits them; (3) isolate the hot key with a `filter`, broadcast-join just that slice, `union` the results back with the non-skewed remainder handled by a normal join.
**Follow-ups.** Why doesn't simply increasing `spark.sql.shuffle.partitions` fix key skew? → More partitions only helps if the *data* is more evenly distributable across them — a single hot key still lands entirely in one partition no matter how many partitions exist, so the straggler task is unaffected.
**Page.** [[spark-performance-tuning]]

### Q9. What does Delta Lake actually add on top of plain Parquet files in object storage, mechanically?
**Answer.** A `_delta_log` directory of JSON/checkpoint commit files that record every add/remove of underlying Parquet files as an atomic, ordered transaction — this is what gives ACID guarantees (a reader either sees a whole committed version or the prior one, never a half-written state) and time travel (query any prior version by replaying the log up to that commit). It also enables `MERGE`/`UPDATE`/`DELETE` on what is otherwise immutable columnar storage, by writing new Parquet files for changed rows and atomically swapping which files are "active" in the log — old files aren't rewritten in place, they're marked removed.
**Follow-ups.** Why does this make `VACUUM` a dangerous command to run carelessly? → It physically deletes files no longer referenced by the current log, which breaks time travel to any version older than the retention window — running it with a short retention right after a bad write means you can no longer time-travel back past the mistake.
**Page.** [[delta-lake]]

### Q10. Design the bronze/silver/gold layering for a pipeline ingesting raw clickstream events destined for an ML feature table.
**Answer.** Bronze: raw events landed as-is (JSON/Avro from Kafka), append-only, schema drift tolerated, kept for replay/audit — no business logic applied. Silver: parsed, deduplicated, type-cast, bad/malformed records quarantined, joined with reference/dimension data — this is the "single source of truth" layer, still event-grained. Gold: aggregated to the grain a consumer actually needs — e.g. a feature table keyed by `(user_id, feature_date)` with rolling window aggregates — and this is what MLflow-tracked training jobs and serving pipelines read from, never bronze or silver directly.
**Follow-ups.** Why keep silver as a separate layer instead of going bronze → gold directly? → Multiple gold tables (features, BI dashboards, ad-hoc analytics) all need the same cleaned, deduplicated event data — computing that once in silver avoids re-implementing (and inevitably diverging) the same cleaning logic in every downstream consumer.
**Page.** [[medallion-architecture]]

### Q11. What does Unity Catalog actually solve that per-workspace Hive metastores didn't?
**Answer.** A classic Hive metastore is scoped to one workspace — the same table name in two workspaces is two unrelated tables with no shared governance, and access control is coarse (workspace-level, not table/column/row-level). Unity Catalog gives one account-level catalog of catalogs → schemas → tables, fine-grained GRANT-based access control (down to column masking and row filters) enforced consistently across every workspace and every compute engine (notebooks, jobs, SQL warehouses), plus lineage tracking that follows a table through every downstream transform and ML model that consumed it.
**Follow-ups.** Why does that lineage matter specifically for an ML team? → When a model's predictions suddenly drift, lineage lets you trace the feature table back through every upstream transform to find which source table's schema or distribution changed — without it you're grepping notebook history.
**Page.** [[databricks-platform]]

### Q12. Design a star schema for an e-commerce orders analytics table. What goes in the fact table vs. the dimensions?
**Answer.** Fact table: one row per order line item, holding only measures (quantity, unit_price, discount, tax) and foreign keys to every dimension (date_key, customer_key, product_key, store_key) — kept as narrow and numeric as possible since it's the largest, most-scanned table. Dimensions: `dim_customer`, `dim_product`, `dim_date`, `dim_store` — wide, descriptive, slowly changing, and small relative to the fact table, denormalized on purpose (a product's category name lives directly on `dim_product` rather than requiring another join to a category table) so that BI queries need fewer joins.
**Follow-ups.** Why is a snowflake schema (normalizing dimensions further) usually a mistake for BI workloads? → It reintroduces the extra joins star schema exists to avoid, for a normalization benefit (saved storage, single source of truth for repeated attributes) that barely matters at dimension-table scale — you're paying query complexity to save space you don't need to save.
**Page.** [[data-modeling-star-schema]]

## Hard

### Q13. Implement a Type 2 slowly changing dimension for `dim_customer` when a customer's address changes. What columns do you need and how does a query "as of" a past date work?
**Answer.** Add `effective_start_date`, `effective_end_date` (or `NULL`/a sentinel like `9999-12-31` for the current row) and `is_current` to `dim_customer`. On an address change: close out the existing row (`UPDATE ... SET effective_end_date = today, is_current = false WHERE customer_id = X AND is_current = true`), then `INSERT` a new row with the new address, `effective_start_date = today`, `is_current = true`. A fact table joins on `customer_key` (a surrogate key, not the natural `customer_id`) to whichever dimension row's date range contained the fact's transaction date — this is what lets you correctly attribute a historical order to the customer's address *at the time of that order*, not their current address.
**Follow-ups.** Why can't you just join facts to dimensions on the natural key and filter by date in the dimension? → Because the natural key (`customer_id`) now maps to multiple dimension rows over time — you'd need the same date-range join logic anyway; the surrogate key exists precisely so the fact table can point at one immutable, unambiguous dimension row per transaction.
**Page.** [[slowly-changing-dimensions]]

### Q14. A Kafka topic feeding your bronze layer has 12 partitions and your Spark Structured Streaming job has `maxOffsetsPerTrigger` unset. What can go wrong, and how does consumer parallelism actually map to partitions?
**Answer.** Kafka partitions are the unit of parallelism and ordering: within one partition, order is guaranteed; across partitions, none is. A Spark Structured Streaming job maps each Kafka partition to a Spark task per micro-batch, so parallelism tops out at 12 no matter how many executor cores you have (13th core sits idle for this stage). With `maxOffsetsPerTrigger` unset, the first trigger after a long outage or a fresh start with `earliest` offsets can try to pull an enormous, unbounded backlog into one micro-batch, causing an OOM or a multi-hour first batch. Setting a bounded per-trigger offset limit turns backlog catch-up into many small, safe, resumable micro-batches instead.
**Follow-ups.** Why can't you get exactly-once end-to-end just from Kafka's own delivery guarantees? → Kafka's producer idempotence and consumer offset commits only guarantee exactly-once *within* Kafka; the sink (your bronze Delta table) needs its own idempotent write path (checkpointed offsets + a deterministic, replay-safe write, which Delta's transaction log provides) to make the whole read-process-write chain exactly-once.
**Page.** [[kafka-and-event-streaming]]

### Q15. Design the data quality gates for a medallion pipeline so a bad upstream schema change or a burst of null keys never reaches the gold feature table an XGBoost training job reads from.
**Answer.** Put validation at the bronze→silver boundary, not earlier (bronze must stay a faithful raw copy) and not only at gold (too late — silver consumers are already poisoned). Define expectations per table: schema conformance (expected columns/types present), null-rate thresholds on key columns, referential checks (foreign keys resolve to a known dimension row), and distributional checks (a numeric column's range/mean hasn't jumped outside a historical band). On Databricks this is commonly `DLT` expectations (`@dlt.expect_or_drop`, `@dlt.expect_or_fail`) or a library like Great Expectations run as a job step; failing records get quarantined to a `_quarantine` table for inspection rather than silently dropped or silently let through.
**Follow-ups.** Why is "expect_or_fail the whole pipeline" sometimes worse than "expect_or_drop and alert"? → A hard pipeline failure on any bad row means one malformed record blocks every downstream consumer's daily refresh; for most feature pipelines, quarantining the bad rows and alerting lets the 99.9% of good data keep flowing while someone investigates the exception — the right choice depends on whether partial-fresh data is more dangerous than stale data for that specific downstream model.
**Page.** [[data-quality-and-validation]]

### Q16. When would you reach for Delta Live Tables instead of hand-rolled PySpark jobs orchestrated by a scheduler, and what do you give up?
**Answer.** DLT lets you declare *what* each table should be (a SQL/PySpark query defining bronze/silver/gold transformations plus expectations) and it figures out *how* — dependency ordering, incremental vs. full refresh, retries, and a managed streaming/batch execution model — rather than you hand-writing a DAG of jobs with manual checkpoint management and manual dependency wiring in an orchestrator. You give up some low-level control: custom retry logic, arbitrary non-tabular side effects mid-pipeline, and some flexibility around exactly when/how a table materializes, in exchange for a lot less orchestration boilerplate and built-in data quality enforcement and lineage.
**Follow-ups.** Why might a team still choose hand-rolled orchestration (e.g. Airflow + plain PySpark jobs) for parts of the pipeline? → When steps genuinely aren't just "produce a table" — calling an external API, triggering a non-Databricks system, or needing conditional branching logic that doesn't map cleanly onto DLT's declarative table-dependency model.
**Page.** [[dlt-declarative-pipelines]]

### Q17. You need to backfill three months of a gold feature table after fixing a bug in the silver-layer transform, without breaking the models currently serving off the old (buggy) feature values. Outline the approach.
**Answer.** Never overwrite the gold table in place while it's live: write the corrected backfill to a new version/table (`feature_table_v2` or a new Delta version), validate it against known-good spot checks and against the old table's row counts/distributions for the overlapping period, then cut over consumers (training jobs, serving) via a pointer/alias swap rather than a destructive rewrite — Delta Lake's own versioning means even an in-place `MERGE` backfill is recoverable via time travel if you catch a problem, but a pointer-based cutover means a bad backfill never touches what's currently serving traffic. Coordinate the model side separately: a model trained on the old feature values needs re-training or at least re-evaluation before it trusts the corrected features, since the bug may have been implicitly "learned around."
**Follow-ups.** Why is re-training sometimes non-negotiable even if the "fix" seems purely to make features more correct? → If the model was trained on the buggy distribution, its learned weights are calibrated to that (wrong) distribution — serving it corrected features at inference time without re-training can silently degrade performance more than the original bug did, because now train/serve feature distributions actively disagree.
**Page.** [[medallion-architecture]]

## Related
See [[moc-data-engineering]].
