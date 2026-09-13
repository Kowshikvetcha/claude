---
title: Apache Spark
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

# Apache Spark

## What it is
The dominant distributed data-processing engine for large-scale ETL, feature engineering, and batch ML — it lets you write transformations that look like single-machine DataFrame code while the engine parallelizes execution across a cluster. For deep architecture and tuning material see [[spark-architecture]], [[pyspark-essentials]], [[spark-performance-tuning]]; this page is the practical entity overview.

## Core concepts
- **Driver / executors**: the driver builds the logical plan and coordinates; executors run tasks on partitions of data in parallel across worker nodes. Understanding this split explains most Spark performance questions (what runs where, what gets shipped back to the driver).
- **Lazy evaluation + DAG**: transformations (`select`, `filter`, `groupBy`) build a logical plan; nothing executes until an action (`show`, `collect`, `write`) triggers it. The Catalyst optimizer rewrites the plan before execution.
- **Partitions**: the unit of parallelism — data is split into partitions distributed across executors; too few partitions underutilizes the cluster, too many adds scheduling overhead. `repartition`/`coalesce` control this explicitly.
- **Wide vs narrow transformations**: narrow transformations (`filter`, `map`) need no data movement between partitions; wide transformations (`groupBy`, `join`, `distinct`) trigger a **shuffle** — data redistribution across the network, the single biggest cost driver in Spark jobs.
- **DataFrame API vs RDD**: DataFrames are the standard interface today — Catalyst-optimized, schema-aware, and usable from SQL, Python, or Scala; RDDs are the lower-level, unoptimized primitive underneath, rarely written directly anymore.
- **Caching**: `.cache()`/`.persist()` materializes a DataFrame in memory (and/or disk) across actions when it's reused multiple times — without it, Spark recomputes the whole lineage from source on every action.
- **Broadcast joins**: when one side of a join is small enough to fit in executor memory, Spark can broadcast it to every executor instead of shuffling both sides — the standard fix for a large-table/small-table join that's shuffling unnecessarily.

## Code
```python
from pyspark.sql import functions as F

events = spark.read.format("delta").load("/path/to/bronze/events")

daily = (events
    .filter(F.col("event_date") >= "2026-01-01")
    .groupBy("user_id", "event_date")
    .agg(F.count("*").alias("event_count"))
)

users = spark.table("dim_users")   # small dimension table

enriched = daily.join(F.broadcast(users), on="user_id", how="left")

enriched.write.format("delta").mode("overwrite").partitionBy("event_date").save("/path/to/silver/daily_activity")
```

## When to use it vs alternatives
- **vs pandas**: pandas is single-machine and simpler for anything that fits in memory on one box; Spark's overhead (JVM startup, distributed scheduling) makes it slower for small data but is the only sane choice once data exceeds single-machine memory — see [[vs-spark-vs-pandas]].
- **vs Ray**: Ray is more general-purpose distributed Python (arbitrary Python tasks, actors, ML training/serving); Spark is purpose-built and more mature for large-scale structured/SQL-style ETL. Teams increasingly use both — Spark for ETL, Ray for training/serving.
- **vs Dask**: Dask mirrors the pandas/NumPy API more closely and is lighter-weight for Python-native distributed compute; Spark has the deeper ecosystem (Delta Lake, structured streaming, SQL engine maturity) for production data platforms.

## Interview angle
**Q. A join is much slower than expected. How do you diagnose and fix it?**
Check the Spark UI's SQL/query plan for a shuffle-heavy join (`SortMergeJoin`) versus a `BroadcastHashJoin`; if one side is small, force a broadcast join with `F.broadcast()` or raise `spark.sql.autoBroadcastJoinThreshold`. If both sides are genuinely large, check for data skew (one key with disproportionately many rows) causing a few tasks to dominate wall-clock time — salting the skewed key is the standard fix.

**Q. Why does `.cache()` sometimes make a job slower, not faster?**
Caching costs memory and, if the data doesn't fit, spills to disk or evicts other cached data, adding overhead without benefit — it only pays off when the cached DataFrame is reused across multiple actions; caching something used exactly once just adds materialization cost for no reuse benefit.

**Q. Explain what a shuffle actually does at the physical level.**
A shuffle writes each task's output partitioned by key to local disk, then executors fetch the partitions relevant to them across the network for the next stage — it's both disk I/O and network-bound, which is why minimizing shuffle stages (via broadcast joins, pre-partitioning, combining aggregations) is the single highest-leverage Spark performance lever.

## Traps
- Calling `.collect()` on a large DataFrame — pulls the entire result to the driver's memory, which crashes or silently OOMs long before it "just returns slow."
- Assuming more partitions is always better — past a point, task scheduling overhead dominates actual work per task.
- Using UDFs (especially Python UDFs) where a built-in `F.*` function exists — Python UDFs break Catalyst's ability to optimize and force row-by-row serialization across the JVM/Python boundary.
- Not partitioning output files sensibly (`partitionBy`) on data that's queried by a predictable filter column — leads to full-table scans downstream.

## Related
[[spark-architecture]], [[pyspark-essentials]], [[spark-performance-tuning]], [[partitioning-and-shuffling]], [[delta-lake]], [[databricks]]
