---
title: PySpark Drill — 10 DataFrame Exercises
type: drill
domain: data-engineering
roles: [ml-engineer, mlops-engineer, data-scientist, fde]
difficulty: core
frequency: high
status: drafted
tags: [drill, pyspark, spark]
updated: 2026-09-13
---

# PySpark Drill — 10 DataFrame Exercises

Assumes a `spark` session already exists. See [[pyspark-essentials]] and
[[spark-performance-tuning]] for the concepts behind each exercise; [[spark-architecture]] for what
executors/stages/shuffles actually are.

## 1. Basic join + column selection

```python
from pyspark.sql import functions as F

orders = spark.table("orders")           # order_id, customer_id, amount, order_date
customers = spark.table("customers")     # customer_id, name, region

result = (
    orders.join(customers, on="customer_id", how="inner")
    .select("order_id", "name", "region", "amount")
)
result.show(5)
```

Prefer `on="customer_id"` (a string/list of join keys) over an explicit `==` condition when the key
name matches on both sides — it avoids a duplicate column in the output.

## 2. Broadcast join for a small dimension table

```python
from pyspark.sql import functions as F

# customers is small (fits comfortably in executor memory, e.g. < a few hundred MB)
result = orders.join(F.broadcast(customers), on="customer_id", how="left")
```

Broadcasting ships the whole small table to every executor, avoiding a shuffle of the large `orders`
table. Spark's cost-based optimizer will often do this automatically below
`spark.sql.autoBroadcastJoinThreshold` (default 10 MB) — the explicit hint matters when the
optimizer's size estimate is wrong (e.g. after several transformations) or the table is a bit above
the auto threshold but still safely broadcastable. See [[spark-performance-tuning]].

## 3. Window function: running total per customer

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

w = Window.partitionBy("customer_id").orderBy("order_date").rowsBetween(
    Window.unboundedPreceding, Window.currentRow
)

result = orders.withColumn("running_total", F.sum("amount").over(w))
```

## 4. Window function: rank within group, filter top-N

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

w = Window.partitionBy("customer_id").orderBy(F.col("amount").desc())

top_orders = (
    orders.withColumn("rnk", F.row_number().over(w))
    .filter(F.col("rnk") <= 3)
    .drop("rnk")
)
```

`row_number()` guarantees exactly 3 rows per customer even on ties; `rank()`/`dense_rank()` can
return more than 3 rows if amounts tie at the boundary — state which one the requirement actually
needs.

## 5. UDF vs native function — and why native wins

```python
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# UDF version — works, but slow: row-by-row Python serialization across the JVM/Python boundary
@F.udf(returnType=DoubleType())
def celsius_to_fahrenheit_udf(c):
    return c * 9.0 / 5.0 + 32.0

df_udf = orders.withColumn("temp_f", celsius_to_fahrenheit_udf(F.col("temp_c")))

# Native version — stays inside the JVM/Catalyst, vectorized, no serialization overhead
df_native = orders.withColumn("temp_f", F.col("temp_c") * F.lit(9.0) / 5.0 + F.lit(32.0))
```

A plain Python UDF forces Spark to serialize each row from the JVM to a Python worker process,
execute row-at-a-time, and serialize the result back — it also opts the whole expression out of
Catalyst's optimizations. Prefer built-in `functions` (or a **pandas UDF** with
`@F.pandas_udf`, which vectorizes via Arrow and is far cheaper than a plain row UDF) whenever the
logic can be expressed that way. Reach for a UDF only when the logic genuinely can't be expressed
with built-ins (a bespoke parsing rule, a call into a non-Spark Python library).

## 6. Handling data skew with salting

```python
from pyspark.sql import functions as F

# large_df has a few keys ("hot keys") with disproportionately many rows, causing skewed partitions
NUM_SALT_BUCKETS = 20

salted_large = large_df.withColumn("salt", (F.rand() * NUM_SALT_BUCKETS).cast("int"))
salted_small = small_df.withColumn(
    "salt", F.explode(F.array([F.lit(i) for i in range(NUM_SALT_BUCKETS)]))
)

result = salted_large.join(
    salted_small,
    on=[salted_large.join_key == salted_small.join_key, "salt"],
    how="inner",
).drop("salt")
```

Salting spreads a hot key across `NUM_SALT_BUCKETS` synthetic partitions on the large side, and
explodes the small side so each salted large row still finds its match — trading one huge skewed
partition for many balanced ones. Alternatives worth naming: Adaptive Query Execution's automatic
skew join handling (`spark.sql.adaptive.skewJoin.enabled`, on by default in modern Spark), or
isolating and broadcast-joining just the hot keys separately from the rest. See
[[spark-performance-tuning]] and [[partitioning-and-shuffling]].

## 7. Repartition vs coalesce

```python
# Increasing partitions before a wide, parallelism-hungry operation — full shuffle
df_repartitioned = df.repartition(200, "customer_id")

# Reducing partitions before writing output files — no shuffle, just merges partitions
df_coalesced = df.coalesce(10)
```

`repartition` triggers a full shuffle and can both increase or decrease partition count, and can
rebalance by key (`repartition(n, "col")`) to fix skew upstream of a join. `coalesce` only merges
existing partitions down, avoiding a shuffle, but cannot increase partition count or fix skew across
partitions that are already unevenly sized. Using `coalesce` to shrink partitions right before a
final write is the common production pattern (fewer, larger output files); using `repartition` to
force a shuffle-based rebalance is the fix for skew.

## 8. Deduplication keeping the latest record per key

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

w = Window.partitionBy("id").orderBy(F.col("updated_at").desc())

deduped = (
    df.withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") == 1)
    .drop("rn")
)
```

This is the standard Delta/medallion-architecture dedup pattern applied when merging CDC records
into a silver table — see [[delta-lake]] and [[medallion-architecture]].

## 9. Reading and interpreting a Spark UI stage

```python
# No code change needed to observe this — but this is the pattern that CAUSES a bad stage:
skewed = large_df.groupBy("rarely_varying_key").agg(F.sum("amount"))
skewed.explain(mode="formatted")   # inspect the physical plan before running
skewed.collect()                   # now check the Spark UI's Stages tab
```

**Q. What do you look for in the Spark UI Stages tab to diagnose a slow job?**
Open the stage and check the **task duration distribution** (median vs. max task time) — a huge gap
between median and max task duration is the signature of **skew**: most tasks finish in seconds
while one or two stragglers run for minutes because they got a disproportionate share of the data.
Also check **shuffle read/write size** per stage (a large, unexpected shuffle often means a join or
`groupBy` exploded row counts or picked a bad partitioning), and **spill (memory and disk)** in the
task metrics, which signals the executor didn't have enough memory for the operation and fell back
to disk — a strong hint to repartition, increase executor memory, or reduce the data volume before
the wide operation. `explain(mode="formatted")` beforehand shows whether the planned join strategy
(broadcast vs. sort-merge) matches expectations.

## 10. Writing partitioned output efficiently

```python
(
    df.repartition("order_date")          # align in-memory partitions with the output partition column
    .write
    .mode("overwrite")
    .partitionBy("order_date")
    .format("delta")
    .save("/mnt/silver/orders")
)
```

Repartitioning by the same column used in `partitionBy` before writing avoids Spark scattering rows
for a single output partition across many tasks, which otherwise produces many small files per
partition (the "small files problem"). Also consider `.option("maxRecordsPerFile", ...)` or, on
Databricks, `OPTIMIZE` with `ZORDER` after the write. See [[delta-lake]] and
[[partitioning-and-shuffling]].

## Related

[[pyspark-essentials]] · [[spark-performance-tuning]] · [[spark-architecture]] ·
[[partitioning-and-shuffling]] · [[delta-lake]] · [[medallion-architecture]] ·
[[qbank-data-engineering]]
