---
title: Data Engineering — Map of Content
type: map
domain: data-engineering
roles: [ml-engineer, mlops-engineer, data-scientist, fde]
updated: 2026-09-13
---

# Data Engineering — Map of Content

## Why this domain is asked
Given the vault owner's Databricks/PySpark/medallion background, this domain is a strength area but also a common interview blind spot for candidates who only know pandas — Spark architecture and medallion-style pipeline design come up constantly at Indian companies running Databricks or similar lakehouse stacks. Expect it woven into system-design rounds and occasionally as its own dedicated round for ML/MLOps Engineer roles.

## Curriculum
| # | Page | Why it matters | Difficulty |
|---|---|---|---|
| 1 | [[data-pipeline-fundamentals]] | ETL/ELT vocabulary every later topic assumes | core |
| 2 | [[batch-vs-streaming]] | Fundamental architectural fork for any pipeline design | core |
| 3 | [[file-formats-parquet-avro]] | Explains why columnar formats dominate lakehouse storage | core |
| 4 | [[warehouse-vs-lake-vs-lakehouse]] | Framing question before any storage-architecture design | core |
| 5 | [[spark-architecture]] | Driver/executor/DAG model behind every Spark question | core |
| 6 | [[pyspark-essentials]] | Directly maps to the vault owner's daily-driver tool | core |
| 7 | [[partitioning-and-shuffling]] | Root cause of most real-world Spark performance problems | intermediate |
| 8 | [[spark-performance-tuning]] | Practical debugging skill, heavily probed at senior level | advanced |
| 9 | [[delta-lake]] | ACID transactions and time travel on top of a data lake | core |
| 10 | [[medallion-architecture]] | Bronze/silver/gold — the dominant Indian lakehouse pattern | core |
| 11 | [[dlt-declarative-pipelines]] | Modern declarative alternative to hand-rolled orchestration | intermediate |
| 12 | [[databricks-platform]] | Platform-specific fluency directly relevant to the target stack | core |
| 13 | [[data-modeling-star-schema]] | Classic warehouse modeling, still asked for analytics systems | intermediate |
| 14 | [[slowly-changing-dimensions]] | Common dimensional-modeling interview trap | intermediate |
| 15 | [[kafka-and-event-streaming]] | Standard real-time ingestion backbone | intermediate |
| 16 | [[data-quality-and-validation]] | Guardrails that keep a pipeline trustworthy | intermediate |

## How it's tested per role
- **ML Engineer**: tested on the feature-pipeline slice — Spark performance, medallion layering, partitioning — as the boundary with model training.
- **MLOps Engineer**: tested on orchestration and reliability — data quality gates, pipeline scheduling, and how bad data gets caught before it reaches a model.
- **Data Scientist**: tested lightly on Spark/pandas-scale reasoning, mostly to confirm they can self-serve features rather than depend entirely on engineering.
- **FDE**: needs broad platform fluency (lakehouse concepts, streaming vs batch) to scope client engagements quickly, more than deep tuning expertise.

## Question bank
See [[qbank-data-engineering]] for the drilled question set.

## Related domains
- [[moc-sql]]
- [[moc-mlops]]
- [[moc-system-design]]
- [[moc-programming]]
