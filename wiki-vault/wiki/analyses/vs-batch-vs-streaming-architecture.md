---
title: Batch vs Streaming Architecture
type: analysis
domain: data-engineering
roles: [ml-engineer, mlops-engineer, data-scientist]
difficulty: advanced
frequency: medium
status: drafted
tags: [batch, streaming, kafka, lambda-architecture, kappa-architecture, data-engineering]
updated: 2026-09-11
sources: []
---

# Batch vs Streaming Architecture

## TL;DR
Batch processes bounded data on a schedule — cheap, simple, and correct-by-default, but adds
staleness equal to the batch interval. Streaming processes unbounded data continuously as events
arrive — low latency, but real operational complexity: state management, exactly-once semantics,
late/out-of-order events, and a much harder debugging story. This page goes past the
[[batch-vs-streaming]] concept page to the architectural decision of *running both* — Lambda
architecture (separate batch + speed layers) vs Kappa architecture (one streaming pipeline that
also serves batch-style reprocessing) — and what that costs in practice.

## The real question being asked
Anyone can say "streaming is faster." The interviewer wants to know whether you understand that
streaming is not a strictly-better upgrade of batch — it trades simplicity and cost for latency,
and most business problems don't need sub-second freshness. They're also testing whether you know
the two classic ways teams try to get both batch's correctness and streaming's freshness (Lambda
and Kappa), and whether you can name what actually breaks in each: Lambda's dual-codebase drift,
Kappa's harder reprocessing story and higher baseline infra cost.

## Side by side

| Dimension | Batch | Streaming |
|---|---|---|
| Latency | Minutes to hours (the batch interval) | Sub-second to seconds |
| Throughput model | Large bounded jobs, high throughput per run | Continuous, per-event or micro-batch |
| Cost profile | Pay for compute only during the run window | Cluster/broker running 24/7, higher baseline cost |
| Complexity | Straightforward: read, transform, write, done | State stores, watermarks, exactly-once/at-least-once semantics, backpressure |
| Failure recovery | Re-run the job — idempotent by construction if written well | Needs checkpointing, offset management, replay from a durable log |
| Late/out-of-order data | Naturally handled — you just wait for the batch window to close | Needs explicit watermarking and windowing logic, see [[kafka-and-event-streaming]] |
| Debugging | Easy — deterministic runs over a fixed dataset | Hard — non-deterministic timing, hard to reproduce exact production conditions |
| Tooling maturity (India context) | Very mature — Spark/Databricks batch jobs are the default in most GCCs | Growing but scarcer expertise — fewer engineers deeply know Flink/Kafka Streams |
| Team/on-call burden | Lower — a failed nightly job pages someone during business hours usually | Higher — a stuck consumer group or lag spike needs on-call attention any time |
| Natural fit | Reporting, feature backfills, nightly model retraining, medallion ETL | Fraud/anomaly detection, real-time personalisation, monitoring/alerting |

## When Batch wins
- The business decision cadence is daily/weekly anyway (e.g. a demand forecast refreshed once a
  day) — sub-second freshness would be wasted engineering effort. See
  [[requirements-and-metrics-definition]].
- You want the simplest, most debuggable, most cheaply-operated system that meets the SLA — most
  ML training pipelines (feature backfills, [[training-pipelines]]) fall here.
- Your medallion architecture ([[medallion-architecture]]) already runs scheduled Bronze→Silver→
  Gold jobs and adding streaming would duplicate a working pipeline for no measurable business
  benefit.

## When Streaming wins
- The value of a decision decays fast — fraud scoring, real-time bidding, live recommendation
  re-ranking, operational alerting — where minutes of staleness directly cost money or safety.
  See [[batch-vs-realtime-inference]].
- You need continuously updated features for online inference (a real-time feature pipeline —
  see [[case-realtime-feature-pipeline]]) rather than features that are hours stale.
- Events naturally arrive as a stream (clickstream, IoT sensors, transaction logs) and batching
  them just to un-batch them at serving time adds latency with no benefit.

## The honest hybrid answer
**Lambda architecture** runs a batch layer (source of truth, reprocesses everything periodically,
corrects errors) and a speed layer (streaming, gives an approximate low-latency view) side by
side, merging results at query time. Its real cost is maintaining two codebases that must produce
consistent logic — the classic failure mode is the batch and streaming transforms silently
drifting apart (a bug fix applied to one path and not the other), producing different numbers for
the "same" metric depending on which layer answered the query.

**Kappa architecture** instead treats everything as a stream — including reprocessing, which is
done by replaying the event log (e.g. Kafka with long retention) through the same streaming job
rather than maintaining a separate batch pipeline. This avoids the dual-codebase drift problem but
requires your streaming engine to handle full historical reprocessing efficiently, and it commits
you to running streaming infrastructure even for workloads that don't strictly need low latency.

In practice, most teams don't pick one architecture and apply it everywhere — they run batch by
default and add a streaming path only for the specific features/signals where latency has
measurable business value, explicitly accepting the operational cost of running both. The honest
answer in an interview is naming that cost, not pretending streaming is free.

## Interview angle
**Q. Why would you not just make everything streaming, since it's strictly more real-time?**
Because streaming isn't strictly better — it trades simplicity and cost for latency. Running
Kafka/Flink infrastructure continuously costs more than a scheduled batch job that runs for twenty
minutes a night, it introduces failure modes batch doesn't have (consumer lag, out-of-order
events, exactly-once semantics), and it needs on-call coverage around the clock. If the downstream
consumer only checks the value once a day, sub-second freshness is wasted spend.

**Q. Compare Lambda and Kappa architecture and name a real failure mode of each.**
Lambda runs separate batch and speed layers merged at query time; its failure mode is logic drift
between the two codebases producing inconsistent results for the same metric. Kappa runs
everything as a stream, replaying the event log for reprocessing; its failure mode is that full
historical reprocessing through a streaming engine is often slower and more resource-intensive
than a purpose-built batch job, and you're paying streaming infra costs even for workloads that
don't need it.

**Follow-up.** How would you decide, for a specific feature, whether it needs the streaming path?
Quantify the cost of staleness in business terms (lost revenue, missed fraud, degraded UX) against
the interval a batch job would give, and compare that to the added infra/on-call cost of a
streaming path for just that feature — this is the same latency-vs-cost tradeoff as
[[latency-and-throughput-budgets]], applied at the feature level rather than the whole system.

## Related
[[batch-vs-streaming]]
[[kafka-and-event-streaming]]
[[batch-vs-realtime-inference]]
[[medallion-architecture]]
[[case-realtime-feature-pipeline]]
[[data-pipeline-fundamentals]]
[[scalability-patterns]]
