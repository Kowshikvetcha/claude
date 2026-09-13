---
title: Apache Kafka
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

# Apache Kafka

## What it is
A distributed event-streaming platform — a durable, ordered, replayable log that decouples producers of data from consumers of it. In ML systems it's the usual backbone for real-time feature pipelines, online inference triggers, and streaming model-monitoring signals; anywhere data needs to move continuously between systems at low latency and high throughput.

## Core concepts
- **Topics & partitions**: a topic is a named stream of records; it's split into partitions, each an ordered, append-only log — partitioning is the unit of parallelism (consumers scale by partition count) and the unit of ordering guarantee (order is only guaranteed *within* a partition, not across the whole topic).
- **Producers, consumers, consumer groups**: producers append records to a topic (optionally choosing a partition key so related records land in the same partition, e.g. all events for one user); consumers read from partitions, and a consumer group divides a topic's partitions among its members so each partition is processed by exactly one consumer in the group at a time — this is how Kafka scales consumption horizontally.
- **Offsets**: each consumer tracks its position (offset) in each partition it reads; committing offsets (automatically or manually) is what makes "resume where I left off" and "reprocess from an earlier point" both possible — this replayability is Kafka's key advantage over a plain message queue that discards messages once delivered.
- **Retention, not delivery-based deletion**: unlike a traditional queue, Kafka retains records for a configured time/size window regardless of whether they've been consumed — multiple independent consumers can read the same topic at their own pace, and a new consumer can replay history within the retention window.
- **Replication & durability**: each partition is replicated across brokers (a leader plus followers); `acks=all` on the producer waits for the required number of replicas to acknowledge before considering a write durable — the durability/latency tradeoff a producer configures explicitly.
- **Kafka Streams / ksqlDB**: stream-processing libraries built on top of Kafka for transforming, aggregating, or joining streams in-flight (windowed aggregations, stream-stream joins) without a separate processing cluster — relevant when the transformation itself needs to happen continuously, not just the transport.
- **Schema registry**: commonly paired with Kafka (Avro/Protobuf + a schema registry) to enforce and evolve message schemas safely across producers/consumers that are deployed independently and can't be upgraded in lockstep.

## Code
```python
from confluent_kafka import Producer, Consumer

producer = Producer({"bootstrap.servers": "kafka:9092"})

def on_delivery(err, msg):
    if err is not None:
        print(f"delivery failed: {err}")

producer.produce(
    "user-events",
    key=str(user_id),                 # same key -> same partition -> preserved order per user
    value=json.dumps(event).encode(),
    callback=on_delivery,
)
producer.flush()

consumer = Consumer({
    "bootstrap.servers": "kafka:9092",
    "group.id": "feature-pipeline",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
})
consumer.subscribe(["user-events"])

while True:
    msg = consumer.poll(1.0)
    if msg is None or msg.error():
        continue
    event = json.loads(msg.value())
    update_online_features(event)     # e.g. write to a low-latency feature store
    consumer.commit(msg)              # manual commit after successful processing
```

## When to use it vs alternatives
- **vs a traditional message queue (RabbitMQ, SQS)**: queues typically delete a message once consumed and don't support replay or multiple independent consumer groups reading the same stream at their own pace as naturally; Kafka's retained, replayable log is the better fit when multiple downstream systems need the same event stream, or reprocessing history matters (e.g. rebuilding a feature store from scratch).
- **vs batch ETL (Spark/Airflow on a schedule)**: batch pipelines process a large chunk on a cadence (hourly/daily) with higher per-record latency but simpler operational model and easier debugging; Kafka-based streaming is the right call when the ML use case genuinely needs low end-to-end latency (fraud detection, real-time recommendations) — see [[batch-vs-streaming]] and [[batch-vs-realtime-inference]].
- **vs a database's own change-data-capture stream**: CDC tools (e.g. Debezium) often publish *into* Kafka rather than compete with it — Kafka is usually the transport layer underneath, not a replacement for the CDC mechanism itself.

## Interview angle
**Q. Why does message ordering only hold within a partition, not across a whole topic, and why is that a real design constraint?**
Kafka parallelizes both writes and reads at the partition level — enforcing total order across the entire topic would mean serializing all writes through one partition, destroying throughput. The practical consequence is that anything requiring ordering (e.g. all events for one entity processed in sequence) must be keyed so those events land on the same partition; get the partition key wrong and you silently lose ordering guarantees you were relying on.

**Q. A consumer group is falling behind (growing consumer lag). What are the levers?**
Increase the number of partitions (only effective up to the number of consumers you can usefully add — consumers beyond the partition count sit idle), scale out consumers within the group up to that partition count, check whether per-message processing time is the bottleneck (slow downstream writes) rather than Kafka throughput itself, and consider whether processing can be batched or offloaded asynchronously.

**Q. How would you use Kafka to build a real-time feature pipeline for online inference, and what failure mode do you have to guard against?**
Stream relevant events into Kafka, have a consumer (or Kafka Streams job) compute/update rolling features and write them to a low-latency store (e.g. a key-value feature store) that the serving path reads at inference time. The main failure mode is training/serving skew (see [[training-serving-skew]]): if the batch pipeline that computed historical training features and the streaming pipeline that computes online features implement the aggregation logic differently (even subtly — a different window boundary, a different null-handling rule), the model sees systematically different feature distributions at serving time than it was trained on.

## Traps
- Choosing a bad or absent partition key — either all traffic funnels through one partition (no parallelism) or related events lose relative ordering because they land on different partitions.
- Relying on `enable.auto.commit=True` for at-least-once semantics without understanding it can commit an offset before processing actually completes — a crash between "message read" and "processing finished" then causes silent data loss on resume, not just duplication.
- Treating Kafka as a database — it's a log with retention windows, not a queryable store; downstream systems still need their own indexed storage for point lookups.
- Ignoring consumer lag monitoring — an undetected slow consumer can fall behind the retention window and start permanently missing data, not just processing it late.

## Related
[[kafka-and-event-streaming]], [[batch-vs-streaming]], [[training-serving-skew]], [[data-pipeline-fundamentals]], [[model-monitoring]]
