---
title: Latency & Throughput Budgets
type: concept
domain: system-design
roles: [ml-engineer, mlops-engineer, ai-engineer, fde]
difficulty: core
frequency: high
status: drafted
tags: [latency, throughput, p99, serving, capacity-planning]
updated: 2026-09-13
sources: []
---

# Latency & Throughput Budgets

## TL;DR
Never talk about "latency" as one number — use **p50/p95/p99** (median vs. tail behaviour), because
tail latency is what breaks user experience and SLAs even when the average looks fine. A serving
pipeline's total latency budget must be allocated explicitly across its stages (feature fetch +
inference + post-processing, say), and latency and throughput trade against each other — batching
requests raises throughput but adds queuing latency per request.

## Intuition
If 99 out of 100 requests come back in 50ms but 1 comes back in 3 seconds, your *average* looks
great (~80ms) but 1% of your users are having a terrible time — and at scale, "1% of requests"
might be tens of thousands of people per day. p99 is the number that captures "how bad is the worst
common case," which is usually what actually determines whether a system feels reliable.

## The maths
**Percentile latency**: $p_{99}$ is the value below which 99% of observed latencies fall — it's
read off the empirical distribution of request latencies, not computed from mean/variance (latency
distributions are typically right-skewed, so mean and stdev are poor summaries).

**Latency budget allocation**: for a pipeline with $n$ sequential stages each taking $t_i$,
$$
t_{\text{total}} = \sum_{i=1}^n t_i \quad (\text{sequential stages, no parallelism})
$$
so if you have an SLA of $T$ ms end-to-end, you must budget $\sum t_i \le T$ across every stage —
e.g. a 100ms budget might split as 20ms feature fetch + 60ms inference + 20ms post-processing,
each with its own p99 target, not just an average.

**Throughput vs. latency (Little's Law)**: for a system in steady state,
$$
L = \lambda \cdot W
$$
where $L$ is the average number of requests in the system, $\lambda$ is arrival rate (throughput),
and $W$ is average time in system (latency). This is why increasing throughput without adding
capacity increases the number of concurrent in-flight requests, which increases queuing delay and
therefore latency — you can't independently maximize both without adding resources.

## Diagram
```mermaid
flowchart LR
    R["Request arrives"] --> F["Feature fetch (budget: 20ms p99)"]
    F --> I["Model inference (budget: 60ms p99)"]
    I --> P["Post-processing (budget: 20ms p99)"]
    P --> RS["Response (total budget: 100ms p99)"]
```

## Code
```python
import time
import numpy as np

latencies_ms = []
for _ in range(N_REQUESTS):
    t0 = time.perf_counter()
    handle_request()
    latencies_ms.append((time.perf_counter() - t0) * 1000)

p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])
print(f"p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms")
```

```python
# Budgeting each stage and asserting it in a load test / monitoring alert
STAGE_BUDGETS_MS = {"feature_fetch": 20, "inference": 60, "postprocess": 20}

def check_stage(stage_name, elapsed_ms):
    budget = STAGE_BUDGETS_MS[stage_name]
    if elapsed_ms > budget:
        log.warning(f"{stage_name} exceeded budget: {elapsed_ms:.1f}ms > {budget}ms")
```

## In practice
- **Use it when:** designing or debugging any real-time serving system — recommendation, fraud
  scoring, search, LLM inference endpoints.
- **Defaults that work:** report p50/p95/p99 (and sometimes p999 for very high-scale systems), never
  just mean latency; budget each pipeline stage individually so a regression in one stage is
  detectable in isolation rather than only visible as an aggregate SLA miss; leave headroom
  (don't budget stages to exactly sum to the SLA — network/serialization overhead and variance eat
  into the nominal budget).
- **Breaks when:** you optimize only for average latency while p99 silently degrades (e.g. a cache
  miss path, a GC pause, a cold model load) — average-only monitoring hides exactly the tail
  behaviour that determines real-world reliability. Also breaks when throughput is scaled up
  (more concurrent requests) without proportionally scaling capacity — Little's Law guarantees
  latency degrades as queuing grows.
- **Cost / latency:** batching (grouping multiple requests for one inference call) raises
  throughput (better hardware utilization, especially on GPUs) but adds latency per individual
  request (must wait for the batch to fill or a timeout) — a direct, quantifiable tradeoff dial
  (batch size / max wait time) you tune based on whether the workload favors latency (real-time,
  small batches) or throughput (batch scoring, large batches).

## Interview angle
**Q. You need to serve model predictions with a 100ms p99 SLA. Walk through how you'd allocate the
budget across the pipeline.**
Break the pipeline into stages (feature fetch, inference, post-processing, network overhead), assign
each a budget based on its expected cost profile (e.g. a feature store lookup ~15-20ms, model
inference dominates at ~50-60ms for anything beyond a trivial model, post-processing/serialization
~10-15ms), leave headroom (don't sum exactly to 100ms), and instrument each stage's own p99
separately so a regression can be attributed to the right component rather than discovered only as
an aggregate SLA breach.

**Follow-up.** Feature fetch p99 has degraded from 20ms to 80ms — what's your triage process?
→ Check whether it's a cache-miss-rate increase (cold keys, TTL expiry storm), a downstream store
degradation (e.g. the feature store's backing database under load), or a network issue — the fact
that it's specifically p99 (not p50) moving suggests a subset of requests hitting a slow path
(cache misses, hot-key contention), not a uniform slowdown.

**Q. Why is p99 latency a better operational metric than average latency for a user-facing system?**
Because user experience and SLA compliance are typically judged by the worst common case, not the
typical case — a system with great average latency but bad p99 means a meaningful fraction of
real users have a bad experience every time, and at scale that's a large absolute number of
unhappy requests even at "just 1%."

**Q. Explain the throughput/latency tradeoff in serving using batching as the example.**
Larger batches amortize fixed per-inference overhead (especially valuable on GPUs, where a bigger
batch better utilizes parallel compute) — raising overall throughput (requests/sec the system can
sustain). But an individual request now waits for the batch to fill (or a timeout to fire) before
being processed, adding latency for that request. The dial (batch size, max wait time) is a direct
throughput-for-latency trade you tune per use case: real-time serving favors small batches/low
wait; offline/batch scoring favors large batches.

**Q. How does Little's Law explain why adding more concurrent users degrades latency, even with no
code change?**
$L = \lambda W$ — if arrival rate $\lambda$ (throughput demand) increases while the system's
processing capacity is unchanged, the number of requests in the system $L$ grows, and since $W$
(latency) is what's driving that growth for a fixed effective service rate, average time in system
must rise — this is exactly what happens as a system approaches saturation (queueing theory: latency
diverges as utilization approaches 100%).

## Traps
- Reporting only mean/average latency in a design or incident discussion — signals you don't know
  what actually determines user-perceived reliability.
- "Just add more replicas to fix p99" without checking whether the tail latency is caused by a
  shared downstream bottleneck (a database, a cache) that horizontal scaling of the service layer
  won't fix — see [[scalability-patterns]].
- Assuming latency budget stages compose additively without leaving headroom — real systems have
  variance and overhead (serialization, network) that eats into the nominal per-stage numbers.
- Treating batching as a free throughput win with no cost — it always adds latency for individual
  requests; the right batch size depends entirely on the use case's latency tolerance.

## Flashcards
Why use p99 instead of average latency?::Average hides tail behavior; p99 captures the worst common case, which usually determines real user-perceived reliability at scale.
How do you allocate a latency budget across a pipeline?::Break the pipeline into stages, assign each a budget (with headroom), and monitor each stage's own percentile latency separately.
State Little's Law and what it implies for throughput/latency::L = λW; increasing arrival rate (throughput) without added capacity increases requests-in-system, which increases average latency.
How does batching trade throughput for latency?::Larger batches raise throughput (better hardware utilization) but add per-request latency (waiting for the batch to fill or a timeout).
Why might p99 latency spike while p50 stays flat?::A subset of requests is hitting a slow path (cache miss, hot-key contention, cold start) while the typical path is unaffected.

## Related
[[scalability-patterns]]
[[caching-strategies]]
[[llm-serving-and-throughput]]
[[batch-vs-realtime-inference]]
