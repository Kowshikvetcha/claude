---
title: Vector Database Options Compared
type: analysis
domain: rag
roles: [ai-engineer, agentic-engineer, ml-engineer, mlops-engineer]
difficulty: intermediate
frequency: high
status: drafted
tags: [vector-db, rag, retrieval]
updated: 2026-09-13
sources: []
---

# Vector Database Options Compared

## TL;DR
There is no single "best" vector database — the real interview question is almost always "do you
even need a dedicated one, or does pgvector in the Postgres you already run cover it?" A library
like FAISS gives you the ANN algorithm with none of the operational machinery; pgvector gives you
"good enough" vector search inside infrastructure you already operate and already trust for backups,
access control, and joins; a managed cloud vector database (Pinecone-style) trades money and vendor
lock-in for near-zero ops at any scale; a self-hosted specialist (Milvus/Qdrant/Weaviate) buys
purpose-built scale and hybrid search at the cost of running another stateful service; and
Elasticsearch/OpenSearch-style hybrid search is the right call when lexical search was always going
to be needed anyway and vector search is the addition, not the other way around.

## The real question being asked
Anyone can name five vector databases. The interviewer wants to know whether you reach for a new
piece of infrastructure reflexively or whether you can argue from the actual constraints — corpus
size, query volume, whether you already run Postgres, whether hybrid search matters, how much
operational surface the team can own. The single highest-signal answer in this space is being able
to say, concretely, when pgvector in an existing Postgres instance is the right call and when it
isn't — because that's the decision most teams actually face (they already have a relational
database; standing up a new vector-specific service is the exception, not the default), and it's
the one candidates most often skip past to talk about ANN algorithms instead.

## Side by side

| Dimension | FAISS (library) | pgvector (Postgres extension) | Managed cloud (Pinecone-style) | Milvus / Qdrant / Weaviate (self-hosted specialist) | Elasticsearch / OpenSearch (hybrid search platform) |
|---|---|---|---|---|---|
| Deployment model | In-process library, no server | Extension inside a Postgres instance you already run | Fully managed SaaS | Self-hosted (or managed variants) dedicated vector service | Self-hosted or managed search platform |
| ANN algorithm | HNSW, IVF, IVF+PQ — your choice, tuned directly (see [[ann-algorithms-hnsw-ivf]]) | HNSW or IVFFlat, fewer tuning knobs exposed than a specialist | HNSW-family, abstracted behind the API — you tune recall via a service-level knob, not the algorithm internals | HNSW/IVF variants, typically the most tunable of the managed-feeling options | Its own approximate kNN (HNSW-based in recent versions) layered onto an inverted-index engine |
| Hybrid (sparse+dense) search | No — bring your own fusion (e.g. combine with a separate BM25 index yourself) | Possible via Postgres full-text search + a manual fusion query, workable but hand-rolled | Increasingly native (varies by vendor), often the differentiator vendors compete on | Native in most of these (varies by product) — a core selling point | Native and mature — this is lexical search's home turf with vector search added on |
| Metadata filtering | None built in — you filter in application code | Native — it's SQL; filter with a `WHERE` clause, joined against any other table | Native, but filter expressiveness and pre- vs post-filter behavior varies by vendor | Native, generally first-class and designed in from the start | Native — filtering is a first-class citizen of the underlying search engine |
| Scale ceiling | Very high (billions of vectors) if you build the sharding/serving layer yourself | Good to tens of millions of rows on solid hardware; becomes the bottleneck earlier than a purpose-built system at very high scale | Very high — scaling is the vendor's problem, not yours | High — designed for hundreds of millions to billions of vectors | High for the search side; vector-specific scale is good but usually not the ceiling-setter for a search-first deployment |
| Operational overhead | You own everything: serving, updates, deletes, sharding — see [[vector-databases]] | Low incremental overhead if Postgres is already operated, monitored, and backed up by your team | Near zero — the entire pitch is "we run it" | Real — it's another stateful service to deploy, monitor, back up, and upgrade | Real, but often already justified/owned for search independent of the RAG use case |
| Cost model | Compute only (you provision the servers) | Marginal — same Postgres instance, same bill, some extra storage/CPU | Usage-based SaaS pricing, scales with vectors stored and queries — can get expensive at scale | Infra cost (compute/storage you provision) plus the engineering cost of operating it | Infra cost, frequently already sunk if search existed before RAG did |
| Best-fit scale/stage | Prototyping, embedded/on-device, or a team willing to build the database layer themselves | Startup/small-to-mid corpus, team already runs Postgres, wants one fewer moving part | Team wants to ship fast and not hire for vector-search ops, cost is secondary to velocity | Corpus and query volume large enough that a purpose-built engine's efficiency matters, and the team can own another service | Search (lexical) was already a requirement independent of RAG, and vector search extends it |

## When a raw library (FAISS) wins
- Prototyping, notebooks, or evaluation harnesses where you want to directly tune HNSW/IVF
  parameters and measure recall@k against an exact baseline — see [[ann-algorithms-hnsw-ivf]].
- Small or mostly-static corpora, or embedded/on-device deployments, where running a separate
  database service is unwarranted overhead relative to the corpus size (well under the point where
  metadata filtering, concurrent updates, or sharding start to matter).
- You're building the operational layer yourself anyway (a custom serving system) and just need the
  ANN index as one component, not the whole database.

## When pgvector wins
- **This is the case that actually decides most real deployments**, and it's the one worth having a
  crisp answer for: you already run Postgres for the application's relational data, the corpus is
  small-to-mid scale (roughly up to a few million vectors on reasonable hardware, sometimes more),
  and you want vector search to live in the same transactional boundary as everything else — one
  join across a `documents` table and its embeddings, one backup story, one access-control model,
  one thing to operate instead of two.
- The team's operational appetite is limited — every additional stateful service is a thing someone
  has to monitor, back up, patch, and understand at 2am; pgvector adds none of that if Postgres is
  already there.
- You need metadata filtering combined with joins against other relational data (user permissions,
  document ownership, business entities) — this is SQL's native strength, and doing it inside
  Postgres avoids re-implementing a join across two different data stores.
- **The honest counter-question an interviewer wants you to raise unprompted**: at what point does
  pgvector stop being enough? Roughly when query latency at your actual corpus size and QPS misses
  your SLA even after tuning (`ivfflat`/`hnsw` index parameters), when write throughput to the
  vector column starts contending with the rest of the application's transactional load, or when
  you need hybrid/native filtering sophistication pgvector doesn't yet provide — at that point a
  purpose-built system earns its operational cost.

## When a managed cloud vector database wins
- Team velocity matters more than cost or control — you want retrieval working this week, not after
  standing up and learning to operate a new stateful service.
- Scale or query volume is large enough, or growing unpredictably enough, that you don't want
  capacity planning to be your problem — the vendor's entire job is making that someone else's
  problem.
- You need enterprise features (multi-region, SOC2-type compliance posture, SLA-backed uptime) that
  would take real engineering time to replicate self-hosted.
- The tradeoff to name out loud: usage-based pricing at scale can get expensive relative to
  self-hosting the equivalent throughput, and you take on a vendor dependency for a component that
  sits in the critical path of every RAG query.

## When a self-hosted specialist (Milvus/Qdrant/Weaviate) wins
- The corpus and query volume are large enough that a purpose-built vector engine's efficiency
  (memory layout, native hybrid search, filtered-ANN performance) meaningfully beats pgvector, but
  the team wants to avoid managed-vendor cost or lock-in and is willing to operate the service
  itself.
- You need first-class hybrid search and rich filtering without hand-rolling the fusion logic
  pgvector would require, and you have the operational maturity (monitoring, backup, on-call) to run
  another database well.
- Data residency, compliance, or cost-at-scale considerations rule out sending vectors to a third-
  party managed service.
- The tradeoff to name out loud: this is real infrastructure ownership — upgrades, capacity
  planning, incident response — for a service that, unlike Postgres, probably isn't already
  something the team has deep operational muscle memory for.

## When Elasticsearch/OpenSearch-style hybrid search wins
- Lexical/full-text search was already a requirement for the product independent of RAG (search-as-
  a-feature, not just retrieval-for-an-LLM) — adding dense vector search to infrastructure you were
  going to run anyway is close to free relative to standing up a second, vector-only system.
- The query mix is genuinely hybrid-heavy — lots of exact-match, faceted, and keyword search needs
  alongside semantic search — and you'd rather have one mature engine do both than fuse across two
  separately-operated systems yourself.
- The tradeoff to name out loud: it's a heavier, more general system than a corpus that only ever
  needs vector search requires — you're paying the operational cost of a full search platform even
  if RAG is the only consumer.

## The honest hybrid answer
Most teams don't pick a vector database in isolation — they pick it as a consequence of what they
already operate. The most common real trajectory: start with pgvector because Postgres is already
there and the corpus is small, and *only* migrate to a specialist (managed or self-hosted) once a
measured bottleneck (latency, write contention, or a hybrid-search feature gap) actually shows up —
not because a blog post said pgvector "doesn't scale." The interview-strong move is naming the
concrete trigger for migration rather than treating the choice as fixed at project start. It's also
common, and worth naming, that many production RAG systems end up using two of these at once
deliberately: e.g. pgvector for the transactional/metadata-joined slice of retrieval and a specialist
system for a much larger, less structured corpus — rather than forcing one engine to be the answer
for every retrieval need in the org.

## Interview angle
**Q. A team wants to add RAG to a product that already runs on Postgres. Do they need a vector
database?**
Not necessarily, and that's the answer that shows judgment rather than reflex. If the corpus is
small-to-mid scale and Postgres already holds the relational context you'd want to join against
(user permissions, document metadata), pgvector gets you working semantic search inside the
transactional boundary you already operate, with one fewer service to run. I'd only reach for a
dedicated vector database once a specific, measured constraint — query latency at real scale, write
contention, or a hybrid-search capability pgvector doesn't cover — actually shows up, not
preemptively.

**Follow-up.** What's the concrete signal that tells you it's time to migrate off pgvector?
Latency or throughput that misses SLA even after tuning its ANN index parameters, write throughput
to the embedding column contending with the app's other transactional load, or a hybrid-search/
filtering requirement that would need to be hand-rolled in SQL when a specialist gives it natively —
any of these, measured, not assumed in advance.

**Q. When would you recommend a managed vector database over self-hosting one of the open-source
specialists (Milvus/Qdrant/Weaviate)?**
When engineering velocity and not having to own another stateful service matters more than the
marginal cost difference — a managed service is the right call when the team doesn't want capacity
planning, upgrades, and incident response for a vector-search service to become their job. I'd
choose self-hosting instead when cost at scale, data residency, or avoiding vendor lock-in outweigh
that operational convenience, and the team already has the muscle to run another production
database well.

**Follow-up.** How would you actually compare candidates before committing, rather than picking by
reputation? Build a recall@k evaluation harness against an exact/flat baseline on your own corpus
and query distribution (see [[ann-algorithms-hnsw-ivf]]), and benchmark each real candidate's
latency/throughput at your actual scale and filter patterns — vendor benchmarks are run on their
chosen corpus and query shape, which frequently doesn't resemble yours.

## Related
[[vector-databases]]
[[ann-algorithms-hnsw-ivf]]
[[hybrid-search-bm25-vector]]
[[embeddings]]
[[reranking]]
[[rag-overview]]
[[case-rag-assistant]]
[[faiss]]
