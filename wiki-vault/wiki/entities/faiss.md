---
title: FAISS
type: entity
domain: rag
roles: [ai-engineer, ml-engineer]
difficulty: core
frequency: medium
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# FAISS

## What it is
Facebook AI's library for fast approximate (and exact) nearest-neighbour search over dense vectors — the computational core that most vector databases either embed directly or reimplement the ideas from. It's a library, not a service: no persistence layer, no metadata filtering server, no networking — you bring that yourself, or you use a full vector database that wraps similar indexing internally. See [[vector-databases]] for that distinction.

## Core concepts
- **Flat (exact) index**: `IndexFlatL2`/`IndexFlatIP` does brute-force distance computation against every vector — exact results, but $O(n)$ per query, fine up to maybe hundreds of thousands of vectors, too slow beyond that.
- **IVF (inverted file) index**: partitions vector space into `nlist` clusters (via k-means) at build time; a query only searches the `nprobe` nearest clusters instead of the whole dataset — approximate, with a tunable speed/recall tradeoff via `nprobe`. See [[ann-algorithms-hnsw-ivf]].
- **HNSW index**: a navigable small-world graph structure giving strong recall/speed tradeoffs without needing a training/clustering step first, at the cost of higher memory use than IVF for the same vector count.
- **Product quantization (PQ)**: compresses each vector into a short code (a concatenation of sub-vector cluster IDs) so the index fits in far less memory than storing full float vectors — trades some recall for a large memory reduction, and is what makes billion-scale indices feasible on commodity hardware. Often combined with IVF (`IVF-PQ`).
- **Training step**: IVF and PQ indices need a `.train()` call on representative vectors before `.add()` — this is what learns the cluster centroids/quantization codebooks; it's a real step people forget, distinct from adding data.
- **Distance metrics**: L2 (Euclidean) vs inner product (IP) — for cosine similarity, normalize vectors to unit length first and use IP; FAISS doesn't have a separate "cosine" index type, this normalization trick is the standard way to get it.
- **GPU support**: FAISS has first-class GPU indices (`faiss.index_cpu_to_gpu`) for very large-scale or high-QPS search — one of its practical advantages over some pure-Python vector libraries.

## Code
```python
import faiss
import numpy as np

d = 768                       # embedding dimension
embeddings = np.random.random((100_000, d)).astype("float32")
faiss.normalize_L2(embeddings)   # normalize for cosine similarity via inner product

nlist = 100
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

index.train(embeddings)       # learns the nlist cluster centroids
index.add(embeddings)
index.nprobe = 10             # search 10 of 100 clusters — speed/recall tradeoff

query = np.random.random((1, d)).astype("float32")
faiss.normalize_L2(query)
scores, ids = index.search(query, k=5)
print(list(zip(ids[0], scores[0])))
```

## When to use it vs alternatives
- **vs a managed vector database (Pinecone, Weaviate, Qdrant, Milvus)**: managed/full databases add persistence, metadata filtering, multi-tenancy, CRUD updates, and networking on top of an ANN index (often FAISS or a similar algorithm underneath) — use them when you need those production features without building them yourself. Use FAISS directly when you control the whole stack already, need maximum control over indexing parameters, or are embedding search into a larger existing system (e.g. inside a Spark job or a research pipeline).
- **vs pgvector**: pgvector keeps vectors inside Postgres alongside relational data — simpler ops story (one database) at the cost of ANN performance/tuning sophistication compared to FAISS's dedicated index types; good when vector search is a secondary feature next to an existing relational workload.
- **vs ScaNN/Annoy**: similar-purpose libraries; FAISS has the broadest adoption, GPU support, and the widest range of index types (flat, IVF, HNSW, PQ, and composites), which is why it's the default reference implementation people compare others against.

## Interview angle
**Q. Your FAISS IVF index has fast queries but recall is worse than expected. What do you tune?**
Increase `nprobe` (search more clusters per query) — the direct recall/latency lever. If that's still insufficient, increase `nlist` for a finer clustering (needs retraining), or switch to HNSW if memory allows, since it typically offers better recall at comparable speed without an `nprobe`-style approximation on top of clustering.

**Q. Why does FAISS need a separate `.train()` step, and what breaks if you skip it or train on the wrong data?**
IVF/PQ indices need to learn cluster centroids (or quantization codebooks) from representative data before vectors can be assigned to clusters/quantized — training on a small or unrepresentative sample produces centroids that don't match the true data distribution, degrading both compression quality and search recall for the *actual* dataset added later.

**Q. When would product quantization be the wrong choice even though it saves memory?**
When recall requirements are strict and the vector count is small enough that a flat or HNSW index comfortably fits in memory anyway — PQ's compression is a lossy approximation, and its recall cost only becomes worth paying once memory (not compute) is the binding constraint, typically at hundreds of millions to billions of vectors.

## Traps
- Forgetting to normalize vectors before using inner-product distance for what's meant to be cosine similarity — silently returns wrong nearest neighbours.
- Adding vectors to an IVF index before calling `.train()` — either errors or (depending on FAISS version/index) silently produces a poorly clustered index.
- Treating FAISS's approximate search as always "close enough" without measuring recall against a ground-truth exact search on a validation set — approximation quality is workload-dependent and should be verified, not assumed.
- Using FAISS as if it were a full vector database — it has no built-in persistence format beyond manual `write_index`/`read_index`, no metadata filtering, and no update-in-place semantics for a changing corpus; that all has to be built around it.

## Related
[[ann-algorithms-hnsw-ivf]], [[vector-databases]], [[embedding-models]], [[rag-overview]], [[llamaindex]]
