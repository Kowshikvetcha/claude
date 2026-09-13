---
title: LlamaIndex
type: entity
domain: rag
roles: [ai-engineer]
difficulty: core
frequency: medium
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# LlamaIndex

## What it is
A framework purpose-built for retrieval-augmented generation — ingesting, indexing, and querying your own data with an LLM. Where LangChain is general-purpose LLM-application orchestration, LlamaIndex's whole design centers on the ingest → index → retrieve → synthesize pipeline that RAG requires, with more built-in structure for that specific job.

## Core concepts
- **Documents & Nodes**: a `Document` is a raw source (a PDF, a webpage); it's split into `Node`s (chunks) which are the actual units embedded and retrieved — the chunking strategy applied here is one of the biggest levers on RAG quality. See [[chunking-strategies]].
- **Data connectors / readers**: pluggable loaders for common sources (PDFs, Notion, Slack, SQL databases, S3) that turn heterogeneous input into `Document`s — the ingestion half of [[document-ingestion-and-parsing]].
- **Index types**: `VectorStoreIndex` (the default — embed nodes, retrieve by similarity) is most common, but LlamaIndex also supports `SummaryIndex` (linear scan, good for small corpora needing exhaustive coverage), `KeywordTableIndex`, and knowledge-graph-backed indices — the index type determines the retrieval strategy, not just the storage.
- **Query engine**: wraps an index with a retrieval step and a response-synthesis step (how retrieved nodes get combined into a final LLM answer — e.g. "refine" iteratively updates an answer node-by-node, "compact" stuffs as many nodes as fit into one prompt).
- **Retrievers & postprocessors**: retrievers fetch candidate nodes (dense, sparse/BM25, or hybrid); postprocessors (rerankers, similarity-score cutoffs, recency filters) refine that candidate set before it reaches the LLM — this is where reranking (see [[reranking]]) plugs in.
- **Storage abstraction**: the vector store, document store, and index store are separate pluggable backends (FAISS, a managed vector DB, a key-value store) — swapping the underlying vector database doesn't change the query-engine code above it.
- **Agents / query-engine tools**: a query engine can itself be exposed as a tool to an agent, letting an LLM decide *when* to retrieve rather than always retrieving — the bridge between plain RAG and [[agentic-rag]].

## Code
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)   # chunks, embeds, and indexes documents

query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact",
)
response = query_engine.query("What is the refund policy for enterprise plans?")
print(response)
for node in response.source_nodes:
    print(node.score, node.node.get_content()[:100])
```

## When to use it vs alternatives
- **vs LangChain for RAG**: LlamaIndex's index/retriever/query-engine abstractions are more purpose-fit and require less glue code for a standard RAG pipeline; LangChain is more general and better when RAG is just one piece of a larger agentic/tool-using application already built in that framework. Many teams use LlamaIndex for ingestion/retrieval and LangChain (or LangGraph) for the surrounding agent logic.
- **vs a hand-rolled retrieval pipeline (embed + FAISS + manual prompt assembly)**: hand-rolling gives full control and is often faster to reason about for a simple, fixed pipeline; LlamaIndex pays off once you need multiple index types, structured query routing, or want to swap components (embedding model, vector store, reranker) without rewriting the pipeline.
- **vs a managed RAG platform**: managed offerings (some vector-DB or cloud-provider RAG services) trade flexibility for less integration work; LlamaIndex is the better choice when you need custom chunking, retrieval logic, or multi-source ingestion a managed black box doesn't expose.

## Interview angle
**Q. What's the practical difference between `response_mode="refine"` and `"compact"` in a query engine, and when does the choice matter?**
"Compact" packs as many retrieved nodes as fit into one prompt and makes a single LLM call — cheaper and faster. "Refine" processes nodes sequentially, updating a running answer with each new node — necessary when the retrieved context doesn't fit in one context window, but slower and more expensive (one LLM call per node) and prone to compounding drift if early nodes bias the running answer. Choose refine for large retrieved contexts that must be exhaustively considered; compact for the common case where top-k fits comfortably.

**Q. How would you diagnose that a LlamaIndex RAG pipeline is retrieving irrelevant chunks?**
Inspect `response.source_nodes` and their similarity scores directly — a chunking strategy that splits mid-thought, an embedding model mismatched to the domain, or a `similarity_top_k` too low/high are the usual suspects; adding a keyword/BM25 hybrid retriever or a reranker as a postprocessor is the standard fix when pure dense retrieval is missing exact-term matches.

**Q. Why might you route different query types to different index types instead of using one `VectorStoreIndex` for everything?**
A vector index is good at semantic similarity but weak at exhaustive or aggregate questions ("summarize all Q3 tickets") where a summary/list index that considers every node is more reliable; a `RouterQueryEngine` can classify the incoming query and dispatch to the appropriate index rather than forcing every query type through the same retrieval strategy.

## Traps
- Using default chunk size/overlap without tuning for the domain — the single highest-leverage RAG-quality knob, and LlamaIndex's defaults are a starting point, not a recommendation.
- Treating `similarity_top_k` as free to raise — more chunks means more tokens in the prompt (cost, latency) and can dilute the LLM's attention with irrelevant context; retrieval quality (via reranking) usually beats brute-forcing top_k up.
- Forgetting to persist the index (`index.storage_context.persist()`) — re-embedding the entire corpus on every process restart is wasteful and, for large corpora, a real cost problem.
- Assuming a single embedding model works equally well across very different document types (code vs prose vs tables) — domain-mismatched embeddings quietly degrade retrieval without an obvious error.

## Related
[[rag-overview]], [[document-ingestion-and-parsing]], [[chunking-strategies]], [[vector-databases]], [[reranking]], [[faiss]]
