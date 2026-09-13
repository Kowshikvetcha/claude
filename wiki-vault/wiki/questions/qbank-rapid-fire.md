---
title: Rapid-Fire Question Bank
type: qbank
domain: system-design
roles: [data-scientist, ml-engineer, ai-engineer, mlops-engineer, agentic-engineer, fde]
difficulty: core
frequency: high
status: drafted
tags: [question-bank, rapid-fire, all-domains]
updated: 2026-09-13
---

# Rapid-Fire Question Bank

> How to use: a pre-interview lightning warm-up, not a study-in-depth bank. Read straight down, answer out loud in one breath, check yourself against the one-liner. If you hesitate on one, that's your signal to go re-read the full concept page or the domain's dedicated `qbank-*`, not to linger here.

## Maths
- **Q:** Why is L1 regularization associated with sparsity but L2 isn't? **A:** L1's subgradient at 0 can zero out a weight and keep it there; L2's gradient shrinks proportionally to the weight and never forces an exact zero.
- **Q:** What does it mean for a matrix to be positive semi-definite, and why does it matter for a Hessian? **A:** $x^TAx \ge 0$ for all $x$; a PSD Hessian means the loss is locally convex, so gradient descent can't get stuck oscillating around a saddle.
- **Q:** Why divide attention scores by $\sqrt{d_k}$? **A:** Dot products of $d_k$-dimensional random vectors have variance $d_k$; scaling by $\sqrt{d_k}$ keeps softmax inputs from saturating.
- **Q:** State Bayes' theorem in one line. **A:** $P(A|B) = P(B|A)P(A)/P(B)$.
- **Q:** Why does cross-entropy pair naturally with KL divergence as a training loss? **A:** Cross-entropy equals the true distribution's entropy plus $KL(P\|Q)$; since entropy is fixed, minimizing cross-entropy is exactly minimizing KL toward the data.

See [[moc-maths]].

## Statistics
- **Q:** What does a 95% confidence interval actually mean? **A:** 95% of intervals built this way over repeated sampling would contain the true parameter — not "95% probability the parameter is in this one."
- **Q:** Type I vs Type II error, one line each? **A:** Type I is a false positive (rejecting a true null); Type II is a false negative (failing to reject a false null).
- **Q:** Why does running 20 tests at $p<0.05$ give about one false positive by chance alone? **A:** Each independent test has a 5% false-positive rate under the null, so expected false positives $\approx 0.05 \times 20 = 1$.
- **Q:** What is statistical power? **A:** The probability of correctly rejecting a false null ($1-\beta$); it rises with larger sample size, larger effect size and higher $\alpha$.
- **Q:** Bootstrap vs permutation testing — when do you use which? **A:** Bootstrap estimates a statistic's sampling distribution/CI by resampling one sample; permutation testing builds a null distribution by shuffling labels to test a hypothesis.

See [[moc-stats]].

## Programming
- **Q:** Average-case time complexity of a hash-map lookup? **A:** $O(1)$ average, $O(n)$ worst case under heavy collisions.
- **Q:** Why prefer vectorized NumPy over a Python for-loop? **A:** NumPy pushes the loop into compiled C and avoids per-element Python object overhead, giving 10-100x speedups.
- **Q:** List vs generator in Python — what's the tradeoff? **A:** A list materializes everything in memory; a generator lazily yields one item at a time, trading memory for one-pass-only iteration.
- **Q:** When do you reach for a set instead of a list for membership checks? **A:** When you need $O(1)$ average lookup instead of a linear scan, at the cost of losing order and duplicates.
- **Q:** What is the GIL and why does it matter for a CPU-bound ML pipeline? **A:** The Global Interpreter Lock lets only one thread execute Python bytecode at a time, so CPU-bound work needs multiprocessing (or a C-extension), not threading, to actually parallelize.

See [[moc-programming]].

## SQL
- **Q:** What is the logical execution order of a SQL query? **A:** FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT.
- **Q:** LEFT JOIN vs INNER JOIN? **A:** INNER JOIN keeps only matching rows in both tables; LEFT JOIN keeps every left-table row, filling unmatched right-side columns with NULL.
- **Q:** WHERE vs HAVING? **A:** WHERE filters rows before aggregation; HAVING filters groups after aggregation.
- **Q:** What can a window function do that GROUP BY can't? **A:** Compute an aggregate or rank per row while still returning every row — GROUP BY collapses rows into one per group.
- **Q:** What's a classic cause of a slow query that an index fixes? **A:** A full table scan on a high-selectivity filter or join column — an index turns that scan into a seek.

See [[moc-sql]].

## Classical ML
- **Q:** Bagging vs boosting, one line each? **A:** Bagging trains independent models in parallel on bootstrapped samples to reduce variance; boosting trains models sequentially, each correcting the last one's errors, to reduce bias.
- **Q:** Why does XGBoost dominate tabular data over deep nets? **A:** Gradient-boosted trees handle heterogeneous, non-smooth tabular features far more sample-efficiently, with far less tuning, than a neural net.
- **Q:** Precision vs recall — when do you optimize for which? **A:** Precision matters when false positives are costly (flagging good transactions as fraud); recall matters when false negatives are costly (missing actual fraud or disease).
- **Q:** What is target leakage? **A:** A feature that encodes information only available after the label is known — e.g. "days until cancellation" as a churn-prediction feature.
- **Q:** Why does k-means need you to choose $k$ up front, and how do you pick it? **A:** It optimizes within-cluster distance for a fixed number of centroids; pick $k$ with the elbow method, silhouette score, or domain knowledge of expected segments.

See [[moc-classical-ml]].

## Deep Learning
- **Q:** Why did ReLU replace sigmoid/tanh as the default activation? **A:** It doesn't saturate for positive inputs, so gradients don't vanish through many layers, and it's cheap to compute.
- **Q:** What causes vanishing gradients in a very deep or recurrent network? **A:** Repeated multiplication of gradients through many layers or timesteps, each with magnitude less than 1, shrinks the signal exponentially toward zero.
- **Q:** Why do transformers use LayerNorm instead of BatchNorm? **A:** LayerNorm normalizes per-example across features, so it doesn't depend on batch statistics — critical for variable-length sequences and small or single-example inference batches.
- **Q:** What problem does Adam solve versus vanilla SGD? **A:** It rescales each parameter's step by a running estimate of its gradient's mean and variance, giving adaptive per-coordinate steps on ill-conditioned loss surfaces.
- **Q:** Why can transformers model long-range dependencies better than RNNs? **A:** Every token attends directly to every other token in one step, instead of information decaying as it passes sequentially through a hidden state.

See [[moc-deep-learning]].

## NLP & LLMs
- **Q:** Causal vs masked language modeling? **A:** Causal LM predicts the next token from only prior tokens (GPT-style, good for generation); masked LM predicts randomly masked tokens using both left and right context (BERT-style, good for representations).
- **Q:** Why does quantizing a model to int8/int4 speed up inference? **A:** Lower-precision weights shrink the memory footprint, and memory bandwidth — not compute — is usually the inference bottleneck.
- **Q:** What does LoRA actually do? **A:** Freezes the pretrained weights and learns a low-rank update added to them, cutting trainable parameters and memory by orders of magnitude versus full fine-tuning.
- **Q:** Why does KV-caching speed up autoregressive decoding? **A:** It stores previously computed key/value projections so each new token only attends against cached history instead of recomputing the whole sequence.
- **Q:** RLHF vs DPO in one line? **A:** RLHF trains a separate reward model and optimizes the policy against it with RL; DPO skips the reward model and directly optimizes the policy on preference pairs with a closed-form loss.

See [[moc-nlp-llm]].

## RAG
- **Q:** What's the single most common reason a RAG system gives a wrong answer? **A:** Retrieval failure — the right chunk was never retrieved, so the generator is grounded on irrelevant context.
- **Q:** Why add a reranker after initial vector retrieval? **A:** Initial ANN search optimizes cheaply for recall over a large candidate set; a reranker then applies a more accurate, expensive model to reorder just the top-k for precision.
- **Q:** When do you reach for hybrid (BM25 + vector) search over pure vector search? **A:** When queries contain exact keywords, IDs, or rare terms that embeddings blur together — lexical search catches what semantic similarity misses.
- **Q:** Why does chunk size matter so much in a RAG pipeline? **A:** Too small loses the context needed to answer; too large dilutes the embedding's relevance signal and wastes context-window budget.
- **Q:** What does GraphRAG add over standard vector RAG? **A:** It indexes entities and relationships as a graph, enabling multi-hop or relationship-heavy questions pure similarity search over chunks can't connect.

See [[moc-rag]].

## Agents
- **Q:** What's the ReAct pattern? **A:** Interleaving reasoning ("thought") steps with tool-invoking "actions" and their "observations," so the model plans and reacts to real tool output instead of reasoning blind.
- **Q:** What distinguishes an "agentic" system from a chained prompt pipeline? **A:** The model itself decides the control flow at runtime (which tool, how many steps, when to stop) rather than a developer hard-coding it in advance.
- **Q:** Why is agent evaluation considered the hardest open problem in the space? **A:** Success is often a multi-step trajectory, not one output, so you must judge the process (right tool, right order), and labeled ground truth for that is expensive.
- **Q:** Name one concrete failure mode of a multi-agent system. **A:** Agents can loop deferring to each other, or one agent's hallucinated intermediate output silently corrupts every downstream agent's input.
- **Q:** What is the Model Context Protocol (MCP) trying to standardize? **A:** A common interface for exposing tools and data sources to LLM agents so a client doesn't need a bespoke integration per tool.

See [[moc-agents]].

## MLOps
- **Q:** Why is model monitoring the most-asked MLOps topic in interviews? **A:** A model degrading silently in production via drift is the single most common real-world ML failure, and it's invisible without dedicated monitoring.
- **Q:** Data drift vs concept drift? **A:** Data drift is a shift in the input feature distribution; concept drift is a shift in the relationship between features and the label.
- **Q:** What problem do feature stores solve? **A:** Training-serving skew — they guarantee the same feature computation logic and values are used offline for training and online for serving.
- **Q:** Shadow deployment vs canary deployment? **A:** Shadow runs the new model on live traffic without serving its predictions, purely for comparison; canary actually serves the new model's predictions to a small slice of real traffic.
- **Q:** Why version data, not just code and models? **A:** To reproduce a training run exactly and debug "which data produced this model" — model behavior is a function of code, data and hyperparameters together.

See [[moc-mlops]].

## Data Engineering
- **Q:** What does bronze/silver/gold mean in a medallion architecture? **A:** Bronze is raw ingested data as-is, silver is cleaned and conformed data, gold is business-level aggregated data ready for consumption or ML features.
- **Q:** Why does a lakehouse (e.g. Delta Lake) beat a plain data lake? **A:** It adds ACID transactions, schema enforcement and time travel on top of cheap object storage, fixing the lake's reliability problems without warehouse cost.
- **Q:** What's the most common root cause of a slow Spark job? **A:** Data skew or bad partitioning, causing a few tasks or executors to do far more work and shuffle far more data than the rest.
- **Q:** Batch vs streaming — what's the deciding factor? **A:** How fresh the result needs to be, weighed against how expensive and complex it is to keep a continuously running pipeline correct and fault-tolerant.
- **Q:** Why do columnar formats like Parquet dominate lakehouse storage? **A:** Analytical queries typically read few columns across many rows — columnar layout skips unread columns and compresses similar values far better than row-oriented formats.

See [[moc-data-engineering]].

## System Design
- **Q:** What's the first thing to nail down before proposing an ML system's architecture? **A:** The requirements and success metrics — latency/throughput budget, evaluation metric, and cost constraints — before any component design.
- **Q:** Why does training-serving skew happen? **A:** The feature computation logic, library versions, or data differ between the offline training pipeline and the online serving path.
- **Q:** CAP theorem in one line? **A:** Under a network partition, a distributed system must choose between consistency (every read sees the latest write) and availability (every request gets a response).
- **Q:** Why is caching such a universal win in ML-serving system design? **A:** Repeated or predictable requests can skip the expensive model call entirely, cutting both latency and compute cost.
- **Q:** What changes when the system-design question is about an LLM instead of a classical model? **A:** Cost-per-call, context-window budget and latency-vs-quality tradeoffs replace classical throughput/replication concerns as the primary constraints.

See [[moc-system-design]].

## Behavioral
- **Q:** What's the STAR structure? **A:** Situation, Task, Action, Result — ground the story, state your specific responsibility, describe what you actually did, then quantify the outcome.
- **Q:** How should you frame a career gap in an Indian interview? **A:** State it factually and briefly, then pivot immediately to what you did with the time — don't over-apologize or over-explain.
- **Q:** What's the biggest mistake candidates make on "tell me about a time you failed"? **A:** Picking a fake-humble non-failure ("I worked too hard") instead of a real mistake with a genuine, evidenced lesson.
- **Q:** Why does asking good questions at the end of an interview matter? **A:** It's the panel's last data point on your seniority and genuine interest — generic questions read as unprepared.
- **Q:** What's the core principle in negotiating an Indian offer with a competing offer in hand? **A:** Get every competing offer fully documented before negotiating, and negotiate the total fixed-plus-variable-plus-ESOP structure, not just the headline number.

See [[moc-behavioral]].

## Related
See [[moc-system-design]] and [[moc-behavioral]] for the two domains this rapid-fire bank straddles most directly; every section above links back to its own domain's map.
