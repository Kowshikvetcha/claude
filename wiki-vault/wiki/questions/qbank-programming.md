---
title: Programming Question Bank
type: qbank
domain: programming
roles: [ml-engineer, mlops-engineer, ai-engineer, data-scientist]
difficulty: core
frequency: high
status: drafted
tags: [question-bank]
updated: 2026-09-13
---

# Programming Question Bank

> How to use: cover the answers, write yours first, then compare. Anything you fumble → the linked concept page's `status` should go back to `drafted`.

## Warm-up

### Q1. Give the time complexity of the common Python dict/list operations you use without thinking, and the one that surprises people.
**Answer.** `dict`/`set` lookup, insert, delete: $O(1)$ average (amortized, hash-based). `list` append: $O(1)$ amortized (occasional realloc doubles capacity). `list` indexing: $O(1)$ (contiguous array). The surprise: `list.insert(0, x)` or `list.pop(0)` are $O(n)$ — every subsequent element shifts, because a Python list is a dynamic array, not a linked list. If you need cheap operations at both ends, that's exactly what `collections.deque` is for ($O(1)$ append/pop at both ends, backed by a doubly-linked block structure).
**Follow-ups.** Why is dict lookup only "average" $O(1)$ and not worst-case? → Worst case is $O(n)$ if every key hashes into the same bucket (adversarial input or a pathological hash function) — Python mitigates this with a strong default hash and open addressing, but the theoretical worst case remains linear.
**Page.** [[big-o-and-complexity]]

### Q2. What's the contract between `__eq__` and `__hash__` in Python, and what breaks if you violate it?
**Answer.** If two objects compare equal (`a == b`), they *must* have the same hash — this is required for dicts/sets to work at all, since a dict looks up by hash bucket first and only then checks equality within that bucket. If you define custom `__eq__` on a class without also defining a matching `__hash__` (or explicitly setting `__hash__ = None` to make it unhashable), Python's default `__hash__` (identity-based) will disagree with your equality — two "equal" objects can land in different buckets, so `obj in my_set` silently returns `False` even when an equal object is already in the set.
**Follow-ups.** Why does defining `__eq__` alone make a class unhashable by default in Python 3? → Because Python assumes the presence of a custom `__eq__` means the default identity-based hash is now wrong, and refuses to guess — you must opt back in explicitly.
**Page.** [[python-data-model-and-idioms]]

### Q3. Explain, mechanistically, how a Python dict resolves a hash collision.
**Answer.** CPython dicts use open addressing over a sparse table: a key's hash determines an initial slot index (`hash(key) % table_size`, roughly — actually a mask over the low bits, resized as a power of two); if that slot is occupied by a different key, a probing sequence (a pseudo-random perturbation of the hash's higher bits) generates the next candidate slot, repeating until an empty slot or a matching key is found. This differs from Java's HashMap, which chains colliding entries into a linked list/tree per bucket — CPython's approach keeps everything in one contiguous array, which is more cache-friendly.
**Follow-ups.** What does this imply about dict resizing, and why does it happen well before the table is "full"? → CPython resizes (grows and rehashes) once the table is roughly two-thirds full, because open-addressing probe sequences degrade sharply in expected length as load factor approaches 1 — resizing early keeps average probe length near constant.
**Page.** [[hashing-and-dictionaries]]

### Q4. State NumPy's broadcasting rule and give one case where broadcasting silently does the wrong thing instead of erroring.
**Answer.** Two shapes are broadcast-compatible if, comparing dimensions from the trailing end, each pair is either equal or one of them is 1 (missing leading dimensions are treated as 1). The silent-wrong-answer trap: adding a `(3,)` array to a `(3,1)` array doesn't error — it broadcasts to a `(3,3)` result via outer-product-style expansion, which is rarely what you meant if you actually wanted elementwise addition of two length-3 vectors. The fix is being deliberate about shape with `.reshape(-1)` or explicit `.squeeze()`/`[:, None]` rather than relying on broadcasting to "figure it out."
**Follow-ups.** Why does this bug disproportionately show up after a `.mean(axis=...)` call? → `.mean(axis=1, keepdims=False)` (the default) drops the reduced dimension entirely rather than setting it to 1, so subtracting the mean back from the original array silently broadcasts along the wrong axis unless you passed `keepdims=True`.
**Page.** [[numpy-essentials]]

### Q5. Why is `df.apply(lambda row: ..., axis=1)` almost always the wrong first instinct on a pandas DataFrame with 10M rows?
**Answer.** `axis=1` apply calls the Python-level lambda once per row, paying full Python interpreter overhead (function call, object boxing/unboxing) 10M times — it's not vectorized, so none of pandas' underlying NumPy C-loop speed is used. A vectorized rewrite (arithmetic directly on `Series`/columns, `np.where`, `.str` accessor methods, or `pd.cut`) pushes the loop down into compiled C, typically 10-100x faster for the same logic. `apply` is reasonable only when the per-row logic is genuinely irreducible to vector ops (e.g. calling an external API per row) — and even then, you'd batch it rather than truly go row by row.
**Follow-ups.** What's the escape hatch when the logic really can't be vectorized and 10M Python-level calls is unacceptable? → Push it to PySpark/Spark UDFs for distributed execution, or use `numba`/Cython to JIT-compile the per-row function itself so the loop body is no longer pure interpreted Python.
**Page.** [[pandas-essentials]]

## Core

### Q6. Solve "find the maximum sum of a contiguous subarray" and name the pattern.
**Answer.** This is Kadane's algorithm, a specific instance of the sliding-window / running-accumulator pattern: track `current_sum = max(num, current_sum + num)` while scanning left to right, and `best = max(best, current_sum)` at each step. The key insight is that a negative running sum can never help a future subarray, so you reset to just the current element whenever the running sum would go negative — this makes it $O(n)$ time, $O(1)$ space instead of the naive $O(n^2)$ check-every-subarray approach.
```python
def max_subarray(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
```
**Follow-ups.** How would you extend this to also return the subarray's start/end indices, not just the sum? → Track a candidate `start` index that resets whenever `cur` resets to `x` alone, and update a `best_start`/`best_end` pair whenever `best` is improved.
**Page.** [[arrays-and-strings-patterns]]

### Q7. When do you reach for BFS vs DFS, and why does BFS — not DFS — give you shortest path in an unweighted graph?
**Answer.** BFS explores the graph in strict distance layers (all nodes at distance $k$ before any at distance $k+1$), using a FIFO queue — the first time you reach a node, you've reached it by the shortest possible number of edges, because every shorter path would have been explored in an earlier layer. DFS explores one path as deep as possible before backtracking (a LIFO stack or recursion), which is right for exhaustive exploration (topological sort, cycle detection, connected components) but gives no shortest-path guarantee — it might stumble onto a node via a long path before a shorter one is ever tried.
**Follow-ups.** What has to change for BFS to still give shortest path on a *weighted* graph? → Plain BFS breaks because edge weights aren't uniform "one hop"; you need Dijkstra's algorithm (a priority queue ordered by accumulated distance instead of a plain FIFO queue) — or Bellman-Ford if negative weights are possible.
**Page.** [[trees-and-graphs-basics]]

### Q8. You're given a fresh DP problem in an interview. What's your systematic process for recognizing and solving it, rather than pattern-matching from memory?
**Answer.** First check for the DP signature: optimal substructure (the answer for the whole problem can be built from answers to smaller subproblems) plus overlapping subproblems (naive recursion recomputes the same subproblem many times — if you drew the recursion tree, would branches repeat?). Then define the state precisely in words before writing any code ("`dp[i]` = the longest valid subsequence ending at index `i`"), write the brute-force recursive recurrence first, add memoization to fix the exponential blowup, and only convert to bottom-up tabulation once the recurrence is verified correct — premature tabulation is where most interview candidates introduce off-by-one bugs.
**Follow-ups.** How do you decide whether a problem needs 1D or 2D state? → Count the independent quantities the recurrence needs to distinguish between subproblems — one sequence index alone (1D) vs. two sequence positions or an index plus a running constraint like "remaining capacity" (2D, as in knapsack or edit distance).
**Page.** [[dynamic-programming-patterns]]

### Q9. A teammate's PySpark pipeline runs a Python UDF row-by-row and it's 50x slower than the equivalent native Spark SQL expression. Explain why, precisely.
**Answer.** A native Spark SQL/DataFrame expression runs entirely inside the JVM using Catalyst-generated, whole-stage-codegen'd bytecode — no serialization boundary crossed. A Python UDF forces Spark to serialize each row's data out of the JVM, pipe it to a separate Python worker process, execute the Python function (paying full CPython interpreter overhead per row plus the GIL's single-threaded execution within that worker), then serialize the result back across the process boundary into the JVM. That round-trip serialization plus losing all of Catalyst's optimization on the UDF's internals is the entire 50x — it's not really "Python is slow," it's "cross-process serialization for every row is slow."
**Follow-ups.** How do pandas UDFs (vectorized UDFs) close most of that gap? → They batch many rows into an Arrow-backed pandas Series per call instead of one row per call, amortizing the serialization overhead across a batch and letting the UDF body use vectorized pandas/NumPy operations instead of a Python-level per-row loop.
**Page.** [[python-performance-and-memory]]

### Q10. Design a small class hierarchy for swappable model backends (e.g. XGBoost vs. a scikit-learn model) behind one training pipeline. What pattern is this and why does it matter for testability?
**Answer.** This is the Strategy pattern: define an abstract interface (`fit`, `predict`, `save`, `load`) that every concrete model wrapper implements, and have the pipeline code depend only on that interface, never on a concrete class. The pipeline can then swap `XGBoostModel` for `SklearnModel` or a `FakeModel` used purely in tests without touching pipeline code — this is what makes the pipeline unit-testable in isolation (inject a fake, deterministic model) rather than every test needing to actually train a real model.
```python
from abc import ABC, abstractmethod

class ModelBackend(ABC):
    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def predict(self, X): ...

class XGBoostModel(ModelBackend):
    def __init__(self, **params):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(**params)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
```
**Follow-ups.** Why is this also what makes MLflow's model-flavor abstraction (`mlflow.pyfunc`) work across frameworks? → `pyfunc` is the same idea at the platform level — any model that implements a `predict(model_input)` contract can be logged, served and swapped interchangeably regardless of its native framework, because downstream tooling only ever depends on that one interface.
**Page.** [[oop-and-design-patterns-for-ml]]

### Q11. How do you unit test a function that reads from a Delta table and writes predictions to another table, without spinning up a real Spark cluster or touching real data?
**Answer.** Separate the function into a pure transformation (takes a DataFrame in, returns a DataFrame out — no I/O) and thin I/O wrappers around it; test the pure part directly against small, hand-constructed DataFrames (a local `SparkSession` in local mode, or plain pandas if the logic doesn't need Spark-specific behavior) and assert on the output rows/schema. For the I/O wrappers, use dependency injection (pass in a table-path/reader function as a parameter) and mock it in tests with `unittest.mock.patch` or a fake reader that returns a fixture DataFrame — the goal is that the test asserts business logic correctness, not "does Spark work" or "is the table there."
**Follow-ups.** What's the tradeoff of using a real local `SparkSession` in tests versus mocking Spark entirely? → A local SparkSession catches real Spark-API misuse (wrong column types, an actually-invalid join) that a full mock would hide, at the cost of slower test startup — most teams keep a small number of local-Spark integration tests and mock aggressively everywhere else.
**Page.** [[testing-python-code]]

### Q12. You're 20 minutes into a live coding round and your first approach to a problem turns out to be wrong. What do you actually do?
**Answer.** Say out loud, explicitly, that the current approach has a flaw and why — silently deleting code and restarting reads as "gave up," while narrating "this breaks when X, because Y" reads as "caught my own bug," which is the actual signal interviewers are grading. Don't discard working sub-pieces; keep whatever logic was correct (e.g. the input parsing) and only rework the broken part. If time is short, state the brute-force fallback explicitly ("I can finish with an $O(n^2)$ version now and optimize after, or keep pushing on the $O(n\log n)$ approach — which would you prefer?") rather than silently running out of time on the optimal approach with nothing working.
**Follow-ups.** Why do interviewers often care more about this moment than about whether you found the optimal solution unaided? → It's the closest proxy they have to how you'll behave when a production approach turns out wrong at 4pm before a demo — recovery process, not first-attempt perfection, is what correlates with actual job performance.
**Page.** [[coding-interview-strategy]]

## Hard

### Q13. Derive an $O(n \log n)$ solution for Longest Increasing Subsequence and explain why the standard $O(n^2)$ DP can't be trivially sped up the same way.
**Answer.** Maintain an array `tails` where `tails[k]` is the smallest possible tail value of an increasing subsequence of length `k+1` seen so far. For each new number, binary-search `tails` for the first entry $\ge$ the number and overwrite it (extending the subsequence family if the number is bigger than everything so far, or "improving" an existing length-class otherwise); the final length of `tails` is the LIS length. This works because `tails` is provably non-decreasing at every step, which is exactly what makes it binary-searchable — the $O(n^2)$ DP (`dp[i]` = LIS ending at `i`, scanning all `j<i`) doesn't expose this monotonic structure directly since it tracks "best ending here," not "smallest tail for this length," so there's nothing to binary search over in that formulation.
**Follow-ups.** Does `tails` at the end actually contain a valid LIS? → No — it's not a real subsequence, just tracks optimal tail values per length; reconstructing the actual LIS requires an auxiliary predecessor array tracked alongside the binary search.
**Page.** [[dynamic-programming-patterns]]

### Q14. Given a dict of task → list of dependencies, detect whether the dependencies are satisfiable at all, and if so return one valid execution order.
**Answer.** This is topological sort on a dependency DAG — if a cycle exists, no valid order exists. Kahn's algorithm: compute in-degree (number of unresolved dependencies) for every node, seed a queue with all in-degree-0 nodes, repeatedly pop a node, append it to the order, and decrement in-degree for everything that depended on it, pushing any that hit zero. If the final order has fewer nodes than the graph, a cycle exists among the leftover nodes (they never reached in-degree 0).
```python
from collections import deque

def topo_order(deps: dict[str, list[str]]) -> list[str] | None:
    indeg = {n: 0 for n in deps}
    for n, ds in deps.items():
        for d in ds:
            indeg[n] += 1
    ready = deque([n for n, d in indeg.items() if d == 0])
    order = []
    while ready:
        n = ready.popleft()
        order.append(n)
        for m, ds in deps.items():
            if n in ds:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
    return order if len(order) == len(deps) else None  # None => cycle
```
**Follow-ups.** How is this the same underlying problem as detecting a circular import in a Python package? → Modules are nodes, `import` statements are dependency edges — a circular import is exactly a cycle in this DAG, which is why Python raises `ImportError` at the point it detects it's revisiting a module still mid-initialization rather than fully resolved.
**Page.** [[trees-and-graphs-basics]]

### Q15. Vectorize a nested-loop pairwise Euclidean distance computation between two sets of points using only NumPy, and explain the trick.
**Answer.** The naive version is $O(n \cdot m)$ Python-level loop iterations. The vectorized trick expands $\|a-b\|^2 = \|a\|^2 - 2a\cdot b + \|b\|^2$ and computes each term as a matrix operation: squared norms per point (cheap, $O(n)$ and $O(m)$), and the cross term as a single matrix multiply $A B^T$ (BLAS-accelerated, highly optimized). Broadcasting then combines an $(n,1)$ column of squared norms with a $(1,m)$ row and the $(n,m)$ cross-term matrix into the full pairwise distance matrix with zero Python-level looping.
```python
import numpy as np

def pairwise_dist(A, B):
    a2 = np.sum(A**2, axis=1)[:, None]      # (n, 1)
    b2 = np.sum(B**2, axis=1)[None, :]      # (1, m)
    cross = A @ B.T                          # (n, m), BLAS matmul
    return np.sqrt(np.maximum(a2 - 2*cross + b2, 0))  # clip tiny negative fp noise
```
**Follow-ups.** Why is the `np.maximum(..., 0)` clip actually necessary and not just defensive paranoia? → Floating-point cancellation in `a2 - 2*cross + b2` can produce a tiny negative number for points that are (near-)identical, where the true squared distance is exactly 0 — `sqrt` of a negative float is `nan`, silently poisoning downstream aggregates.
**Page.** [[numpy-essentials]]

### Q16. Explain the difference between an iterator and a generator in Python's data model, and why a generator is the right tool for streaming a large file through a processing pipeline.
**Answer.** An iterator is any object implementing `__iter__` (returns itself) and `__next__` (returns the next value or raises `StopIteration`) — the general protocol. A generator is a specific, convenient way to *produce* an iterator: any function with a `yield` statement, when called, returns a generator object that implements that protocol automatically, pausing and resuming execution state at each `yield` rather than requiring you to hand-write a class with manual state tracking. For a large file, a generator (`for line in open(path): yield process(line)`) holds only one line/record in memory at a time — memory usage is $O(1)$ in the input size — versus reading the whole file into a list first, which is $O(n)$ memory and can OOM on a file bigger than RAM.
**Follow-ups.** What breaks if downstream code calls `len()` or indexes into a generator like a list? → Both fail — a generator has no `__len__` and no `__getitem__` with random access, because it doesn't hold all its values at once; if you need those, you must materialize it (`list(gen)`), which reintroduces the full memory cost you were trying to avoid.
**Page.** [[python-data-model-and-idioms]]

## Related
See [[moc-programming]].
