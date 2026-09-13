---
title: Python Coding Drill — 12 Problems
type: drill
domain: programming
roles: [data-scientist, ml-engineer, ai-engineer, mlops-engineer, agentic-engineer, fde]
difficulty: core
frequency: high
status: drafted
tags: [drill, coding]
updated: 2026-09-13
---

# Python Coding Drill — 12 Problems

This is the **5-years-experience bar**, not competitive programming: recognise the pattern fast,
write clean idiomatic Python, state complexity, mention the edge cases out loud. See
[[coding-interview-strategy]] for how these get evaluated, [[big-o-and-complexity]] for complexity
talk, [[arrays-and-strings-patterns]], [[hashing-and-dictionaries]], [[trees-and-graphs-basics]],
[[dynamic-programming-patterns]] for the underlying patterns.

## 1. Two Sum

**Pattern:** hashing / one-pass lookup.

```python
def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    seen = {}  # value -> index
    for i, n in enumerate(nums):
        complement = target - n
        if complement in seen:
            return seen[complement], i
        seen[n] = i
    return None
```

**Complexity:** $O(n)$ time, $O(n)$ space. The trap is reaching for nested loops ($O(n^2)$) —
a hash map trades space for a single pass.

## 2. Longest substring without repeating characters

**Pattern:** sliding window + hash set.

```python
def longest_unique_substring(s: str) -> int:
    seen: dict[str, int] = {}
    start = 0
    best = 0
    for end, ch in enumerate(s):
        if ch in seen and seen[ch] >= start:
            start = seen[ch] + 1
        seen[ch] = end
        best = max(best, end - start + 1)
    return best
```

**Complexity:** $O(n)$ time (each index visited once as `end`, `start` only moves forward),
$O(\min(n, |\Sigma|))$ space.

## 3. Group anagrams

**Pattern:** hashing with a canonical key.

```python
from collections import defaultdict

def group_anagrams(words: list[str]) -> list[list[str]]:
    groups: dict[tuple, list[str]] = defaultdict(list)
    for w in words:
        key = tuple(sorted(w))
        groups[key].append(w)
    return list(groups.values())
```

**Complexity:** $O(n \cdot k \log k)$ where $k$ is max word length (sorting each word); can drop to
$O(n \cdot k)$ using a 26-count tuple as the key instead of sorting.

## 4. Merge intervals

**Pattern:** sort + linear scan.

```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```

**Complexity:** $O(n \log n)$ (dominated by the sort), $O(n)$ space for output.

## 5. Top-K frequent elements

**Pattern:** hashing + heap (or bucket sort for the $O(n)$ variant).

```python
import heapq
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    counts = Counter(nums)
    return [item for item, _ in heapq.nlargest(k, counts.items(), key=lambda x: x[1])]
```

**Complexity:** $O(n \log k)$ with a heap of size $k$; a bucket-sort-by-frequency approach gets this
to $O(n)$ when $k$ is small relative to $n$ — mention both, implement the heap one.

## 6. Validate binary search tree

**Pattern:** tree traversal with bounds.

```python
from dataclasses import dataclass

@dataclass
class Node:
    val: int
    left: "Node | None" = None
    right: "Node | None" = None

def is_valid_bst(root: Node | None) -> bool:
    def helper(node, low, high):
        if node is None:
            return True
        if not (low < node.val < high):
            return False
        return helper(node.left, low, node.val) and helper(node.right, node.val, high)
    return helper(root, float("-inf"), float("inf"))
```

**Complexity:** $O(n)$ time, $O(h)$ space for recursion stack ($h$ = tree height). The classic trap:
checking only `node.left.val < node.val < node.right.val` locally, which misses violations from a
grandparent two levels up — bounds must propagate down the whole path.

## 7. Level-order traversal (BFS)

**Pattern:** BFS with a queue.

```python
from collections import deque

def level_order(root: Node | None) -> list[list[int]]:
    if root is None:
        return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

**Complexity:** $O(n)$ time and space. Use `deque`, never `list.pop(0)` — that's $O(n)$ per pop and
turns the whole traversal quadratic.

## 8. Number of islands (grid BFS/DFS)

**Pattern:** graph traversal on an implicit grid graph.

```python
def num_islands(grid: list[list[str]]) -> int:
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if visited[r][c] or grid[r][c] == "0":
            return
        visited[r][c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dfs(r + dr, c + dc)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and not visited[r][c]:
                count += 1
                dfs(r, c)
    return count
```

**Complexity:** $O(rows \times cols)$ time and space. For large grids, prefer an explicit stack over
recursion to avoid Python's recursion-depth limit.

## 9. Coin change (min coins)

**Pattern:** 1-D bottom-up DP.

```python
def coin_change(coins: list[int], amount: int) -> int:
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != INF else -1
```

**Complexity:** $O(\text{amount} \times \text{len(coins)})$ time, $O(\text{amount})$ space. State it
as "min coins to make amount $a$ = 1 + min over coins $c$ of (min coins for $a-c$)" — the recurrence
is the answer to "how did you think of this," not the code.

## 10. Longest common subsequence

**Pattern:** 2-D DP on strings.

```python
def lcs(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

**Complexity:** $O(mn)$ time and space; space reducible to $O(\min(m,n))$ by keeping only two rows —
a good follow-up to volunteer. See [[dynamic-programming-patterns]] for the general "two-string DP"
template this generalises (edit distance, LCS, longest common substring all share this grid shape).

## 11. LRU cache

**Pattern:** hash map + doubly linked list (or `OrderedDict` in interview-realistic Python).

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

**Complexity:** $O(1)$ for both `get` and `put` — `OrderedDict` gives you the linked-list ordering
for free. Know the from-scratch version (dict + manual doubly linked list) as a follow-up; using
`OrderedDict` outright is legitimate at this level unless the interviewer explicitly bans built-ins.

## 12. K closest points to origin

**Pattern:** heap-based partial sort (top-K without fully sorting).

```python
import heapq

def k_closest(points: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    return heapq.nsmallest(k, points, key=lambda p: p[0] ** 2 + p[1] ** 2)
```

**Complexity:** $O(n \log k)$ using a max-heap of size $k$ (which `heapq.nsmallest` does
internally); avoid computing `sqrt` — squared distance preserves ordering and avoids float ops.
Full sort is $O(n \log n)$ and acceptable for small $n$, but naming the heap approach shows you
know when $k \ll n$ matters.

## Related

[[coding-interview-strategy]] · [[arrays-and-strings-patterns]] · [[hashing-and-dictionaries]] ·
[[trees-and-graphs-basics]] · [[dynamic-programming-patterns]] · [[big-o-and-complexity]] ·
[[python-data-model-and-idioms]] · [[qbank-programming]]
