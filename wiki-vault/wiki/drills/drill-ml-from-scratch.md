---
title: ML From Scratch — 8 NumPy Implementations
type: drill
domain: classical-ml
roles: [data-scientist, ml-engineer, ai-engineer]
difficulty: core
frequency: high
status: drafted
tags: [drill, from-scratch, numpy]
updated: 2026-09-13
---

# ML From Scratch — 8 NumPy Implementations

"Implement X from scratch" is a staple of ML Engineer and Data Scientist loops — it tests whether
you actually understand the update rule, not whether you can call `.fit()`. Every implementation
below uses only `numpy`. Run each one on synthetic data before your interview; do not just read it.

## 1. Linear regression via gradient descent

See [[linear-regression]], [[gradient-descent-variants]], [[matrix-calculus-and-gradients]].

```python
import numpy as np

def fit_linear_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iters: int = 1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(n_iters):
        y_pred = X @ w + b
        error = y_pred - y
        grad_w = (2 / n_samples) * X.T @ error
        grad_b = (2 / n_samples) * np.sum(error)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b
```

The gradient is the derivative of mean squared error $\frac{1}{n}\sum (Xw + b - y)^2$ w.r.t. $w$
and $b$ — know this derivation cold, it is asked as a follow-up almost every time.

## 2. Logistic regression via gradient descent

See [[logistic-regression]], [[loss-functions]].

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def fit_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iters: int = 1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(n_iters):
        z = X @ w + b
        y_pred = sigmoid(z)
        error = y_pred - y                       # same form as linear regression!
        grad_w = (1 / n_samples) * X.T @ error
        grad_b = (1 / n_samples) * np.sum(error)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b
```

The gradient of binary cross-entropy w.r.t. the logits collapses to `y_pred - y` — the same
residual-times-input form as squared-error linear regression. That cancellation (softmax/sigmoid +
cross-entropy) is a favourite "why does this simplify" interview question — see
[[loss-functions]].

## 3. k-nearest neighbours (classification)

See [[k-nearest-neighbours]], [[vector-norms-and-distances]].

```python
import numpy as np
from collections import Counter

def knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int = 5):
    predictions = []
    for x in X_query:
        dists = np.linalg.norm(X_train - x, axis=1)     # Euclidean distance to every train point
        nearest_idx = np.argsort(dists)[:k]
        nearest_labels = y_train[nearest_idx]
        predictions.append(Counter(nearest_labels).most_common(1)[0][0])
    return np.array(predictions)
```

**Complexity:** $O(n \cdot d)$ per query for brute force ($n$ = train size, $d$ = dimensions) — the
reason production systems use approximate neighbour structures (KD-trees, HNSW) instead. See
[[ann-algorithms-hnsw-ivf]] for the scaled-up version of this idea.

## 4. k-means clustering

See [[clustering-kmeans]].

```python
import numpy as np

def kmeans(X: np.ndarray, k: int, n_iters: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(n_iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)   # (n, k)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, labels
```

Guard the empty-cluster case (`if np.any(labels == j) else centroids[j]`) — interviewers plant this
edge case deliberately. k-means minimises within-cluster sum of squares and is sensitive to
initialisation; mention k-means++ as the production-grade seeding strategy.

## 5. A single backprop step (2-layer MLP, manual)

See [[backpropagation]], [[neural-network-fundamentals]], [[activation-functions]].

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def forward_backward_step(X, y, W1, b1, W2, b2, lr=0.01):
    # forward
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_pred = 1 / (1 + np.exp(-z2))               # sigmoid output, binary target

    # loss: binary cross-entropy (not computed explicitly, only its gradient)
    n = X.shape[0]
    dz2 = (y_pred - y.reshape(-1, 1)) / n        # dL/dz2, using the same sigmoid+BCE simplification
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_grad(z1)                     # chain rule through ReLU
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    # gradient descent update
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1
    return W1, b1, W2, b2, y_pred
```

Walk the chain rule out loud: $\frac{\partial L}{\partial z_2} \to \frac{\partial L}{\partial W_2}
\to \frac{\partial L}{\partial a_1} \to \frac{\partial L}{\partial z_1}$ (through the ReLU
derivative, which zeroes gradients for inactive units — this is *why* dead ReLUs stop learning) $\to
\frac{\partial L}{\partial W_1}$. This is the single most commonly asked from-scratch item for
ML/AI Engineer roles. See [[vanishing-and-exploding-gradients]] for what goes wrong at depth.

## 6. Naive Bayes (Gaussian, from scratch)

See [[naive-bayes]], [[bayes-theorem-and-conditional-probability]].

```python
import numpy as np

def fit_gaussian_nb(X: np.ndarray, y: np.ndarray):
    classes = np.unique(y)
    params = {}
    for c in classes:
        X_c = X[y == c]
        params[c] = {
            "mean": X_c.mean(axis=0),
            "var": X_c.var(axis=0) + 1e-9,           # avoid divide-by-zero
            "prior": len(X_c) / len(X),
        }
    return params

def predict_gaussian_nb(X: np.ndarray, params: dict):
    classes = list(params.keys())
    log_probs = np.zeros((len(X), len(classes)))
    for i, c in enumerate(classes):
        mean, var, prior = params[c]["mean"], params[c]["var"], params[c]["prior"]
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mean) ** 2 / var, axis=1)
        log_probs[:, i] = log_likelihood + np.log(prior)
    return np.array(classes)[np.argmax(log_probs, axis=1)]
```

Work in log-space (`log_probs`) — multiplying many small probabilities underflows; summing log
probabilities does not. The conditional-independence assumption ("naive") is why this is fast and
why it breaks on correlated features — a standard trap question.

## 7. PCA via eigendecomposition

See [[dimensionality-reduction-pca]], [[eigen-decomposition-and-svd]].

```python
import numpy as np

def pca(X: np.ndarray, n_components: int):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)         # eigh: cov is symmetric, gives real, sorted-asc
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    components = eigvecs[:, :n_components]
    explained_variance_ratio = eigvals[:n_components] / eigvals.sum()
    X_reduced = X_centered @ components
    return X_reduced, explained_variance_ratio
```

Use `np.linalg.eigh` (not `eig`) because the covariance matrix is symmetric — it's faster and
numerically more stable, and a natural follow-up is "why not `eig`?" Production PCA uses SVD on the
centred data matrix directly rather than forming the covariance matrix explicitly, which avoids
squaring the condition number — mention this if asked about numerical stability at scale.

## 8. Decision tree split (Gini gain, single split)

See [[decision-trees]].

```python
import numpy as np

def gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def best_split(X: np.ndarray, y: np.ndarray):
    n_samples, n_features = X.shape
    parent_gini = gini(y)
    best_gain, best_feat, best_thresh = -1, None, None
    for feat in range(n_features):
        thresholds = np.unique(X[:, feat])
        for t in thresholds:
            left_mask = X[:, feat] <= t
            if left_mask.sum() == 0 or left_mask.sum() == n_samples:
                continue
            left_y, right_y = y[left_mask], y[~left_mask]
            weighted_gini = (
                len(left_y) / n_samples * gini(left_y)
                + len(right_y) / n_samples * gini(right_y)
            )
            gain = parent_gini - weighted_gini
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, t
    return best_feat, best_thresh, best_gain
```

**Complexity:** $O(n \cdot d \cdot n \log n)$ naively (sorting thresholds per feature per split);
production implementations (and XGBoost/LightGBM — see [[xgboost-deep-dive]]) presort once and use
histogram binning to avoid re-scanning. This single-split routine is the recursive building block
of a full tree — mention that you'd recurse on each child until a stopping criterion (max depth,
min samples, or zero gain) is hit.

## Related

[[linear-regression]] · [[logistic-regression]] · [[k-nearest-neighbours]] ·
[[clustering-kmeans]] · [[backpropagation]] · [[naive-bayes]] · [[dimensionality-reduction-pca]] ·
[[decision-trees]] · [[gradient-descent-variants]] · [[qbank-classical-ml]]
