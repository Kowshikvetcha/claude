---
title: Maths Question Bank
type: qbank
domain: maths
roles: [data-scientist, ml-engineer, ai-engineer]
difficulty: core
frequency: high
status: drafted
tags: [question-bank]
updated: 2026-09-13
---

# Maths Question Bank

> How to use: cover the answers, write yours first, then compare. Anything you fumble → the linked concept page's `status` should go back to `drafted`.

## Warm-up

### Q1. What does it mean for a matrix to be positive semi-definite, and why do you care about it in ML?
**Answer.** A symmetric matrix $A$ is PSD if $x^T A x \ge 0$ for all $x$. Covariance matrices and Hessians at a convex minimum are PSD. It matters because a PSD Hessian guarantees the loss surface is locally convex (no saddle in that neighbourhood), and a PSD kernel matrix is what makes an SVM kernel valid (Mercer's condition).
**Follow-ups.** How do you check PSD-ness numerically? → All eigenvalues ≥ 0, or all leading principal minors ≥ 0 (Sylvester's criterion), or Cholesky succeeds.
**Page.** [[eigen-decomposition-and-svd]]

### Q2. What's the difference between eigendecomposition and SVD, and why does every matrix have an SVD but not every matrix has an eigendecomposition?
**Answer.** Eigendecomposition $A = V\Lambda V^{-1}$ requires $A$ square and diagonalizable. SVD $A = U\Sigma V^T$ exists for any real matrix (even non-square) because it decomposes the linear map into rotation-scale-rotation using orthonormal bases on both sides, avoiding the need for $A$ to have a full set of linearly independent eigenvectors.
**Follow-ups.** How does PCA use SVD instead of eigendecomposition of the covariance matrix? → Running SVD directly on the mean-centred data matrix avoids explicitly forming $X^TX$, which is numerically unstable and squares the condition number.
**Page.** [[eigen-decomposition-and-svd]]

### Q3. Derive the gradient of $f(x) = x^T A x$ with respect to $x$.
**Answer.** $\nabla_x (x^TAx) = (A + A^T)x$; if $A$ is symmetric this simplifies to $2Ax$. This shows up constantly in deriving normal equations and Hessians of quadratic losses.
**Follow-ups.** What's the Hessian of this function? → $A + A^T$ (constant, which is why quadratics have simple, global convergence properties for gradient descent).
**Page.** [[matrix-calculus-and-gradients]]

### Q4. Why is L1 regularization associated with sparsity but L2 is not? Give the geometric and the sub-gradient argument.
**Answer.** Geometrically, the L1 ball has corners on the axes; the loss contours are likelier to first touch the constraint region at a corner, zeroing out some coordinates. Algebraically, the L1 penalty's sub-gradient at 0 is a set $[-\lambda,\lambda]$, so any parameter whose data-gradient magnitude is smaller than $\lambda$ gets pushed exactly to 0 and stays there. L2's gradient shrinks proportionally to the weight value and vanishes as the weight approaches 0, so it never forces an exact zero.
**Follow-ups.** What is elastic net trying to fix? → Combines L1 sparsity with L2's better behaviour under correlated features (L1 alone picks one arbitrarily among correlated features).
**Page.** [[lagrange-multipliers-and-constraints]]

### Q5. State Bayes' theorem and use it to explain why a positive result on a rare-disease test can still mean you probably don't have the disease.
**Answer.** $P(D|+) = \frac{P(+|D)P(D)}{P(+)}$. If prevalence $P(D)$ is very low (say 0.1%) and the test has a nontrivial false-positive rate, the denominator $P(+) = P(+|D)P(D) + P(+|\lnot D)P(\lnot D)$ is dominated by the false positives from the huge healthy population, so $P(D|+)$ stays low even for a "95% accurate" test.
**Follow-ups.** How does this map onto precision in a classification problem? → $P(D|+)$ is exactly precision; this is why precision collapses on imbalanced data even with a good recall/specificity.
**Page.** [[bayes-theorem-and-conditional-probability]]

## Core

### Q6. Derive the KL divergence between two Gaussians and explain why it isn't symmetric.
**Answer.** For $\mathcal{N}(\mu_1,\sigma_1^2)$ and $\mathcal{N}(\mu_2,\sigma_2^2)$: $KL = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12$. Asymmetry comes from KL being an expectation under the first distribution $P$ of $\log(P/Q)$ — swapping $P$ and $Q$ changes which distribution's support/mass you're weighting the log-ratio by.
**Follow-ups.** Where does this asymmetry matter practically? → Forward KL ($P$=data) is mode-covering (used in MLE fitting); reverse KL ($P$=model) is mode-seeking (used in variational inference), giving qualitatively different fitted distributions on multimodal data.
**Page.** [[information-theory-entropy-kl]]

### Q7. Why does gradient descent with a badly-conditioned Hessian zig-zag, and what do momentum and Adam do about it?
**Answer.** A large condition number (ratio of largest to smallest Hessian eigenvalue) means the loss surface is a stretched ellipse; a step size good for the steep direction overshoots in the shallow one, causing oscillation. Momentum averages gradients over steps, damping the oscillating component while reinforcing the consistent one. Adam additionally rescales each coordinate by its running RMS gradient magnitude, approximating a per-coordinate adaptive step that flattens the effective condition number.
**Follow-ups.** Why can Adam generalize worse than SGD+momentum on some tasks? → Its adaptive per-parameter scaling can converge to sharper minima / different basins; this is an active empirical debate, not a settled theorem.
**Page.** [[gradient-descent-variants]]

### Q8. Set up and solve the Lagrangian for the hard-margin SVM.
**Answer.** Minimize $\frac12\|w\|^2$ s.t. $y_i(w^Tx_i+b)\ge 1$. Lagrangian: $L = \frac12\|w\|^2 - \sum_i\alpha_i[y_i(w^Tx_i+b)-1]$, $\alpha_i \ge 0$. Setting $\partial L/\partial w = 0$ gives $w=\sum_i \alpha_i y_i x_i$; $\partial L/\partial b=0$ gives $\sum_i\alpha_i y_i = 0$. Substituting back gives the dual, which depends on data only through dot products $x_i^Tx_j$ — the basis for kernel methods.
**Follow-ups.** What do the KKT complementary slackness conditions tell you about support vectors? → $\alpha_i>0$ only for points exactly on the margin (support vectors); everything else has $\alpha_i=0$ and doesn't affect $w$.
**Page.** [[lagrange-multipliers-and-constraints]]

### Q9. Explain the bias-variance decomposition of expected squared error, with the algebra.
**Answer.** $E[(y-\hat f(x))^2] = \text{Bias}[\hat f(x)]^2 + \text{Var}[\hat f(x)] + \sigma^2$. Derived by adding and subtracting $E[\hat f(x)]$ inside the square and using that the cross term vanishes in expectation. $\sigma^2$ is irreducible noise; the other two trade off as model complexity changes.
**Follow-ups.** Where does this decomposition break down in practice? → It assumes squared-error loss and a fixed training distribution; classification error, model averaging (ensembles), and double descent don't obey the same clean tradeoff.
**Page.** [[convexity-and-optimization-basics]]

### Q10. Why does $\sqrt{d_k}$ appear in the scaled dot-product attention formula? Do the variance derivation.
**Answer.** If $q,k$ have i.i.d. components with mean 0, variance 1, their dot product $q\cdot k = \sum_{i=1}^{d_k} q_ik_i$ has variance $d_k$ (sum of $d_k$ independent products each of variance 1). Without scaling, logits grow with $d_k$, pushing softmax into a saturated regime with near-zero gradients. Dividing by $\sqrt{d_k}$ renormalizes the variance back to 1.
**Follow-ups.** What happens if you skip this scaling in practice at high $d_k$ (e.g. 128)? → Attention weights become near one-hot early in training, gradients vanish through softmax, and training destabilizes or stalls.
**Page.** [[matrix-calculus-and-gradients]]

### Q11. What common probability distribution would you use to model: (a) number of support tickets per hour, (b) time between customer arrivals, (c) A/B test conversion outcome — and why?
**Answer.** (a) Poisson — counts of rare independent events in a fixed interval. (b) Exponential — memoryless continuous waiting time, the continuous analogue of the Poisson process. (c) Bernoulli per user, aggregated to Binomial across users — a single yes/no outcome per trial.
**Follow-ups.** When does Poisson break down for ticket counts? → When events aren't independent (an outage causes a burst), producing overdispersion — you'd then reach for a Negative Binomial.
**Page.** [[common-probability-distributions]]

### Q12. Derive why cross-entropy loss pairs naturally with softmax, i.e. why the gradient simplifies so nicely.
**Answer.** With softmax output $\hat y_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ and cross-entropy $L=-\sum_i y_i\log\hat y_i$, the gradient w.r.t. logits works out to $\partial L/\partial z_i = \hat y_i - y_i$ — a clean, bounded, well-scaled signal. This is because softmax is the inverse link function for the categorical distribution in the exponential family, and cross-entropy is the negative log-likelihood; the exponential-family algebra always cancels the normalizer's log-derivative against the loss's log.
**Follow-ups.** What breaks this cancellation? → Using MSE on softmax outputs instead of cross-entropy — the derivative then carries an extra $\hat y_i(1-\hat y_i)$ term that saturates and slows learning.
**Page.** [[expectation-variance-covariance]]

### Q13. What is convexity, and why is it such a big deal for optimization guarantees?
**Answer.** $f$ is convex if for all $x,y$ and $t\in[0,1]$, $f(tx+(1-t)y) \le tf(x)+(1-t)f(y)$; equivalently the Hessian is PSD everywhere. For convex $f$, any local minimum is global, so gradient descent with a suitable step size provably converges. Most deep learning losses are non-convex, which is why we settle for "good enough" local minima and rely on empirical/heuristic arguments (loss landscape flatness, overparameterization) rather than convergence guarantees.
**Follow-ups.** Is logistic regression's loss convex? Is a 2-layer neural net's? → Logistic loss is convex in the weights (composition of convex loss with linear map, plus convex log-sum-exp). A neural net's loss is generally non-convex due to the composition of nonlinear layers.
**Page.** [[convexity-and-optimization-basics]]

## Hard

### Q14. Derive the normal equation for linear regression from first principles and explain when you'd rather never form it explicitly.
**Answer.** Minimize $\|Xw-y\|^2$; gradient $2X^T(Xw-y)=0 \Rightarrow w = (X^TX)^{-1}X^Ty$. In practice you avoid forming $(X^TX)^{-1}$ directly: it squares the condition number of $X$, amplifying numerical error, and is $O(d^3)$. Instead use QR or SVD-based least squares solvers, which operate on $X$ directly and are numerically stable even when $X^TX$ is near-singular (multicollinearity).
**Follow-ups.** What if $X^TX$ is singular? → No unique solution; use the Moore-Penrose pseudoinverse (minimum-norm solution via SVD) or add ridge regularization to make it invertible.
**Page.** [[eigen-decomposition-and-svd]]

### Q15. Prove that KL divergence is always non-negative (Gibbs' inequality) and connect it to why cross-entropy is a valid loss to minimize.
**Answer.** $KL(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} = -\sum_x P(x)\log\frac{Q(x)}{P(x)} \ge -\log\sum_x P(x)\frac{Q(x)}{P(x)}$ by Jensen's inequality (log is concave) $= -\log\sum_x Q(x) \ge -\log 1 = 0$. Since $H(P,Q) = H(P) + KL(P\|Q)$ and $H(P)$ is fixed (the data's true entropy, not a function of the model), minimizing cross-entropy $H(P,Q)$ is exactly minimizing $KL(P\|Q)$, driving your model distribution $Q$ toward the true $P$.
**Follow-ups.** When is KL exactly zero? → Only when $P=Q$ almost everywhere — this is why cross-entropy has a floor at the data's true entropy, not zero.
**Page.** [[information-theory-entropy-kl]]

### Q16. Walk through why the Hessian of the softmax + cross-entropy loss in logistic regression is PSD, and what that guarantees.
**Answer.** For logistic regression the Hessian is $X^T S X$ where $S = \text{diag}(p_i(1-p_i))$ with $p_i\in(0,1)$, so $S$ has non-negative diagonal entries, making $X^TSX$ PSD for any $X$ (it's a weighted Gram matrix, quadratic form $v^TX^TSXv = \sum_i s_i(x_i^Tv)^2 \ge 0$). This guarantees the logistic loss is convex, so any stationary point found by gradient descent or Newton's method is the global optimum.
**Follow-ups.** Why can Newton's method converge in very few iterations for logistic regression compared to a deep net? → It uses the exact (or Fisher-approximated) Hessian to take curvature-aware steps, which is tractable at logistic-regression's parameter scale but $O(d^2)$-$O(d^3)$ cost is prohibitive at neural-net scale — hence first-order or quasi-Newton (L-BFGS) methods there.
**Page.** [[matrix-calculus-and-gradients]]

### Q17. You're told a dataset has 50 features and rank-deficient design matrix ($X^TX$ singular). Explain three distinct mathematical fixes and what each actually changes about the solution.
**Answer.** (1) Ridge: adds $\lambda I$ to $X^TX$ before inverting, which shifts all eigenvalues up by $\lambda$, guaranteeing invertibility and shrinking the solution toward zero along low-variance directions — biased but stable. (2) PCA/dimensionality reduction: project onto the top-$k$ singular directions, discarding the null-space directions entirely rather than shrinking them. (3) Pseudoinverse via SVD: picks the minimum-$L_2$-norm solution among the infinite exact solutions, without any bias-inducing shrinkage — but doesn't fix downstream numerical instability if used for prediction beyond the training data's span.
**Follow-ups.** Which of these is "doing regularization" and which is just "resolving non-uniqueness"? → Ridge and PCA change what solution is preferred (regularization, injecting bias for stability); the pseudoinverse alone just picks one exact solution out of infinitely many without changing the model class.
**Page.** [[lagrange-multipliers-and-constraints]]

## Related
See [[moc-maths]].
