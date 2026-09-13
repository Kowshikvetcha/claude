---
title: Deep Learning Question Bank
type: qbank
domain: deep-learning
roles: [data-scientist, ml-engineer, ai-engineer]
difficulty: core
frequency: high
status: drafted
tags: [question-bank]
updated: 2026-09-13
---

# Deep Learning Question Bank

> How to use: cover the answers, write yours first, then compare. Anything you fumble → the linked concept page's `status` should go back to `drafted`.

## Warm-up

### Q1. Why can a single-hidden-layer network with enough units approximate any continuous function (universal approximation), yet deep networks are still preferred in practice?
**Answer.** The universal approximation theorem guarantees existence of a wide-enough shallow network approximating any continuous function on a compact domain, but says nothing about *how many* units are needed or whether such a network is learnable by gradient descent. Depth lets you compose simple functions hierarchically — each layer reuses and combines features from the previous one — which empirically achieves the same expressivity with far fewer total parameters for functions with compositional structure (images, language), and is far more trainable in practice.
**Follow-ups.** What's the cost of that depth? → Deeper networks are harder to optimize (vanishing/exploding gradients, a more complex loss landscape) and need architectural aids (normalization, residual connections, careful initialization) a shallow net doesn't.
**Page.** [[neural-network-fundamentals]]

### Q2. Why did ReLU replace sigmoid/tanh as the default activation, and what failure mode does ReLU itself introduce?
**Answer.** Sigmoid/tanh saturate for large $|z|$, driving their derivative toward 0 ($\sigma'(z) \le 0.25$), which compounds multiplicatively across layers during backprop and causes vanishing gradients in deep nets. ReLU ($\max(0,z)$) has gradient exactly 1 for $z>0$, so gradients pass through unattenuated for active units, and it's cheap to compute. Its own failure mode is "dying ReLU": if a unit's weights drift such that its pre-activation is negative for every training example, its gradient is permanently 0 and it never updates again.
**Follow-ups.** How do Leaky ReLU / GELU address this, and why did GELU become standard in transformers? → Leaky ReLU gives a small non-zero slope for $z<0$ so the unit can recover; GELU is smooth and non-monotonic near 0 (weighting inputs by their probability under a Gaussian CDF), which empirically improves optimization in very deep transformer stacks over the hard ReLU kink.
**Page.** [[activation-functions]]

### Q3. Derive the backprop update for a weight in a 2-layer sigmoid network with MSE loss, showing where the chain rule multiplies through.
**Answer.** For $L = \frac12(y-\hat y)^2$, $\hat y = \sigma(z_2)$, $z_2 = w_2 a_1$, $a_1 = \sigma(z_1)$, $z_1 = w_1 x$: $\frac{\partial L}{\partial w_2} = (\hat y - y)\cdot \sigma'(z_2)\cdot a_1$, and one layer further, $\frac{\partial L}{\partial w_1} = (\hat y-y)\sigma'(z_2) \cdot w_2 \cdot \sigma'(z_1) \cdot x$ — each layer back adds one more multiplicative local-derivative term ($w_2$, then $\sigma'(z_1)$), which is exactly why the gradient at early layers is a *product* of many terms that can vanish or explode.
**Follow-ups.** Why is backprop "reverse-mode automatic differentiation" rather than "symbolic differentiation"? → It propagates actual numeric values forward, then actual gradient values backward through cached intermediates, reusing each computation once (linear in the number of operations), rather than building and simplifying a symbolic derivative expression.
**Page.** [[backpropagation]]

### Q4. Why does using MSE loss for multi-class classification train worse than cross-entropy, even though both are valid losses?
**Answer.** With a softmax output and MSE, the gradient w.r.t. logits carries an extra $\hat y_i(1-\hat y_i)$ factor from the softmax derivative, shrinking toward 0 exactly when the prediction is confidently wrong (saturated) — the case where the largest corrective gradient is needed. Cross-entropy's gradient w.r.t. logits is simply $\hat y_i - y_i$, staying proportional to the error regardless of saturation, so learning doesn't stall on confidently-wrong predictions.
**Follow-ups.** When is MSE still the right loss in deep learning? → Regression outputs with no probability interpretation, or Gaussian-distributed errors — pairing MSE with a linear output layer avoids the saturation issue entirely.
**Page.** [[loss-functions]]

### Q5. Why does initializing all weights identically break training, and how does Xavier/He initialization fix the deeper problem of signal scale across layers?
**Answer.** Identical initialization makes every unit in a layer compute the identical function and receive the identical gradient, so they update identically forever — the network collapses to one effective unit per layer regardless of width (the symmetry-breaking problem). Beyond that, activations can shrink toward 0 or blow up as they propagate through layers at the wrong random scale; Xavier/He init scales initial weight variance by $1/\text{fan\_in}$ (or $2/\text{fan\_in}$ for ReLU) specifically so activation variance is preserved layer-to-layer at initialization.
**Follow-ups.** Why does He init use a factor of 2 where Xavier uses 1? → ReLU zeroes roughly half its inputs, halving the variance passed through compared to a symmetric activation like tanh, so He doubles the initialization variance to preserve the same forward signal scale.
**Page.** [[weight-initialization]]

## Core

### Q6. Show why stacking many sigmoid layers causes vanishing gradients, and explain why this is a depth problem, not a per-layer one.
**Answer.** The gradient at layer 1 of an $n$-layer sigmoid network involves a product of $n$ terms of the form $w_k\sigma'(z_k)$; since $\sigma'(z)\le 0.25$ everywhere, each factor is bounded well below 1 unless weights are large enough to compensate (risking explosion instead) — the product shrinks geometrically, $O(0.25^n)$ worst case. A shallow 2-3 layer net doesn't suffer badly since there are too few multiplicative terms for the shrinkage to matter — it's specifically the compounding across many layers that vanishes gradients at early layers of a deep stack.
**Follow-ups.** Name two architectural fixes beyond switching to ReLU. → Residual/skip connections (an additive gradient path bypassing the multiplicative chain) and normalization layers (keeping pre-activation scale stable so $\sigma'(z)$ doesn't sit in the flat region).
**Page.** [[vanishing-and-exploding-gradients]]

### Q7. Derive Adam's update rule and explain what problem the bias-correction terms specifically solve.
**Answer.** Adam maintains EMAs of the gradient ($m_t = \beta_1 m_{t-1}+(1-\beta_1)g_t$) and its square ($v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$), then updates $\theta_t = \theta_{t-1} - \eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ — a per-parameter step scaled inversely by recent gradient magnitude. Both $m_0,v_0=0$, biasing early estimates toward 0 especially when $\beta_1,\beta_2$ are close to 1; bias-correction $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$ rescales early-step estimates back up to what an unbiased average would give, decaying to a no-op as $t$ grows.
**Follow-ups.** Why can plain SGD with momentum generalize better than Adam despite converging slower? → Adam's per-coordinate adaptive scaling can steer optimization into sharper minima or a different basin of the non-convex surface than SGD's isotropic steps — empirically correlated with worse generalization on some tasks, though the theory is still debated.
**Page.** [[optimizers-sgd-adam]]

### Q8. Why do transformer training runs almost always use learning-rate warmup before decay, instead of starting at the peak rate immediately?
**Answer.** Early in training, weights sit near initialization and Adam's second-moment estimate $v_t$ is still noisy (few samples averaged), so a large step at $t=0$ can push parameters into a bad region the optimizer struggles to recover from — worse in architectures with attention/layer norm stacked deeply. Warmup ramps the rate up slowly, letting moment estimates stabilize and the local landscape become better-conditioned, before switching to a decay schedule (cosine, linear, inverse-sqrt) as training approaches convergence.
**Follow-ups.** What breaks if you skip warmup on a very deep/large model? → Training loss can spike or diverge in the first few hundred steps — larger models are more sensitive since instability from an unstable early step propagates and amplifies through more compounding transformations.
**Page.** [[learning-rate-schedules]]

### Q9. Why do transformers use LayerNorm instead of BatchNorm, given BatchNorm is standard in CNNs?
**Answer.** BatchNorm normalizes each feature across the *batch* dimension, requiring stable batch statistics — this breaks for variable-length sequences (padding skews per-position batch stats) and for small/single-example batches (autoregressive generation, one token at a time). LayerNorm normalizes across the *feature* dimension per token independently, making it batch-size-invariant and consistent between training (large batches) and inference (batch size 1, token-by-token decoding).
**Follow-ups.** What's the mechanical optimization benefit of LayerNorm beyond enabling variable batch size? → It re-centers/rescales each layer's input distribution, keeping activation magnitudes well-conditioned for the next layer's weights and the optimizer's step size — part of why deep transformer stacks train without heavy per-layer tuning.
**Page.** [[batch-normalization-and-layernorm]]

### Q10. Explain dropout as approximate ensemble averaging, and derive why activations need rescaling between train and test.
**Answer.** Dropout randomly zeroes each unit with probability $p$ each forward pass — equivalent to training an exponential number of "thinned" sub-networks sharing weights; test time with the full network approximates averaging over all those sub-networks. But each unit was only active with probability $1-p$ during training, so at test time with all units active, outputs would be systematically larger than downstream layers were trained to expect. Inverted dropout scales surviving activations by $1/(1-p)$ *during training*, so test time needs no rescaling and expected activation magnitude matches.
**Follow-ups.** Why is dropout rarely used inside attention/FFN blocks of very large modern LLMs? → At pretraining scale, data is often large/diverse enough (and models trained for less than one epoch in some cases) that overfitting isn't the dominant risk the way it is on smaller labeled datasets, so scale and data diversity are prioritized over architectural regularization.
**Page.** [[dropout-and-regularization-dl]]

### Q11. Derive the receptive-field growth of stacked 3x3 convolutions, and explain why parameter sharing makes CNNs so much more sample-efficient than a fully-connected layer on images.
**Answer.** A single 3x3 conv gives each output unit a 3x3 receptive field; stacking a second 3x3 layer gives a 5x5 receptive field in the *original* input (overlapping 3x3 regions), and $n$ stacked $k\times k$ convs give receptive field $n(k-1)+1$ — growing linearly with depth while per-layer parameter count stays fixed. Parameter sharing (same small filter at every location) encodes the prior that a feature detector should behave identically regardless of position — a fully-connected layer would need separate weights per pixel location and could never generalize a detector learned in one position to an unseen one.
**Follow-ups.** Why do two stacked 3x3 convs often outperform one 5x5 conv with the same receptive field? → Fewer parameters ($2\times9=18$ vs $25$ per channel pair) plus an extra nonlinearity between them, increasing representational capacity for the same receptive field and lower compute.
**Page.** [[convolutional-neural-networks]]

### Q12. Derive why residual connections in ResNet solve the degradation problem — deeper plain networks performing worse even on training data.
**Answer.** In a plain deep network each block must learn the full mapping $H(x)$ from scratch; if the optimal mapping is close to identity, a stack of nonlinear layers struggles to approximate identity via gradient descent — an optimization failure, not overfitting, since even training error worsens with depth. A residual block instead learns $F(x)=H(x)-x$ and outputs $F(x)+x$; if identity is optimal, the block just drives $F(x)\to0$, a far easier target, and the additive skip path gives gradients a direct route back to earlier layers, bypassing the multiplicative vanishing-gradient chain.
**Follow-ups.** Does adding residual connections risk gradients "shortcutting" so much that layers stop learning anything useful? → No in practice — the residual path guarantees a stable minimum gradient flow, but each block's weight layers still receive their own local gradient through $F(x)$, so residual networks empirically learn meaningfully different features per block, not identity everywhere.
**Page.** [[cnn-architectures]]

### Q13. Explain how an LSTM's gating specifically prevents the vanishing gradient problem of vanilla RNNs, at the level of the recurrence equation.
**Answer.** A vanilla RNN's update $h_t=\tanh(Wh_{t-1}+Ux_t)$ means the gradient back through time is a product of $\prod_t W^T\text{diag}(\tanh'(\cdot))$ terms — repeated multiplication by $W$ and a sub-1 tanh-derivative compounds toward 0 (or explodes) over long sequences. An LSTM's cell state instead updates *additively*: $c_t = f_t\odot c_{t-1}+i_t\odot\tilde c_t$, gated by $f_t\in(0,1)$. When $f_t\approx1$, the gradient of $c_t$ w.r.t. $c_{t-1}$ is close to 1 — an identity-like path (similar in spirit to a ResNet skip) — giving gradients a route through many timesteps without repeated multiplicative shrinkage.
**Follow-ups.** Why does this only mitigate, not eliminate, long-range gradient issues, motivating attention/transformers? → The gates still involve learned, saturating sigmoids that can close, and cumulative effects across hundreds of steps can still degrade; attention instead gives direct, non-recurrent access from any output position to any input position in $O(1)$ path length, sidestepping the sequential-recurrence bottleneck entirely.
**Page.** [[recurrent-networks-and-lstm]]

## Hard

### Q14. Derive scaled dot-product attention end to end, including why the $1/\sqrt{d_k}$ scaling is necessary, and state its computational complexity.
**Answer.** Given queries $Q$, keys $K$, values $V$, attention computes $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ — $QK^T$ scores similarity between every query-key pair, softmax turns each row into a distribution over which tokens to attend to, and the result is a weighted average of value vectors. Without scaling, since $q\cdot k$ sums $d_k$ independent unit-variance terms, its variance grows with $d_k$; for large $d_k$ this saturates softmax (near one-hot, vanishing gradients). Dividing by $\sqrt{d_k}$ renormalizes variance back to about 1. Complexity is $O(n^2 d_k)$ in time and memory for sequence length $n$, since the full $n\times n$ attention matrix is materialized.
**Follow-ups.** Why is that $O(n^2)$ term the central bottleneck motivating FlashAttention and sparse/linear attention? → Memory and compute scale quadratically with sequence length, so doubling context length quadruples attention cost — at long context this dominates total cost, forcing either memory-access tricks (FlashAttention) or architectural approximations that skip most pairs (sparse/linear attention).
**Page.** [[attention-mechanism]]

### Q15. Why does the transformer need explicit positional encoding when RNNs don't, and why can transformers parallelize training across the sequence while RNNs can't?
**Answer.** Self-attention is permutation-equivariant by construction — $QK^TV$ treats input as an unordered set with no order baked in, unlike an RNN whose recurrence $h_t=f(h_{t-1},x_t)$ inherently encodes order through its dependency chain. Positional encoding injects order explicitly before attention. Parallelism follows the same distinction: an RNN's $h_t$ depends on $h_{t-1}$, forcing sequential computation; a transformer's attention for each output position depends only on already-available input representations, so every position can be computed simultaneously during training.
**Follow-ups.** Why does that parallelism advantage disappear during autoregressive inference? → Each new token still must be generated one at a time — its embedding must exist before it can be attended to — so decoding is inherently sequential regardless of architecture, which is exactly the gap KV-caching addresses by avoiding recomputation of prior tokens' attention on every step.
**Page.** [[transformer-architecture]]

### Q16. Explain data parallelism, tensor (model) parallelism, and pipeline parallelism for distributed training, and what specifically limits scaling in each.
**Answer.** Data parallelism replicates the full model per device, splits the batch, and all-reduces gradients after backward — simple as long as the model fits on one device, but all-reduce communication cost grows with model size/device count, eventually bottlenecking on network bandwidth. Tensor parallelism splits individual weight matrices/layers across devices, enabling models too large for one device, but needs frequent, low-latency communication within a single forward/backward pass — practical mainly within a fast-interconnect node. Pipeline parallelism splits the model by layer across devices, staggering micro-batches through — reducing per-step communication versus tensor parallelism but introducing "bubble" idle time while the pipeline fills/drains.
**Follow-ups.** Why do large LLM training runs combine all three ("3D parallelism") rather than picking one? → Each axis hits a different bottleneck (device memory, intra-node bandwidth, inter-node bandwidth) — combining lets tensor parallelism run within a fast-interconnect node, pipeline parallelism across nodes, and data parallelism across replicated node groups, matching each strategy to the communication pattern it tolerates best.
**Page.** [[distributed-training]]

### Q17. Do the memory arithmetic: how much GPU memory does training a 7B-parameter model in fp32 require just for weights/gradients/Adam states, and how does mixed precision change this?
**Answer.** Full fp32 training needs 4 bytes/param each for weights, gradients, and two Adam moments ($m$, $v$) — roughly 20 bytes/param total, or ~140 GB for 7B params, before activations. Mixed-precision training keeps a master fp32 copy of weights/Adam states (for numerical stability of small updates) but computes forward/backward in fp16/bf16, storing an additional low-precision weight copy — memory doesn't shrink as much as naively expected (still ~16-18 bytes/param), but compute throughput roughly doubles on tensor-core hardware and activation memory (scaling with batch size/sequence length) drops by half.
**Follow-ups.** Why does fp16 (not bf16) sometimes need loss scaling? → fp16 has a much narrower exponent range than bf16/fp32, so small gradients can underflow to zero; loss scaling multiplies the loss by a large constant before backward (keeping gradients in fp16's representable range) and divides gradients back down before the optimizer step, avoiding underflow without changing the update direction.
**Page.** [[mixed-precision-and-memory]]

### Q18. Derive the reparameterization trick used in a VAE and explain exactly why it's necessary for training with backprop.
**Answer.** A VAE's encoder outputs $\mu,\sigma$ for a latent $z\sim\mathcal{N}(\mu,\sigma^2)$, but sampling $z$ directly is a stochastic node with no defined gradient w.r.t. $\mu,\sigma$. The reparameterization trick rewrites $z=\mu+\sigma\odot\epsilon$ with $\epsilon\sim\mathcal{N}(0,I)$ sampled independently of the network's parameters — the randomness is isolated in $\epsilon$, external to the computation graph, and $z$ becomes a deterministic, differentiable function of $\mu,\sigma$, so gradients flow through them via standard backprop.
**Follow-ups.** What does the VAE's KL term regularize, and what happens if you drop it? → It penalizes the encoder's posterior $q(z|x)$ for deviating from the prior $\mathcal{N}(0,I)$, keeping the latent space smooth and preventing memorization of each input to an arbitrary point; dropping it collapses the model toward a plain deterministic autoencoder that reconstructs well but whose latent space has no meaningful structure for sampling/interpolation.
**Page.** [[autoencoders-and-representation-learning]]

## Related
See [[moc-deep-learning]].
