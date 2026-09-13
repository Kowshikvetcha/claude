---
title: Statistics Question Bank
type: qbank
domain: stats
roles: [data-scientist, ml-engineer]
difficulty: core
frequency: high
status: drafted
tags: [question-bank]
updated: 2026-09-13
---

# Statistics Question Bank

> How to use: cover the answers, write yours first, then compare. Anything you fumble → the linked concept page's `status` should go back to `drafted`.

## Warm-up

### Q1. What does a 95% confidence interval actually mean? Give the wrong answer people give and correct it.
**Answer.** The wrong (but common) answer: "there's a 95% probability the true parameter is in this interval." Correct: if you repeated the sampling-and-CI-construction procedure many times, 95% of the constructed intervals would contain the true (fixed, non-random) parameter. The randomness is in the interval, not the parameter.
**Follow-ups.** How does this differ from a Bayesian credible interval? → A credible interval *does* make the probability statement about the parameter directly, because the parameter is treated as a random variable with a posterior distribution.
**Page.** [[confidence-intervals]]

### Q2. State the Central Limit Theorem precisely and explain why it justifies using a z-test even when the underlying data isn't normal.
**Answer.** For i.i.d. random variables with finite mean $\mu$ and variance $\sigma^2$, the sample mean $\bar X_n$ satisfies $\sqrt{n}(\bar X_n - \mu)/\sigma \to \mathcal{N}(0,1)$ as $n\to\infty$. So even if the raw data is skewed or bounded, the *sampling distribution of the mean* becomes approximately normal for reasonably large $n$, which is what a z-test on a mean actually relies on.
**Follow-ups.** How large is "large enough"? → Depends on how far from normal/how skewed the underlying distribution is; rule of thumb $n \ge 30$ is common but a heavily skewed or heavy-tailed distribution needs much more, or a non-parametric alternative.
**Page.** [[central-limit-theorem]]

### Q3. Explain Type I and Type II error using a concrete business example, and state which one is more costly when launching a new fraud-detection model.
**Answer.** Type I (false positive): rejecting $H_0$ when it's true — flagging a legitimate transaction as fraud. Type II (false negative): failing to reject $H_0$ when it's false — missing actual fraud. In fraud detection the relative cost depends on the business: a false decline angers a good customer (revenue + trust loss); a missed fraud is a direct monetary loss. Usually missed fraud (Type II) is costlier in $, but false declines at scale (Type I) hurt retention — the real answer is "compute both costs and pick your threshold on expected loss," not a generic claim.
**Follow-ups.** How does the significance level $\alpha$ relate to Type I error rate? → $\alpha$ *is* the Type I error rate you're willing to tolerate by construction.
**Page.** [[type-i-and-type-ii-errors]]

### Q4. Why do p-values get misinterpreted so often? State the correct definition.
**Answer.** A p-value is $P(\text{data as extreme or more extreme} \mid H_0 \text{ true})$ — not $P(H_0\text{ true}\mid\text{data})$, and not the probability the result is due to chance. A p-value of 0.03 does not mean there's a 3% chance the null is true; it means that under the null, data this extreme would occur 3% of the time.
**Follow-ups.** What's the relationship between p-value and effect size? → None directly — a tiny p-value can come from a trivial effect with huge sample size, and a large p-value can hide a meaningful effect measured with too little data. Always report effect size + CI alongside p.
**Page.** [[p-values-and-significance]]

### Q5. What's statistical power, and how do sample size, effect size and $\alpha$ interact with it?
**Answer.** Power $= 1 - \beta$, the probability of correctly rejecting a false null. It increases with larger sample size, larger true effect size, larger $\alpha$ (more tolerance for false positives), and smaller variance. In practice you fix a desired power (typically 0.8), a minimum detectable effect, and $\alpha$, then solve for required sample size before running the test — not after.
**Follow-ups.** What happens if you peek at results early and stop when significant? → Inflates the true Type I error rate far above the nominal $\alpha$ (this is the classic "optional stopping" / peeking problem) — needs sequential testing corrections if you must peek.
**Page.** [[statistical-power-and-sample-size]]

## Core

### Q6. You ran an A/B test on 20 different metrics and 3 came back significant at p<0.05. What's wrong, and what would you do?
**Answer.** With 20 independent tests at $\alpha=0.05$, expected false positives by chance alone is 1, so 3 "hits" is barely above noise. Apply a multiple-testing correction: Bonferroni ($\alpha/m$, conservative) or Benjamini-Hochberg FDR control (less conservative, controls expected proportion of false discoveries among rejections) — the right choice depends on whether you need strict family-wise error control or just a a reasonable false-discovery rate for exploratory analysis.
**Follow-ups.** Why is Bonferroni often too conservative for real product teams? → It controls the probability of *any* false positive across all tests, which gets very strict as $m$ grows, killing power to detect real but modest effects — BH is usually the practical default.
**Page.** [[multiple-testing-correction]]

### Q7. Design an A/B test to measure whether a new recommendation algorithm increases session watch-time. Walk through the full design.
**Answer.** Define the metric (mean or median watch-time per session, pick based on skew), unit of randomization (user, not session, to avoid interference), compute MDE from historical variance and desired power, run a sample-size calculation, randomize with hashing on user ID for consistent bucketing, run for a full business cycle to avoid day-of-week confounds, pre-register the primary metric and stopping rule to avoid p-hacking, and analyze with a t-test or, if watch-time is skewed, a non-parametric test or log-transform / bootstrap CI.
**Follow-ups.** What guardrail metrics would you add? → Metrics that shouldn't regress even if the primary metric improves — e.g. crash rate, latency, unsubscribe rate, revenue per user — to catch a change that "wins" on the primary metric by degrading something else.
**Page.** [[ab-testing-design]]

### Q8. Name three classic A/B testing pitfalls beyond sample size, and how each is detected.
**Answer.** (1) Sample ratio mismatch (SRM): the observed traffic split deviates from the intended split (e.g. 48/52 instead of 50/50) — detect with a chi-square goodness-of-fit test on bucket sizes; if it fails, the whole experiment is suspect regardless of the metric result. (2) Novelty/primacy effects: an effect that decays or grows over the test window — detect by plotting the metric delta over time rather than trusting a single pooled number. (3) Network interference: users in the same social graph affecting each other across arms, violating SUTVA — mitigated with cluster-level randomization.
**Follow-ups.** Why does SRM alone invalidate an otherwise "significant" result? → It signals a bug in randomization or logging (e.g. one arm's page loads slower and drops more users before assignment is logged), meaning your two groups are no longer comparable — any metric difference could be confounded by whatever caused the mismatch.
**Page.** [[ab-testing-pitfalls]]

### Q9. Explain the difference between correlation, confounding and causation with an example, and describe one method to estimate a causal effect from observational data.
**Answer.** Ice-cream sales correlate with drowning deaths — both driven by a confounder (hot weather), not by one causing the other. To estimate causal effect from observational data without randomization, use techniques like propensity score matching (match treated/untreated units with similar covariate distributions to mimic randomization) or difference-in-differences (compare the *change* in outcome before/after treatment between treated and control groups, controlling for time-invariant confounders and common trends).
**Follow-ups.** What's the core assumption diff-in-diff relies on that's often violated? → The parallel trends assumption — that treatment and control groups would have moved similarly absent treatment; violated by e.g. treated stores being systematically higher-growth beforehand.
**Page.** [[causal-inference-basics]]

### Q10. Derive the MLE for a Bernoulli parameter and explain why MLE can overfit with small samples.
**Answer.** For $n$ i.i.d. Bernoulli$(p)$ trials with $k$ successes, log-likelihood $\ell(p) = k\log p + (n-k)\log(1-p)$; setting $d\ell/dp = 0$ gives $\hat p_{MLE} = k/n$. This overfits with small $n$: e.g. $k=0$ out of 3 gives $\hat p=0$, an extreme point estimate the data doesn't really support (a single future success would be "impossible" under this estimate, which is overconfident). A Bayesian approach with a Beta prior (e.g. Beta(1,1)) smooths this to $\hat p = (k+1)/(n+2)$ — Laplace smoothing.
**Follow-ups.** Where does this exact issue show up in practice with naive Bayes classifiers? → Zero-frequency problem — a word never seen with a class in training zeroes out the whole posterior for that class; fixed with additive (Laplace) smoothing.
**Page.** [[maximum-likelihood-estimation]]

### Q11. How would you use bootstrapping to get a confidence interval for a metric where you don't trust the CLT-based formula (e.g. median, or a ratio metric)?
**Answer.** Resample the observed data with replacement $n$ times (same size as original), compute the statistic on each resample, repeat thousands of times to build an empirical sampling distribution, then take the 2.5th/97.5th percentiles as the CI (percentile bootstrap), or use the bias-corrected-and-accelerated (BCa) variant for skewed statistics. This avoids relying on an analytical formula for the statistic's standard error, which often doesn't exist cleanly for medians or ratios.
**Follow-ups.** What's the difference between bootstrap and permutation testing, and when do you use which? → Bootstrap estimates the sampling distribution/CI of a statistic by resampling with replacement from one sample; permutation testing tests a null hypothesis (e.g. no difference between groups) by shuffling group labels and recomputing the test statistic to build a null distribution — bootstrap for estimation, permutation for hypothesis testing.
**Page.** [[resampling-bootstrap-and-permutation]]

### Q12. Explain Bayesian A/B testing (e.g. Beta-Binomial for conversion rate) and how you'd communicate results to a non-technical stakeholder differently from frequentist.
**Answer.** Model each arm's conversion rate with a Beta prior, update to a Beta posterior after observing conversions/non-conversions (conjugate to Binomial likelihood), then directly compute $P(\text{variant B better than A})$ by sampling from both posteriors and comparing, or via closed-form. This lets you say "there's an 87% probability B is better" — a direct probability statement stakeholders intuitively want, versus the frequentist "p<0.05, reject the null," which people constantly misread as that same probability statement anyway.
**Follow-ups.** What's the tradeoff versus frequentist testing? → Bayesian approaches need a prior (a source of both flexibility and potential bias/gameability) and don't have the same well-understood family-wise error control machinery for multiple simultaneous tests, though this is an active area with its own corrections.
**Page.** [[bayesian-inference-basics]]

### Q13. Give three distinct hypothesis tests you'd choose in three different scenarios, and why each is the right one.
**Answer.** (1) Comparing mean revenue per user between two independent groups with roughly normal-ish distributions and unknown equal variances → two-sample t-test (or Welch's t-test if variances differ). (2) Comparing conversion rates between two groups (categorical outcome) → chi-square test of proportions or a z-test for proportions. (3) Comparing a metric before/after for the *same* users (paired data, non-normal) → Wilcoxon signed-rank test rather than a paired t-test.
**Follow-ups.** Why prefer Welch's t-test over Student's t-test by default? → Student's assumes equal variances across groups; Welch's doesn't, and in practice unequal variances (e.g. different group sizes/distributions) are common enough that Welch's is now the safer default.
**Page.** [[common-statistical-tests]]

## Hard

### Q14. Walk through why "statistically significant" results from underpowered studies are often not just noisy but systematically overestimate effect size (the "winner's curse" / type M error).
**Answer.** With low power, only unusually large sample effect sizes cross the significance threshold — so among the subset of results that get published/acted on, the observed effect is a biased, inflated estimate of the true effect (type M, "magnitude," error), even though the test itself has correct Type I error control. This is why replications of "significant" underpowered findings routinely show smaller effects — regression to the mean plus selection on significance.
**Follow-ups.** What's a practical fix at a company running many small experiments? → Pool/shrink effect estimates across experiments (empirical Bayes / hierarchical modeling) rather than trusting each individual point estimate at face value, and prioritize sufficiently powered tests over a large volume of underpowered ones.
**Page.** [[statistical-power-and-sample-size]]

### Q15. You need to run sequential/continuous monitoring on an experiment (stakeholders want to see results daily) without inflating Type I error. How do you do it correctly?
**Answer.** Standard fixed-horizon tests assume you look once; looking repeatedly and stopping at the first significant p-value under standard $\alpha=0.05$ inflates the true false-positive rate substantially. Correct approaches: group sequential testing with alpha-spending functions (e.g. O'Brien-Fleming boundaries) that allocate a shrinking significance budget across looks, or "always valid" sequential testing methods (e.g. mixture SPRT-based confidence sequences) that give a valid p-value/CI at any stopping time by construction.
**Follow-ups.** Why do naive "peek daily, stop when p<0.05" setups fail so badly in practice? → Under the null, a random walk of the test statistic will cross the 0.05 boundary at some point with much higher than 5% probability if you check it enough times — the "multiple looks" problem is really multiple testing in disguise.
**Page.** [[ab-testing-pitfalls]]

### Q16. Explain Simpson's Paradox with a concrete example relevant to a hiring or conversion-rate analysis, and how you'd detect it.
**Answer.** A trend that holds in aggregated data reverses when the data is split by a confounding subgroup — classic example: department A hires women at a higher rate than men, department B does too, but pooled across departments it looks like women are hired at a lower rate, because women applied disproportionately to the more competitive department. Detect it by always checking whether a headline aggregate result is stable when sliced by plausible confounders (department, cohort, traffic source) before trusting it, and by understanding the causal structure (is the subgroup a confounder or a mediator — conditioning on a mediator can itself introduce bias).
**Follow-ups.** Why can't you just "always condition on every available subgroup" to be safe? → If the subgroup variable is a *mediator* (caused by treatment, and itself causing the outcome) rather than a confounder, conditioning on it blocks part of the true causal effect and biases the estimate the other way — you need the causal graph, not just the correlation, to know which variables to control for.
**Page.** [[causal-inference-basics]]

## Related
See [[moc-stats]].
