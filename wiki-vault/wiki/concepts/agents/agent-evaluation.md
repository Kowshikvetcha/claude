---
title: Agent Evaluation
type: concept
domain: agents
roles: [agentic-engineer, ai-engineer]
difficulty: advanced
frequency: high
status: drafted
tags: [agents, evaluation, llm-as-judge, trajectory]
updated: 2026-09-11
sources: []
---

# Agent Evaluation

## TL;DR
Evaluating an agent means judging both whether it reached the goal (task success) and
how it got there (trajectory) — a lucky wrong path that stumbles onto the right answer
should score worse than a clean one, because it won't generalize. This makes agent
eval structurally harder than single-turn LLM eval: the output space is a sequence of
decisions with a huge branching factor, not one response.

## Intuition
Grading a single-turn LLM answer is like grading an essay — you read the final text.
Grading an agent is like grading a chess game — the final position matters, but so does
whether each move made sense given the position at the time. An agent that deletes the
wrong file and then "fixes" it by deleting the right one too still shouldn't get full
marks just because the directory ends up correct.

## The maths
There isn't a closed-form derivation here, but the *decomposition* is the interview
content. Define an agent trajectory as a sequence of (state, action, observation)
triples:

$$
\tau = \left( (s_0, a_0, o_0), (s_1, a_1, o_1), \ldots, (s_T, a_T, o_T) \right)
$$

Task success is a terminal, binary (or graded) judgment on the final state $s_{T+1}$
against a goal specification $G$:

$$
\text{Success}(\tau) = \mathbb{1}[s_{T+1} \models G]
$$

Trajectory quality is a judgment over the *path*, typically decomposed per-step and
aggregated — e.g. an LLM judge scores each $a_t$ given $(s_t, o_{t-1})$ against a
rubric (necessity, correctness of tool choice, no destructive/irreversible action
without gate), and the aggregate is something like a weighted average or a
minimum (a single catastrophic step should tank the score, not average out):

$$
\text{TrajectoryScore}(\tau) = \min_t \; \text{StepScore}(s_t, a_t, o_t)
$$

using `min` rather than `mean` is itself an interview-relevant design choice: it
models the reality that one irreversible bad action (deleting prod data) isn't
compensated by nine good ones.

## Diagram
```mermaid
flowchart TD
    A["Agent trajectory recorded"] --> B["Task success check<br/>did final state satisfy goal?"]
    A --> C["Trajectory evaluation<br/>was each step reasonable?"]
    C --> D["Step-level LLM judge<br/>rubric: necessity, tool choice, safety"]
    B --> E["Combined score"]
    D --> E
    E --> F["Regression suite over<br/>held-out task set"]
```

## Code
```python
# Minimal sketch: LLM-as-judge scoring a single agent trajectory.
# In practice the judge model, prompt, and rubric are versioned like model code.

from dataclasses import dataclass

@dataclass
class Step:
    state_summary: str
    action: str
    observation: str

def build_judge_prompt(goal: str, steps: list[Step], final_state: str) -> str:
    trace = "\n".join(
        f"Step {i}: state={s.state_summary} | action={s.action} | result={s.observation}"
        for i, s in enumerate(steps)
    )
    return f"""You are grading an AI agent's trajectory.

Goal: {goal}

Trajectory:
{trace}

Final state: {final_state}

Score on two axes, 1-5 each, and justify briefly:
1. TASK_SUCCESS: did the final state satisfy the goal?
2. TRAJECTORY_QUALITY: was every step necessary, safely chosen, and non-redundant?
   A single irreversible or destructive unnecessary action caps this at 1,
   regardless of the outcome.

Return JSON: {{"task_success": int, "trajectory_quality": int, "reasoning": str}}"""

# Run this judge across a fixed eval set of tasks with known-good reference
# trajectories, then track task_success and trajectory_quality as separate
# regression metrics over time — collapsing them into one number hides
# "got there by luck" failures.
```

## In practice
- **Use it when:** any agent that takes multiple actions, especially with side effects
  (writing files, calling paid APIs, sending messages, modifying data) — a chatbot with
  no tools can mostly rely on single-turn LLM eval instead.
- **Defaults that work:** maintain a fixed, versioned eval set of tasks with known-good
  reference trajectories (or at least known-good *checkpoints* the trajectory must
  pass through); score task success and trajectory quality separately, never as one
  blended metric; re-run the eval suite on every prompt, tool-schema, or model change,
  the same discipline as a model quality gate in [[ci-cd-for-ml]].
- **Breaks when:** the task has many valid paths (e.g. "research this topic") — a
  rigid reference trajectory over-penalizes valid alternative strategies, so the judge
  rubric needs to score *properties* of good paths (efficiency, no redundant calls, no
  unsafe actions) rather than exact-path matching.
- **Cost / latency:** LLM-as-judge over full trajectories is expensive at scale (you're
  paying for another full model pass per eval case) — sample a subset for expensive
  frontier-judge runs and use cheaper heuristic checks (schema validation, step count,
  did-it-call-the-required-tool) for the rest of the CI gate.

## Interview angle
**Q. Why is agent evaluation harder than evaluating a single LLM call?**
A single-turn eval judges one input-output pair against a rubric or reference. An
agent's output space is a sequence of decisions, each conditioned on the (partially
unpredictable) result of the previous action — the same starting task can legitimately
produce many different valid trajectories, and an invalid trajectory can still
accidentally land on the correct final answer. You need both an outcome metric and a
process metric, and they can disagree.

**Follow-up.** Give a concrete example where task success and trajectory quality
disagree. → An agent asked to "clean up temp files" that runs `rm -rf` on a broader
directory than intended, and happens to not delete anything important this time —
task succeeded, trajectory was reckless and would fail catastrophically on a slightly
different filesystem state.

**Q. How would you build a regression test suite for an agent that browses the web?**
Fix a set of representative tasks with stable (or mocked) environments — real websites
change, so either snapshot the pages/API responses the agent will see, or use a
sandboxed test environment. Define success criteria per task (exact answer, or a
verifier function), log full trajectories, and score both success rate and trajectory
metrics (number of steps, redundant navigation, whether it recovered from a dead end
versus looping).

**Follow-up.** How do you keep this suite from becoming flaky? → Pin external
dependencies (mock or snapshot the environment), fix the model/temperature where
possible for a deterministic core, and track flake rate itself as a metric — a task
that passes 60% of the time on an unchanged agent is telling you something about
variance, not about the agent being "sometimes right."

**Q. What does using `min` instead of `mean` for trajectory scoring buy you, and what
does it cost?**
It buys you sensitivity to a single catastrophic step — appropriate when actions are
irreversible or high-stakes. It costs you granularity: a genuinely good trajectory
that stumbles on one minor, recoverable misstep gets the same rock-bottom score as one
that did real damage, so you often want a hybrid — a hard gate on any
safety-critical step, and a mean/weighted score for the rest.

## Traps
- Reporting only task success rate — it hides "right answer, terrifying path" failures
  that will bite in production the moment the environment differs slightly.
- Using an LLM judge with no fixed rubric or reference — scores drift between runs and
  aren't comparable across model or prompt versions; version the judge prompt like code.
- Treating agent eval as a one-time pre-launch check — tool schemas, prompts, and
  underlying models change continuously, so this needs to be a CI gate, not a one-off
  report (see [[ci-cd-for-ml]]).
- Using the same model as both the agent and the judge without any structural
  separation — self-evaluation bias is real; where possible use a different model
  family or at least a fixed, non-agent-influenced judge prompt.

## Flashcards
What two axes does agent evaluation need beyond single-turn LLM eval?::Task success (did the final state meet the goal) and trajectory quality (was the path reasonable and safe).
Why can task success alone be misleading for agents?::A trajectory can reach the correct final state through an unsafe, redundant, or lucky path that won't generalize or is unacceptable in production.
Why might you aggregate step scores with min instead of mean?::To make a single catastrophic or irreversible action dominate the score, since averaging would let it hide behind otherwise-good steps.
What's a practical way to keep a browsing/tool-using agent's eval suite non-flaky?::Snapshot or mock the external environment so tasks are deterministic and reproducible instead of depending on live, changing websites/APIs.
Why version an LLM-judge prompt like code?::Because judge scores aren't comparable across runs if the rubric silently drifts — you need to attribute score changes to the agent, not the judge.

## Related
[[agent-fundamentals]]
[[react-and-reasoning-loops]]
[[llm-evaluation]]
[[agent-guardrails-and-safety]]
[[ci-cd-for-ml]]
[[ml-testing-strategy]]
