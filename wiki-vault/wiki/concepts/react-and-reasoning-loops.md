---
title: ReAct and Reasoning Loops
type: concept
domain: agents
roles: [ai-engineer, agentic-engineer, ml-engineer]
difficulty: core
frequency: high
status: drafted
tags: [react, agents, reasoning, tool-use, chain-of-thought]
updated: 2026-09-13
sources: []
---

# ReAct and Reasoning Loops

## TL;DR

ReAct (Reason + Act) interleaves explicit reasoning ("thought") with tool calls ("action") and
their results ("observation") in a loop, instead of doing all reasoning up front and then acting
once. That interleaving lets the agent revise its plan based on real information from the world
rather than committing to a plan derived purely from what it imagined would happen — the single
biggest reason it beats pure chain-of-thought on tasks that need external, changing, or unknown
information. It also introduces its own failure modes: infinite loops, tool misuse, and reasoning
that drifts from what the observations actually say.

## Intuition

Pure chain-of-thought is like planning a whole road trip from memory before checking a single map
or traffic report — confident, coherent, and potentially wrong the moment reality (a closed road,
a wrong assumption) diverges from what you assumed. ReAct is checking the map after every turn:
think about what to do next, do it, look at what actually happened, then think again. The agent
never commits more than one step ahead to unverified assumptions about the world.

## The maths

There isn't a formal derivation here — ReAct is a prompting/control-flow pattern, not a training
objective — but the loop has a precise structure worth stating exactly, because it's the interview
answer:

At each timestep $t$, given the trajectory so far $h_t = (T_1, A_1, O_1, \dots, T_{t-1}, A_{t-1}, O_{t-1})$:

1. **Thought** $T_t$: the model generates free-text reasoning about what it knows, what's missing,
   and what to do next — conditioned on $h_t$.
2. **Action** $A_t$: the model emits a structured tool call (name + arguments), derived from $T_t$.
3. **Observation** $O_t$: the environment/tool executes $A_t$ and returns a result, appended to the
   trajectory as ground truth the model did not generate.
4. Repeat, with $h_{t+1} = h_t \cup (T_t, A_t, O_t)$, until the model emits a final answer instead
   of another action.

The key property versus pure CoT: $T_t$ is conditioned on real observations $O_1, \dots, O_{t-1}$
that came from outside the model, not just on the model's own prior tokens. This grounds the
reasoning chain in the actual state of the world at each step, rather than in the model's
(possibly wrong) beliefs about what a tool would return.

## Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent (LLM)
    participant T as Tool / Environment

    U->>A: "What's the weather-adjusted demand forecast for tomorrow?"
    A->>A: Thought: need current weather + baseline forecast
    A->>T: Action: get_weather(location, date)
    T-->>A: Observation: "Rain, 18°C"
    A->>A: Thought: rain historically drops demand ~12%; need baseline
    A->>T: Action: get_baseline_forecast(sku, date)
    T-->>A: Observation: "baseline = 1,240 units"
    A->>A: Thought: apply weather adjustment to baseline
    A->>U: Final answer: "~1,090 units, adjusted for rain"
```

## Code

```python
from dataclasses import dataclass

@dataclass
class Step:
    thought: str
    action: str | None
    action_input: dict | None
    observation: str | None

def react_loop(question: str, tools: dict, llm_call, max_steps: int = 6) -> str:
    """Minimal ReAct loop. `llm_call(history)` returns either
    {'thought': ..., 'action': name, 'action_input': {...}} or
    {'thought': ..., 'final_answer': ...}."""
    history: list[Step] = []
    seen_actions = set()  # crude loop guard

    for step_num in range(max_steps):
        response = llm_call(question, history)

        if "final_answer" in response:
            return response["final_answer"]

        action = response["action"]
        action_input = response.get("action_input", {})
        fingerprint = (action, tuple(sorted(action_input.items())))

        if fingerprint in seen_actions:
            # Same action + same args repeated -> classic infinite-loop symptom.
            return "Aborting: repeated identical action detected, likely stuck."
        seen_actions.add(fingerprint)

        if action not in tools:
            observation = f"Error: unknown tool '{action}'. Available: {list(tools.keys())}"
        else:
            try:
                observation = tools[action](**action_input)
            except Exception as e:
                observation = f"Tool error: {e}"

        history.append(Step(response["thought"], action, action_input, str(observation)))

    return "Aborting: max steps reached without a final answer."
```

## In practice

- **Use it when:** the task needs information the model doesn't have and can't derive from
  reasoning alone — current data, a calculation too error-prone to do in-context, a lookup in a
  proprietary system, multi-step tasks where later steps depend on earlier results.
- **Defaults that work:** cap max steps hard (5–10 for most tasks); log the full thought/action/
  observation trace for every run (this is your debugging and eval surface, see
  [[agent-evaluation]]); give the model a way to say "I don't have enough information" rather than
  forcing an action every step; validate tool arguments against a schema before execution (see
  [[tool-calling-and-function-schemas]]).
- **Breaks when:** the tool set is ambiguous (multiple tools that look applicable, so the model
  guesses); observations are long/noisy and eat context budget across many turns; the task doesn't
  actually need iteration (a single-shot RAG or function call would do), in which case ReAct adds
  latency and cost for no benefit.
- **Cost / latency:** every loop iteration is a full LLM call plus a tool round-trip — cost and
  latency scale roughly linearly with the number of steps taken, so an unbounded or poorly-guided
  loop is a direct cost and latency risk, not just a correctness one.

**Why interleaving beats pure CoT, concretely:** pure CoT reasons about the world using only its
training-time knowledge and whatever's in the prompt — if it needs "today's exchange rate" it must
either hallucinate a plausible number or refuse. ReAct can call a tool for the actual rate mid-chain
and continue reasoning from the true value. This matters most when (a) the answer depends on
volatile or private information, (b) intermediate steps can fail in ways only discoverable by
trying them, or (c) the search space is too large to plan exhaustively up front and needs
information gathered along the way to prune it.

**Failure modes:**

- **Infinite / repetitive loops:** the model repeats the same (or near-identical) action because an
  observation didn't resolve its uncertainty, or because it misreads a failure as a reason to retry
  identically rather than adapt. Fix with a hard step cap, a repeated-action detector (see code
  above), and prompting the model to explicitly state why this attempt differs from the last.
- **Tool misuse:** wrong tool chosen, malformed arguments, or a tool called with inputs that are
  technically valid but semantically nonsensical for the task (e.g., searching with a query that
  restates the whole question instead of key terms). Fix with strict schema validation, clear
  per-tool docstrings/descriptions the model actually reads, and few-shot examples of correct
  invocation in the system prompt.
- **Observation misreading / reasoning drift:** the thought at step $t+1$ ignores or contradicts
  what $O_t$ actually said, especially over long trajectories where earlier observations fall out
  of the effective context. Fix with observation summarization and periodic re-grounding
  ("here is what we know so far") rather than relying on raw context accumulation — see
  [[agent-memory]].
- **Premature termination or over-iteration:** stopping with a low-confidence guess versus never
  stopping because the model always finds "one more thing to check" — both need explicit stopping
  criteria rather than leaving it to the model's judgment alone.

## Interview angle

**Q. Why does ReAct outperform plain chain-of-thought on tool-using tasks?**
Because CoT's reasoning is only ever conditioned on the model's own generated tokens and the
original prompt — it can't incorporate new information mid-reasoning. ReAct's thoughts are
conditioned on real observations returned by tools, so the reasoning chain can correct course based
on ground truth instead of the model's assumptions about what would happen.

**Follow-up.** Is ReAct strictly better than CoT, then? → No — for tasks that don't need external
information (pure logic/arithmetic self-contained in the prompt), ReAct adds tool-call latency and
cost with no benefit; CoT alone is often sufficient and cheaper.

**Q. How do you prevent an agent from looping forever?**
A hard maximum step count, detection of repeated identical (action, arguments) pairs, and giving
the model an explicit "insufficient information, terminate" exit path so it isn't forced to always
produce another action. Also log and eval on trajectories to see if loops correlate with specific
tool-description ambiguity, which is often the root cause.

**Q. An agent keeps calling the wrong tool for a task type. How do you debug it?**
Look at the raw thought/action/observation trace, not just the final answer — check whether the
tool's name/description is ambiguous relative to a similarly-named tool, whether few-shot examples
in the system prompt demonstrate correct selection, and whether the tool list is too large for the
model to reliably discriminate (consider narrowing the available toolset per task type, see
[[agent-guardrails-and-safety]]).

**Q. How would you evaluate whether ReAct is worth its added latency for a given task?**
A/B or offline comparison: same task set run with pure CoT/single-shot vs. ReAct, measured on
accuracy (task-specific), latency, and cost per resolved task — not accuracy alone. If the
uplift is marginal and the task rarely needs external information, the added round trips aren't
justified.

## Traps

- Describing ReAct as "just chain-of-thought with tools" — the interleaving (react to real
  observations, not just plan once and execute) is the actual mechanism, and missing it is a
  common giveaway of shallow understanding.
- Not naming failure modes (loops, tool misuse) when asked about ReAct in production — every
  agent framework question expects you to know what breaks, not just how it works.
- Using ReAct for tasks that don't need iteration — it's not a universal upgrade, it's a tool for
  tasks with genuine information-gathering needs.
- Assuming a longer trajectory (more steps) is always better/more thorough — beyond a point, context
  degradation and cost outweigh marginal reasoning improvement.

## Flashcards

What does ReAct interleave, and in what order?::Thought (reasoning) → Action (tool call) → Observation (tool result), repeated until a final answer.
What is the core reason ReAct beats pure chain-of-thought on tool-using tasks?::Its reasoning at each step is conditioned on real observations from tools/environment, not just the model's own prior generated tokens.
Name two concrete failure modes of ReAct loops.::Infinite/repetitive loops (same action retried) and tool misuse (wrong tool or malformed/nonsensical arguments).
How do you guard against an agent looping forever?::A hard max-step cap plus detection of repeated identical (action, arguments) pairs, and an explicit "insufficient information" exit path.
When is ReAct not worth using over plain CoT or a single-shot call?::When the task doesn't need external/changing information — the extra tool round trips add latency and cost with no accuracy benefit.
Why can reasoning "drift" over long ReAct trajectories?::Earlier observations fall out of effective context or get misweighted, so later thoughts stop being properly grounded in what was actually observed.

## Related

[[agent-fundamentals]], [[tool-calling-and-function-schemas]], [[planning-and-task-decomposition]], [[agent-memory]], [[agent-guardrails-and-safety]], [[prompt-engineering]], [[agent-evaluation]]
