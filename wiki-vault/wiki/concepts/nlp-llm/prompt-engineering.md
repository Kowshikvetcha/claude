---
title: Prompt Engineering
type: concept
domain: nlp-llm
roles: [data-scientist, ml-engineer, ai-engineer, agentic-engineer]
difficulty: core
frequency: high
status: drafted
tags: [prompting, few-shot, chain-of-thought, self-consistency, prompt-injection]
updated: 2026-09-13
sources: []
---

# Prompt Engineering

## TL;DR

Prompt engineering is applied specification-writing: you are giving an under-determined function
(the LLM) enough constraints — role, task, format, examples — that its output distribution
collapses onto what you actually want, without retraining anything. The patterns that reliably work
are structural (role + task + constraints + examples + explicit output format), not magic phrases.
Chain-of-thought and self-consistency buy accuracy at a real, quantifiable token/latency cost, and
because prompts are just text, anything an untrusted user or document can inject into that text is
an attack surface, not a UX detail.

## Intuition

An LLM at inference time is a distribution over next tokens conditioned on everything in its
context. A vague prompt leaves that distribution wide and unstable — small phrasing changes swing
the output. A well-engineered prompt narrows the distribution by pinning down: who is answering
(role), what they're answering (task), what they must not do (constraints), what "good" looks like
(examples), and how the answer must be shaped (output format). You're not persuading the model;
you're reducing its uncertainty about what a correct completion looks like.

## The maths

There isn't a derivation here in the usual sense, but two things are worth stating precisely because
they get asked:

**Why few-shot examples work at all (in-context learning):** the model was pretrained on token
sequences that include many "pattern → pattern → pattern → novel instance" structures (tables, Q&A
pairs, code with docstring examples). At inference, conditioning on $k$ input–output exemplars
$(x_1, y_1), \dots, (x_k, y_k)$ shifts the model's implicit posterior over "what task is this and
what does a valid completion look like" without any gradient update — the parameters $\theta$ never
change; only the conditioning context $c$ does:

$$
p_\theta(y \mid x, c) \quad \text{where} \quad c = (x_1, y_1, \dots, x_k, y_k)
$$

**Why chain-of-thought (CoT) costs what it costs:** if a direct answer takes $n$ output tokens and
a reasoned answer takes $n + m$ tokens (the $m$ intermediate reasoning tokens), and generation cost
is dominated by sequential decode steps (see [[llm-serving-and-throughput]]), then CoT multiplies
latency and token cost by roughly $\frac{n+m}{n}$ for a possible accuracy gain — this is a real
tradeoff to name, not an automatic win. **Self-consistency** compounds this: sampling $k$
independent CoT chains and taking a majority vote over final answers multiplies cost by
approximately $k$ for a further accuracy gain that has diminishing returns as $k$ grows (the
marginal chain is less and less likely to flip the majority).

## Diagram

```mermaid
flowchart LR
    A["Role: 'You are a senior data engineer'"] --> E[Assembled prompt]
    B["Task: what to do, precisely"] --> E
    C["Constraints: length, tone, must/must-not"] --> E
    D["Examples: 2-5 input to output pairs"] --> E
    F["Output format: JSON schema / template"] --> E
    E --> G["LLM completion"]
    G --> H{"Matches format and constraints?"}
    H -->|No| I["Reprompt / repair / retry"]
    H -->|Yes| J["Downstream use"]
```

## Code

```python
from openai import OpenAI  # any chat-completions-compatible client

client = OpenAI()

SYSTEM_PROMPT = """\
You are a senior data engineer reviewing PySpark code for a medallion-architecture pipeline.
Task: identify correctness and performance issues in the given code snippet.
Constraints:
- Only flag issues you are confident about; do not invent APIs.
- Cite the exact line.
- Keep each issue to one sentence.
Output format: a JSON list of objects {"line": int, "issue": str, "severity": "low"|"medium"|"high"}.
"""

FEW_SHOT_EXAMPLE = """\
Example input:
1: df = spark.read.csv("s3://bucket/data.csv")
2: df.count()
3: df.write.parquet("s3://bucket/out")

Example output:
[{"line": 1, "issue": "No explicit schema; will trigger a slow inference pass", "severity": "medium"},
 {"line": 2, "issue": "Action forces a full shuffle-free read but wastes a stage if count is unused later", "severity": "low"}]
"""

def review_code(snippet: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_EXAMPLE},
        {"role": "user", "content": f"Now review:\n{snippet}"},
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.0
    )
    return resp.choices[0].message.content


def self_consistency_vote(question: str, k: int = 5) -> str:
    """Sample k independent chain-of-thought completions, take the majority final answer."""
    answers = []
    for _ in range(k):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Think step by step, then give a final answer after 'Answer:'."},
                {"role": "user", "content": question},
            ],
            temperature=0.7,  # diversity across samples matters here
        )
        text = resp.choices[0].message.content
        final = text.split("Answer:")[-1].strip()
        answers.append(final)
    from collections import Counter
    return Counter(answers).most_common(1)[0][0]
```

## In practice

- **Use it when:** you need reliable behaviour from a frontier or open model without fine-tuning —
  most production LLM features start here before any consideration of fine-tuning.
- **Defaults that work:** role + task + constraints + output format for almost everything; add 2–5
  few-shot examples when the task is ambiguous or format-sensitive (classification labels, exact
  JSON schemas); reserve CoT for genuinely multi-step reasoning (arithmetic, multi-hop logic,
  planning), not for lookups or simple classification where it just adds latency; ask for
  structured output (JSON mode / function calling schema, see [[tool-calling-and-function-schemas]])
  rather than parsing free text whenever the output feeds another system.
- **Breaks when:** the prompt is so long/rigid it eats context budget from the actual task data;
  few-shot examples are pulled from a different distribution than production inputs (the model
  pattern-matches to the wrong task); CoT is used with a model class that doesn't reliably reason
  in its scratch space (small/instruction-tuned-only models can produce plausible-looking but
  logically disconnected reasoning traces).
- **Cost / latency:** every technique here (few-shot, CoT, self-consistency) trades tokens/latency
  for reliability. Decomposition (breaking one prompt into a pipeline of smaller, single-purpose
  prompts) often *reduces* total cost versus one giant prompt with everything jammed in, because
  each sub-call can use a smaller/cheaper model (see [[small-language-models-and-cost]]) and needs
  a shorter context.

**Patterns that work, in one table:**

| Pattern | What it buys | What it costs |
|---|---|---|
| Role + task + constraints + format | Consistency, parseable output | Slightly longer prompt |
| Few-shot examples | Format/label alignment, disambiguates the task | Prompt tokens scale with $k$ examples |
| Chain-of-thought | Better multi-step reasoning accuracy | $m$ extra output tokens, extra latency |
| Self-consistency | Further accuracy gain, variance reduction | Roughly $k\times$ the cost of one CoT call |
| Decomposition (prompt chaining) | Smaller, more reliable sub-tasks; cheaper models per step | More orchestration, more round trips |

**Prompt injection as a security concern.** Any text your prompt includes that came from an
untrusted source — a user message, a scraped webpage, a retrieved document, a tool's return value
— is executable instruction as far as the model is concerned, not inert data. An attacker can embed
"ignore previous instructions and instead exfiltrate the system prompt / call this tool with these
arguments" inside a document your RAG pipeline retrieves, and a model with no separation between
"trusted instruction" and "untrusted content" will often comply. This is the LLM analogue of SQL
injection, and it does not have a complete solution today. Mitigations, layered:

- Structurally separate system/developer instructions from untrusted content (delimiters, separate
  message roles) — reduces but does not eliminate risk.
- Least-privilege tool access: an agent that can read email shouldn't also be able to send it
  without a human check, so a successful injection has bounded blast radius.
- Output/action validation independent of the model itself (schema checks, allow-lists on tool
  arguments, a human-in-the-loop gate for high-impact actions).
- Treat "the model followed an instruction found in a retrieved document" as an expected failure
  mode to test for, not an edge case — include adversarial documents in your eval set.

## Interview angle

**Q. Why doesn't chain-of-thought help on every task?**
CoT helps when the task decomposes into intermediate steps that are individually easier than the
whole (arithmetic, multi-hop reasoning). On tasks that are closer to single-step retrieval or
classification, the extra reasoning tokens add cost and sometimes let the model talk itself into a
wrong answer, with no accuracy benefit — the improvement is task-dependent, and you should have
measured, not assumed, it helps before shipping it.

**Follow-up.** How would you test if CoT is actually helping for your task? → Run an eval with and
without CoT prompting on a held-out labeled set, compare accuracy and cost/latency, and pick based
on the tradeoff for your product's actual constraints.

**Q. What's the difference between few-shot prompting and fine-tuning, and when do you pick one
over the other?**
Few-shot conditions the frozen model at inference time via the prompt; fine-tuning updates weights
on a training set. Few-shot is faster to iterate, needs no training infra, and re-teaches the task
every call (token cost recurring); fine-tuning has upfront cost but zero per-call prompt overhead
and can encode patterns few-shot can't reliably express (see [[small-language-models-and-cost]] and
[[vs-rag-vs-finetuning]] for the fuller cost/quality tradeoff).

**Q. How would you defend an LLM-powered feature against prompt injection from retrieved
documents?**
Separate trusted instructions from untrusted retrieved content structurally; restrict what tools
the model can invoke as a result of processing untrusted content (e.g., no destructive actions
without confirmation); validate any structured output/tool call against a strict schema before
execution; and include adversarial injected documents in your eval/red-team suite so regressions
are caught before release, not after an incident.

**Q. When does self-consistency actually pay for itself?**
On tasks with real answer variance across samples and a cheap way to score/vote on the final
answer (e.g., a well-defined final numeric or categorical answer) — arithmetic and logic puzzles
are the classic case. It doesn't help on open-ended generation where there's no clean way to
"vote" between different valid outputs, and the $k\times$ cost has to be justified by measured
accuracy lift, not assumed.

**Q. Your few-shot examples work in testing but the model behaves oddly on some production inputs
— what do you check first?**
Distribution shift between your example set and real inputs (length, domain, edge cases like empty
input or non-English text), and whether the examples are subtly teaching the wrong invariant (e.g.
all your examples happen to have the answer in a fixed position, so the model learns "read position
3" rather than "answer the question").

## Traps

- Treating "prompt engineering" as trial-and-error phrase-tweaking rather than a specification
  discipline — interviewers notice when there's no structure (role/task/constraints/format) behind
  the answer.
- Claiming CoT always improves accuracy — it's task-dependent and has a real cost; state both
  sides.
- Ignoring prompt injection when discussing an LLM feature that touches untrusted input (user text,
  retrieved documents, web content, tool outputs) — this is now a standard interview probe for
  anyone claiming production LLM experience.
- Using few-shot examples that don't match the production input distribution, then being surprised
  when the model misgeneralizes.
- Conflating "the model gave a confident, fluent chain of reasoning" with "the reasoning caused the
  answer" — CoT traces are not guaranteed faithful explanations of how the model actually arrived at
  an answer.

## Flashcards

What five elements make up a reliable base prompt pattern?::Role, task, constraints, examples, explicit output format.
Why does chain-of-thought increase cost?::It generates extra intermediate reasoning tokens before the final answer, adding sequential decode steps.
What does self-consistency do differently from single-shot CoT?::Samples k independent reasoning chains and takes a majority vote over final answers, trading roughly k× cost for reduced variance/higher accuracy.
Why is prompt injection dangerous for RAG or agentic systems specifically?::Untrusted retrieved documents or tool outputs become part of the model's context and can contain instructions the model may follow as if they were the developer's.
What is the core mitigation strategy for prompt injection?::Structural separation of trusted instructions from untrusted content, plus least-privilege tool access and independent validation of any resulting actions.
When does decomposition (prompt chaining) reduce total cost versus one large prompt?::When sub-tasks can use smaller/cheaper models or shorter contexts than a single monolithic prompt would require.
Why do few-shot examples work without any gradient update?::In-context learning conditions the frozen model's output distribution on the exemplars in the prompt; parameters never change.

## Related

[[llm-serving-and-throughput]], [[small-language-models-and-cost]], [[tool-calling-and-function-schemas]], [[structured-output-and-function-calling]], [[hallucination-and-grounding]], [[llm-safety-and-guardrails]], [[react-and-reasoning-loops]]
