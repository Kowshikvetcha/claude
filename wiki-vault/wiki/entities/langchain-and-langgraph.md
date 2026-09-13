---
title: LangChain and LangGraph
type: entity
domain: agents
roles: [ai-engineer, agentic-engineer]
difficulty: core
frequency: high
status: drafted
tags: [tooling]
updated: 2026-09-13
sources: []
---

# LangChain and LangGraph

## What it is
LangChain is a framework for composing LLM calls with prompts, tools, memory, and retrieval into chains; LangGraph is its lower-level sibling for expressing agents as explicit state machines/graphs, built for the control that pure chains lack once logic branches, loops, or needs human intervention. Together they're one of the most common ways production LLM applications get built and, correspondingly, one of the most common things asked about in agentic-engineer interviews.

## Core concepts
- **Chains (LangChain "classic")**: a linear (or lightly branching) pipeline — prompt template → LLM call → output parser → next step — expressed declaratively via LCEL (`prompt | llm | parser`), the `|` operator composing `Runnable` objects.
- **Runnable interface**: the common abstraction (`invoke`, `batch`, `stream`, `ainvoke`) that prompts, models, retrievers, and parsers all implement — this uniformity is what makes chain composition with `|` work at all.
- **Tools & tool calling**: a `Tool` wraps a Python function with a name/description/schema the LLM can choose to invoke; LangChain standardizes this across model providers so the same tool definitions work with different underlying LLMs.
- **LangGraph's state machine model**: a graph of nodes (each a function that reads/updates shared state) and edges (including conditional edges that route based on state) — this is what lets an agent loop ("keep calling tools until done"), branch, or pause for human approval, none of which map cleanly onto a linear chain.
- **Checkpointing/persistence**: LangGraph can persist state at each step, enabling resumable, interruptible agent runs and human-in-the-loop pauses — a chain has no native concept of "pause here and resume later with human input."
- **Memory**: conversation history, summarization buffers, or vector-store-backed long-term memory attached to a chain/agent — separate from the LLM's context window, it's how state persists across turns/sessions. See [[agent-memory]].
- **Agent executor / ReAct loop**: LangChain's higher-level agent abstractions implement variants of reason-act-observe loops (see [[react-and-reasoning-loops]]) on top of tool calling — LangGraph is increasingly the recommended way to build this explicitly rather than through the more opaque legacy `AgentExecutor`.

## Code
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Simple LCEL chain
prompt = ChatPromptTemplate.from_template("Summarize this ticket in one sentence:\n\n{ticket}")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm | StrOutputParser()
summary = chain.invoke({"ticket": "User can't reset password, gets a 500 error..."})

# LangGraph: a minimal loop that calls a tool until the model stops requesting one
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list

def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)          # a ToolNode executing requested tool calls
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()
```

## When to use it vs alternatives
- **LangChain chains vs LangGraph**: use plain chains (LCEL) for linear, predictable pipelines (retrieve → prompt → generate); reach for LangGraph the moment you need loops, conditional branching, multiple agents coordinating, or human-in-the-loop checkpoints — forcing that logic into a chain gets brittle fast.
- **vs writing raw API calls**: for a single prompt-to-response call, LangChain adds abstraction overhead with little benefit; it earns its keep once you're composing multiple steps, swapping model providers, or need consistent tool-calling across models.
- **vs other agent frameworks**: LangGraph competes with things like a plain custom loop over an LLM API, or other orchestration frameworks — see [[agent-frameworks-landscape]] for the fuller comparison. LangChain/LangGraph's advantage is ecosystem breadth (integrations for most vector stores, model providers, tools already exist); the cost is an abstraction layer that occasionally makes debugging harder than a hand-written loop would be.

## Interview angle
**Q. When would you deliberately choose a hand-rolled loop over LangChain/LangGraph for an agent?**
When the logic is simple enough (a handful of tool calls, no branching) that the abstraction adds more debugging surface than it saves, or when you need to squeeze out latency/control that a framework's generality works against — many production teams start with LangGraph for prototyping and rewrite the hot path once the design stabilizes.

**Q. Why does LangGraph model agents as an explicit graph instead of a chain?**
Because real agent behaviour is rarely linear — it loops (call a tool, observe, decide whether to call another), branches (different paths depending on model output), and sometimes needs to pause for a human. A graph with conditional edges and per-node state updates expresses that directly; forcing it through a linear chain requires escape hatches (custom control flow bolted onto a supposedly-declarative pipeline) that erode the abstraction's value.

**Q. How do you keep a LangChain/LangGraph pipeline observable and debuggable in production?**
Attach tracing (e.g. LangSmith or an equivalent OpenTelemetry-based tracer) so every chain/graph step, prompt, and tool call is logged with inputs/outputs and latency — without it, a multi-step agent failure is very hard to localize to a specific node versus the LLM's own reasoning.

## Traps
- Reaching for a heavy agent framework when a single well-crafted prompt and one API call would do the job — added complexity with no accuracy or maintainability benefit.
- Treating LangChain's abstraction layers as a substitute for understanding what's actually being sent to the model — debugging requires knowing the underlying prompt/tool-call payload, not just the chain's Python code.
- Letting an agent loop run unbounded — always set a max-iteration/step limit; an agent that keeps calling tools without converging burns cost and can loop indefinitely.
- Assuming memory (conversation buffers) and the model's context window are the same thing — memory determines what *gets included* in the prompt; if it's not pruned/summarized, it silently exceeds the context window as conversations grow.

## Related
[[agent-fundamentals]], [[react-and-reasoning-loops]], [[tool-calling-and-function-schemas]], [[agent-frameworks-landscape]], [[multi-agent-systems]]
