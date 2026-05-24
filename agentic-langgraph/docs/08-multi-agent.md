# 08 · Multi-Agent Patterns

## What it is

Multi-agent systems = multiple LLM-driven "agents" with **different roles**, working together. Examples:
- **Researcher** finds info → **Writer** drafts → **Editor** revises
- **Planner** decomposes task → **Worker** agents execute subtasks → **Aggregator** combines results
- **Supervisor** routes user intent to one of N specialist agents

LangGraph supports several topologies natively.

## Why it matters

Single-prompt agents hit ceilings: they confuse roles, lose track of long plans, can't parallelize. Multi-agent decomposition unlocks harder tasks. This is where the field is heading in 2026 — every serious agent product is multi-agent under the hood.

Caveat: **most "multi-agent" pitches are over-engineered**. Start single-agent. Only split when you have a measured failure mode that decomposition fixes.

## Key concepts

| Concept | What it is |
|---|---|
| Supervisor / router | Top-level agent that dispatches to specialists. |
| Specialist / worker | Domain-specific agent. Often = one prompt + tool subset. |
| Subgraph | A LangGraph inside another LangGraph. Used to encapsulate an agent. |
| Handoff | Mechanism for transferring control between agents. |
| Network topology | Any-to-any. Agents talk freely. Powerful, fragile. |
| Hierarchical | Tree. Supervisor → subordinates. Most production systems. |
| Swarm | Peer agents pass tokens. Hot research area (OpenAI Swarm, CrewAI). |
| Reflection | Agent critiques and revises its own output. |

## Common topologies

### 1. Supervisor (most common)
```
            ┌─────────┐
   user ──▶ │supervisor│
            └────┬────┘
        ┌───────┼───────┐
        ▼       ▼       ▼
    researcher writer  coder
```
Use when intents fall into clean buckets.

### 2. Sequential pipeline
```
user → researcher → writer → editor → user
```
Use when stages are well-defined and order-dependent.

### 3. Reflexion / critic loop
```
generator → critic → (revise?) → generator → ... → done
```
Use for hard reasoning tasks where quality matters more than latency.

### 4. Network / swarm
```
agent A ⇄ agent B ⇄ agent C    (any-to-any)
```
Use for open-ended exploration. Hard to make reliable.

## Code patterns

### Supervisor with handoffs (LangGraph)
```python
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

def supervisor(state) -> Command[Literal["researcher", "writer", "__end__"]]:
    decision = supervisor_llm.invoke(state["messages"])   # returns "researcher"|"writer"|"done"
    target = "__end__" if decision == "done" else decision
    return Command(goto=target, update={"next": decision})

def researcher(state) -> Command[Literal["supervisor"]]:
    out = research_llm.invoke(state["messages"])
    return Command(goto="supervisor", update={"messages": [out]})

def writer(state) -> Command[Literal["supervisor"]]:
    out = writer_llm.invoke(state["messages"])
    return Command(goto="supervisor", update={"messages": [out]})

builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_edge(START, "supervisor")
graph = builder.compile()
```

### Sequential pipeline
```python
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "editor")
builder.add_edge("editor", END)
```

### Subgraph encapsulation
```python
# Build the writer as its own graph...
writer_graph = writer_builder.compile()

# ...then use it as a node in the parent graph
parent_builder.add_node("writer", writer_graph)
```

## Frameworks beyond LangGraph

| Tool | When to consider |
|---|---|
| **LangGraph** | Default. Maximum control, mature, integrates with everything. |
| **CrewAI** | Higher-level abstractions, faster prototyping. Less flexible. |
| **AutoGen (Microsoft)** | Strong for conversational multi-agent. Heavier. |
| **OpenAI Swarm** | Minimal, OpenAI-only. Good for learning patterns. |
| **Llama Index Agents** | If you're already deep in LlamaIndex. |

Learn LangGraph first. Add others only if a project demands them.

## What trips beginners up

- **Over-decomposition**: 7 agents for a task one agent could do. Adds latency + cost + bug surface.
- **Unclear roles**: agents step on each other.
- **No termination**: supervisor never declares "done" → infinite loop.
- **Shared state corruption**: two agents writing to the same key simultaneously.
- **Hidden context loss**: handing off without summarizing means the next agent restarts blind.
- **Skipping evals**: multi-agent makes outcomes noisier — you NEED evals (topic 05) more than ever.

## Mini-project

Build a **content production crew**:
1. **Researcher** agent: uses Tavily + your RAG over /docs to gather facts.
2. **Outliner** agent: produces a structured outline (Pydantic, from topic 03).
3. **Writer** agent: drafts each section against the outline.
4. **Editor** agent: critiques the draft and returns either `{"status":"approved"}` or `{"status":"revise", "feedback":"..."}`.
5. **Supervisor**: orchestrates; loops Writer → Editor until approved or 3 rounds.
6. Evaluate with a held-out set of 10 article topics. Track: word count, citation count, editor-approval rate, total cost.

Bonus: turn it into the topic-07 FastAPI deployment.

## Resources

- [LangGraph: Multi-agent concepts](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [LangGraph: Hierarchical agent teams](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/)
- ["Stop Building AI Tools Backwards"](https://hazyresearch.stanford.edu/blog/2025-01-05-flame) — when NOT to multi-agent
- [Anthropic: Building effective agents](https://www.anthropic.com/research/building-effective-agents) — classic essay
