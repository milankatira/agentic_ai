# 01 · LangGraph Fundamentals

## What it is

LangGraph is a low-level orchestration framework for building stateful LLM agents as **graphs of nodes and edges**. Each node is a Python function that reads and writes shared state. Edges control what runs next. It's built by LangChain Inc, but is independent — you can use it without LangChain.

## Why it matters

Production LLM apps aren't single calls — they're **workflows** with retries, branching, tool use, memory, and human checkpoints. LangChain "chains" are linear; LangGraph **graphs** can loop, branch, and recover. Almost every production agent system in 2026 (Klarna, Replit, Elastic, etc.) is built on it.

## Key concepts

| Concept | What it is |
|---|---|
| `StateGraph` | The graph builder. Knows the shape of shared state. |
| `State` | A `TypedDict` describing all data that flows through nodes. |
| Node | A function `(state) -> partial_state_update`. |
| Edge | A directed connection between nodes. Static or conditional. |
| `START` / `END` | Sentinel nodes for graph entry and exit. |
| Reducer (`add_messages`) | Function that says how to *merge* state updates (e.g. append messages instead of overwrite). |
| Conditional edge | Edge that calls a function to pick the next node at runtime. |
| `ToolNode` | Prebuilt node that executes any tool calls in the latest message. |
| `tools_condition` | Prebuilt router: "if last message has tool calls → tools, else → END". |
| Compilation | `graph_builder.compile()` returns the runnable `graph`. |

## Mental model

```
START → node_A → (conditional) → node_B → node_A   (loop)
                              → END
```

The graph passes a single `state` dict through nodes. Each node returns a *partial update*; reducers merge those updates back into state.

## Code patterns you must know cold

### Minimal state with message history
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]   # appends instead of overwrites
```

### Tool-calling agent skeleton
```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def llm_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")          # loop back so LLM can read tool result

graph = builder.compile()
```

### Streaming
```python
for event in graph.stream({"messages": "hi"}):
    for value in event.values():
        print(value["messages"][-1].content)
```

## What trips beginners up

- Forgetting the second arg to `add_node` (passing only the name) → silent `RuntimeError`.
- Passing the function object to `add_edge` instead of the string name.
- Forgetting `tools → llm` edge after `tools_condition` → graph completes after 1 tool call without ever using the result.
- Forgetting that `add_messages` *appends*; overwriting messages by accident.

## Mini-project (do this before moving on)

Build a **research assistant agent**:
- Tools: `TavilySearch` + a custom `calculator(expression: str) -> float`
- Should answer questions like: *"What was Apple's stock price yesterday, and what's 7% of that?"*
- Must loop tools → LLM until done
- Add a third tool `save_note(content: str) -> str` that writes to a local file

When you can build this from scratch without referring to a tutorial, move to topic 02.

## Resources

- [LangGraph docs](https://langchain-ai.github.io/langgraph/) — *official, current*
- [LangChain Academy: Introduction to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph) — *free, ~6 hours*
- [LangGraph examples repo](https://github.com/langchain-ai/langgraph/tree/main/examples) — *steal patterns from here*
