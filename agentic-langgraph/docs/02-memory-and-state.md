# 02 · Memory & State (Checkpointing)

## What it is

**Memory** is the agent's ability to remember things across runs. LangGraph provides this via **checkpointers** — pluggable backends that persist graph state after every node execution. A checkpoint includes the full state dict plus a thread ID, so you can resume any conversation later.

## Why it matters

Without memory, every user turn is independent — the bot can't reference what the user said two messages ago. With memory, you get:
- **Multi-turn chat** that remembers context
- **Human-in-the-loop**: pause, ask a human, resume
- **Time travel debugging**: replay a graph from any checkpoint
- **Persistent agents** that survive process restarts

## Key concepts

| Concept | What it is |
|---|---|
| Checkpointer | Pluggable backend that saves state after each node. |
| `MemorySaver` | In-memory checkpointer. Disappears on restart. Dev only. |
| `SqliteSaver` | Persistent local SQLite-backed checkpointer. |
| `PostgresSaver` | Production checkpointer. Use this. |
| Thread ID | Unique conversation/session identifier. Pass via config. |
| `config={"configurable": {"thread_id": "..."}}` | How you tell the graph which conversation to load/save. |
| `graph.get_state(config)` | Inspect current state for a thread. |
| `graph.get_state_history(config)` | Replay all checkpoints — time travel. |
| Interrupt | Pause the graph mid-run, awaiting human input. |

## Short-term vs. long-term memory

- **Short-term (thread state)**: messages within one conversation. Handled by checkpointer.
- **Long-term (cross-thread)**: facts to remember across sessions ("user's name is Milan"). Implemented with a **Store** (key-value) — separate from the checkpointer. LangGraph provides `BaseStore` and an in-memory + Postgres impl.

## Code patterns

### Add checkpointing
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user-42"}}

graph.invoke({"messages": [HumanMessage("Hi, I'm Milan")]}, config=config)
graph.invoke({"messages": [HumanMessage("What's my name?")]}, config=config)
# Second call has access to first message via state — agent says "Milan"
```

### Inspect state history (time travel)
```python
for snapshot in graph.get_state_history(config):
    print(snapshot.values["messages"][-1].content, snapshot.next)
```

### Human-in-the-loop
```python
from langgraph.types import interrupt, Command

def approval_node(state):
    decision = interrupt({"question": "Approve this action?"})  # pauses here
    return {"approved": decision}

# resume after human input:
graph.invoke(Command(resume="yes"), config=config)
```

## What trips beginners up

- Forgetting `config` on subsequent `invoke()` calls → graph creates a fresh thread, no memory.
- Using `MemorySaver` in production → state lost on restart.
- Confusing state-within-a-thread (checkpointer) with cross-thread facts (Store).
- Letting message history grow unbounded → context overflows + cost balloons. Trim with a summarizer node.

## Mini-project

Extend your topic-01 research assistant with:
1. `SqliteSaver` so conversations survive restart.
2. A `/reset <thread_id>` CLI command that clears one thread.
3. A summarizer node: when `len(messages) > 20`, summarize the oldest 10 into one system message.
4. Human-in-the-loop: before any `save_note` tool call, pause and ask the user to confirm.

## Resources

- [LangGraph: Persistence concepts](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph: Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [LangGraph Memory tutorial](https://langchain-ai.github.io/langgraph/tutorials/memory/agentic_memory/)
