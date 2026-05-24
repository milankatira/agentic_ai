# 06 · Observability (Tracing, Cost, Latency)

## What it is

Observability = being able to **see inside** your LLM app at runtime: what prompts were sent, what tools fired, how long each step took, how many tokens it cost, and **why** a given output looked the way it did. Implemented via **tracing** (every step → a span in a trace) and **metrics** (counters, histograms, gauges).

## Why it matters

Without traces, debugging an agent is impossible. You'll have user reports like "it gave a weird answer at 3pm" and no way to reproduce. With traces, you click the trace ID, see every node, every prompt, every tool output, every retry. Two minutes to root cause.

Cost tracking is also non-negotiable. A buggy agent can burn $200/day in tokens without anyone noticing until billing.

## Key concepts

| Concept | What it is |
|---|---|
| Trace | A tree of spans for one user request, end-to-end. |
| Span | A timed unit of work (one LLM call, one tool call, one node). |
| LangSmith | LangChain's managed observability platform. Free tier exists. |
| Langfuse | OSS / self-hostable alternative to LangSmith. |
| OpenTelemetry (OTEL) | Vendor-neutral tracing standard. LangGraph exports OTEL traces. |
| Auto-instrumentation | LangChain/LangGraph integrate automatically with LangSmith via env var. |
| Token accounting | Tracking prompt + completion tokens per call. Multiply by price → cost. |
| Latency budget | The total time budget you allow per user request. |

## The 4 questions you must be able to answer in 30 seconds

For any given request in production:
1. **What prompt did the LLM see?** (full text, after templating)
2. **What did the LLM return?** (raw output)
3. **How long did it take?** (per step + total)
4. **How much did it cost?** (tokens × price)

If your stack can't answer these instantly, you don't have observability.

## Code patterns

### Enable LangSmith tracing (one env var)
```bash
# .env
LANGSMITH_API_KEY=ls__xxx
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=my-agent
```
That's it. All LangChain + LangGraph calls now appear in LangSmith UI.

### Tag traces by user/session
```python
from langsmith import traceable

@traceable(name="user_turn", metadata={"user_id": user_id})
def handle_message(graph, user_id, text):
    return graph.invoke({"messages": text}, config={"configurable": {"thread_id": user_id}})
```

### Custom span
```python
from langsmith import trace

with trace(name="custom-step", inputs={"x": 1}) as span:
    result = do_work()
    span.add_outputs({"y": result})
```

### Cost tracking via callback
```python
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    graph.invoke({"messages": "hi"})
print(cb.total_cost, cb.total_tokens)
```

### Self-hosted: Langfuse
```python
from langfuse.langchain import CallbackHandler
handler = CallbackHandler(public_key="...", secret_key="...")
graph.invoke({"messages": "hi"}, config={"callbacks": [handler]})
```

## What to alert on in production

- **Error rate** > X% over 5 min — bug or upstream outage
- **p95 latency** > budget — model regression or slow tool
- **Tokens per request** > expected — prompt growing unexpectedly
- **Cost per day** > budget — runaway agent
- **Tool failures** — Tavily down, DB unreachable, etc.

## What trips beginners up

- Building first, observing later — by the time you need traces, it's too late to add them.
- Logging only inputs/outputs without intermediate steps → still blind to *why*.
- Forgetting to filter logs for PII before storing.
- No structured metadata (`user_id`, `feature_flag`, `model_version`) → can't slice traces.
- Confusing tracing (per-request detail) with monitoring (aggregate metrics). You need both.

## Mini-project

Add observability to your topic-04 RAG project:
1. Wire up **LangSmith** with `LANGSMITH_TRACING=true`. Confirm every node shows.
2. Add `user_id` and `chunk_strategy` tags to every trace.
3. Track cost per query — log it. Build a tiny pandas notebook computing **$ per user-turn**.
4. Add a dashboard (Grafana or LangSmith's own) showing: p50/p95 latency, error rate, daily cost.
5. Deliberately break something (kill Qdrant). Confirm you can trace the failure to the failed tool span.

## Resources

- [LangSmith docs](https://docs.smith.langchain.com/) — start here
- [Langfuse docs](https://langfuse.com/docs) — OSS option
- [OpenLLMetry](https://github.com/traceloop/openllmetry) — OTEL for LLMs, vendor-agnostic
- [Charity Majors on observability](https://charity.wtf/) — general OBS philosophy, applies here
