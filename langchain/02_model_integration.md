# 02 — Model Integration: Invoke, Stream, Batch — Deep Dive

> Companion to `02_model_integration.ipynb`.

## What you'll learn

- The two ways to construct a chat model: `init_chat_model("provider:model")` factory vs direct class import.
- How `BaseChatModel` unifies providers behind one interface.
- The three execution patterns (`invoke`, `stream`, `batch`), with a decision matrix.
- How `max_concurrency` interacts with provider rate limits.
- What lives in `response_metadata` and `additional_kwargs`.

---

## Core concepts

### The unified `BaseChatModel` interface

Every provider's wrapper class inherits from `BaseChatModel` and is therefore guaranteed to expose:

| Method | Returns | Use for |
|---|---|---|
| `.invoke(input)` | one `AIMessage` | single request/response |
| `.stream(input)` | iterator of `AIMessageChunk` | chunk-by-chunk reads |
| `.batch(inputs, config={...})` | list of `AIMessage` | parallel prompts |
| `.bind_tools(tools)` | new model with tools attached | tool-calling (see notebook 03) |
| `.with_structured_output(schema)` | new model that returns the schema | enforce Pydantic / JSON output |

That uniformity is the entire selling point of LangChain at the model layer. Once you've learned one provider, you've learned them all.

### Two construction paths

**Factory (`init_chat_model`)** — preferred for most code.

```python
from langchain.chat_models import init_chat_model

m1 = init_chat_model("openai:gpt-4o")
m2 = init_chat_model("anthropic:claude-sonnet-4-5")
m3 = init_chat_model("google_genai:gemini-2.5-flash")
m4 = init_chat_model("groq:qwen/qwen3-32b")
```

The factory imports the right package and constructs the subclass. Pass extra kwargs the same way:

```python
m = init_chat_model("openai:gpt-4o", temperature=0.2, max_tokens=1024)
```

**Direct class import** — when the factory doesn't surface a kwarg you need, or you want IDE autocomplete on provider-specific options.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

m = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0,
    max_output_tokens=512,
    safety_settings={...},   # Gemini-specific
)
```

Same `.invoke()` / `.stream()` / `.batch()` afterwards — only the construction differs.

---

## The three execution patterns

### Invoke — blocking single call

```python
response = model.invoke("Summarise the French Revolution.")
print(response.content)
```

- Returns one `AIMessage`.
- Blocks until the full response arrives.
- This is what every other pattern is built on.

### Stream — incremental chunks

```python
for chunk in model.stream("Explain token streaming in 3 paragraphs."):
    print(chunk.content, end="", flush=True)
```

- Yields `AIMessageChunk` — same interface as `AIMessage` but represents a partial response.
- Concatenate `.content` across chunks to assemble the full text.
- Only the **transport** changes; the model still generates the same tokens it would for invoke.

### Batch — parallel prompts

```python
responses = model.batch(
    ["Q1...", "Q2...", "Q3..."],
    config={"max_concurrency": 5},
)
```

- Returns a list of `AIMessage`, in input order.
- Uses a thread pool internally — each prompt is a separate HTTP request.
- `max_concurrency` caps simultaneous requests. **Set it below your provider's per-minute rate limit** or you'll get `429 Too Many Requests`.

### Decision matrix

| Use case | Pattern | Rationale |
|---|---|---|
| Classification, entity extraction, formatting | `invoke` | No UX benefit from streaming; just need the result. |
| Chat UI, long-form generation | `stream` | Perceived latency drops dramatically. |
| Bulk dataset labelling, parallel evaluation | `batch` | Network round-trips amortised across many prompts. |
| Inside an agent's tool-calling loop | `invoke` | The loop needs the complete `AIMessage` before deciding the next step. |

### `max_concurrency` and rate limits

Groq free tier is around 30 requests/minute on most models. If you call `model.batch(prompts, config={"max_concurrency": 30})` with 30 prompts, you'll hit the limit and start getting 429s on the next batch. Practical rule: divide your provider's per-minute cap by ~10, that's a safe `max_concurrency` for spikes.

---

## Walkthrough of the notebook

### Cells 1–2: setup

`load_dotenv()` before any `init_chat_model` call. Both `GOOGLE_API_KEY` and `GROQ_API_KEY` are needed since we touch both providers.

### Cells 3–7: three invoke variants

1. **`init_chat_model("google_genai:gemini-2.5-flash")`** — the factory path. Notice the explicit `google_genai:` prefix; without it, Vertex AI is chosen by default and you get a deprecation warning.
2. **`ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")`** — the direct-import path. Same `.invoke()`, just a different way to construct.
3. **`init_chat_model("groq:qwen/qwen3-32b")`** — different provider entirely, **same interface**. That's the point.

### Cell 9: streaming

The prompt is intentionally bland and technical so you can watch streaming clearly. With Groq's qwen model you'll also see `<think>...</think>` tokens — those are the model's reasoning trace, which Groq exposes in the stream.

### Cell 11: batching

Three unrelated prompts; the thread pool sends them concurrently. The `max_concurrency: 3` config means "up to 3 in flight at once" — since there are exactly 3 prompts, all three start immediately. The truncation in the print statement is cosmetic so the output stays readable.

---

## Anatomy of a response

Every `AIMessage` carries metadata. Worth knowing:

```python
response = model.invoke("Hi")

response.content              # the text answer
response.tool_calls           # list of tool-call requests (empty in this case)
response.response_metadata    # {"model_name": "...", "finish_reason": "stop", "token_usage": {...}, ...}
response.usage_metadata       # {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
response.additional_kwargs    # provider-specific extras (Groq reasoning_content, Gemini signatures)
response.id                   # unique id, useful for tracing
```

`usage_metadata` is the cross-provider-standard token count — prefer it over digging through `response_metadata["token_usage"]` which has provider-specific shapes.

---

## Common pitfalls

1. **Forgetting the provider prefix on Gemini.** `init_chat_model("gemini-2.5-flash")` defaults to Vertex AI and prints a deprecation warning. Always use `"google_genai:gemini-2.5-flash"`.
2. **Batching past the rate limit.** `max_concurrency` defaults to a sensible number, but if you raise it on a free-tier provider, expect 429s. Drop to 3–5 for Groq free tier.
3. **Streaming and not flushing.** Without `end=""` and `flush=True`, the print buffer eats your stream and you see nothing until the end — defeating the purpose.
4. **Treating `additional_kwargs` as standard.** It's provider-specific. Stable fields are `.content`, `.tool_calls`, `.response_metadata`, `.usage_metadata`.
5. **Concatenating chunks as `response.content`.** Each chunk is *itself* an `AIMessageChunk` — you concatenate `chunk.content` strings, not `response.content`.

---

## Further reading

- [`langchain_theory.md` §3](./langchain_theory.md#3-the-three-execution-patterns) — the same patterns in conceptual form.
- Next: [`03_tools.ipynb`](./03_tools.ipynb) — `bind_tools` and the manual tool-execution loop.
