# LangChain Learning Course

A hands-on, four-notebook walkthrough of LangChain's core primitives: models, messages, tools, and agents.

---

## Setup

**1. Python environment**

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt     # or your equivalent install command
```

**2. Environment variables**

Create a `.env` file in the repo root (or in `langchain/`):

```bash
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
```

- `GROQ_API_KEY` — sign up at [console.groq.com](https://console.groq.com/) (free tier).
- `GOOGLE_API_KEY` — get one from [aistudio.google.com](https://aistudio.google.com/apikey) (free).

**3. Launch Jupyter**

```bash
jupyter lab
```

Then open `langchain/01_intro.ipynb`.

---

## Learning path

Work through the materials in this order. Each notebook has a companion `.md` deep-dive — keep it open in a second pane while you run cells.

```
                                       ┌──────────────────────────┐
                                       │ langchain_theory.md      │  ← read first
                                       │ (the conceptual map)     │
                                       └──────────┬───────────────┘
                                                  │
        ┌────────────────────┬────────────────────┼──────────────────────┬──────────────────┐
        ▼                    ▼                    ▼                      ▼                  ▼
┌───────────────┐  ┌───────────────────────┐  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ 01_intro      │→ │ 02_model_integration  │→ │ 03_tools     │→ │ 04_messages      │→ │ 06_middleware     │
│ + 01_intro.md │  │ + ..._integration.md  │  │ + 03_tools.md│  │ + 04_messages.md │  │ + 06_middleware.md│
└───────────────┘  └───────────────────────┘  └──────────────┘  └──────────────────┘  └───────────────────┘
```

| # | Notebook | Companion deep-dive | One-line summary |
|---|---|---|---|
| 0 | — | [`langchain_theory.md`](./langchain_theory.md) | Conceptual reference; read this first. |
| 1 | [`01_intro.ipynb`](./01_intro.ipynb) | [`01_intro.md`](./01_intro.md) | Build your first ReAct agent with `create_agent` + `@tool`. |
| 2 | [`02_model_integration.ipynb`](./02_model_integration.ipynb) | [`02_model_integration.md`](./02_model_integration.md) | Invoke / stream / batch, with the unified `init_chat_model` interface. |
| 3 | [`03_tools.ipynb`](./03_tools.ipynb) | [`03_tools.md`](./03_tools.md) | `@tool`, `bind_tools`, and the manual tool-call loop. |
| 4 | [`04_messages.ipynb`](./04_messages.ipynb) | [`04_messages.md`](./04_messages.md) | The message schema: System, Human, AI, Tool — and how `tool_call_id` links them. |
| 6 | [`06_middleware.ipynb`](./06_middleware.ipynb) | [`06_middleware.md`](./06_middleware.md) | `SummarizationMiddleware` + `HumanInTheLoopMiddleware`: pause, summarise, approve / edit / reject. |

---

## Provider Map

The notebooks deliberately mix providers to teach the unified interface. Each notebook's lead-in cell repeats this so you know why you're seeing what you're seeing.

| Notebook | Provider | Model | Why |
|---|---|---|---|
| `01_intro` | `google_genai` | `gemini-2.5-flash` | Tool calling is solid; ReAct loop runs cleanly. |
| `02_model_integration` | mixed | qwen-3-32b (Groq) + gemini-2.5-flash + gemini-2.5-flash-lite | Demonstrates the *same* `.invoke()` across providers. |
| `03_tools` | `groq` | `qwen/qwen3-32b` | Groq exposes `reasoning_content` so you can see the model's tool-call thinking. |
| `04_messages` | `groq` | `qwen/qwen3-32b` | Continuity with `03_tools`. |
| `06_middleware` | `groq` | `qwen/qwen3-32b` | Same model used as agent **and** as the summariser inside `SummarizationMiddleware`. |

---

## Troubleshooting

**`DeprecationWarning: Inferred model_provider='google_vertexai'`**
→ You passed `"gemini-2.5-flash"` without a provider. Use `"google_genai:gemini-2.5-flash"` instead.

**`UserWarning: Your application has authenticated using end user credentials from Google Cloud SDK without a quota project`**
→ Same root cause — Vertex AI path is being taken. Switch to `google_genai`.

**Groq `rate_limit_exceeded` (HTTP 429)**
→ Free tier has tight per-minute caps. For `model.batch(...)`, set `config={"max_concurrency": 3}` or lower.

**`ImportError: cannot import name 'AIMessage' from 'langchain.messages'`**
→ Older `langchain` versions exposed messages at `langchain_core.messages`. Use:
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
```

**Jupyter doesn't pick up the `.env` file**
→ Make sure `load_dotenv()` runs *before* `init_chat_model(...)`. The first cell in each notebook does this.

---

## Next steps after the course

Once you finish these four notebooks, the natural follow-on topics are:

- **Structured output** (`with_structured_output(...)`) — force the model to return a Pydantic model.
- **Retrieval-Augmented Generation (RAG)** — vectorstores, retrievers, document loaders.
- **LangGraph** — for agents with more complex state machines than `create_agent` produces.
- **Tracing with LangSmith** — debugging and observability for chains and agents.
