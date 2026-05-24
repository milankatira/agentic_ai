# Applied AI Engineer — Learning Path

A structured roadmap from "I built a tutorial chatbot" to "I can ship production AI systems."

Work through these in order. Each topic has a dedicated explainer with **what it is, why it matters, key concepts, concrete tools, and a mini-project**.

---

## The Path

| # | Topic | File | Time | Status |
|---|-------|------|------|--------|
| 01 | LangGraph Fundamentals | [01-langgraph-fundamentals.md](./01-langgraph-fundamentals.md) | 1–2 wk | doing now |
| 02 | Memory & State (Checkpointing) | [02-memory-and-state.md](./02-memory-and-state.md) | 3–5 days | next |
| 03 | Structured Outputs (Pydantic) | [03-structured-outputs.md](./03-structured-outputs.md) | 3 days | |
| 04 | Retrieval-Augmented Generation (RAG) | [04-rag.md](./04-rag.md) | 2–3 wk | |
| 05 | Evals — How to know your AI works | [05-evals.md](./05-evals.md) | 1–2 wk | |
| 06 | Observability (Tracing, Cost, Latency) | [06-observability.md](./06-observability.md) | 3–5 days | |
| 07 | Deployment (FastAPI, Docker, Cloud) | [07-deployment.md](./07-deployment.md) | 1–2 wk | |
| 08 | Multi-Agent Patterns | [08-multi-agent.md](./08-multi-agent.md) | 1–2 wk | |
| 09 | Python & SWE Fundamentals | [09-python-fundamentals.md](./09-python-fundamentals.md) | parallel | always |

---

## How to use this path

1. **Read the explainer**, no skipping.
2. **Do the mini-project at the bottom of each file** — building beats reading.
3. **Push your code to GitHub** with a README explaining what you built and why.
4. After topics 4–7, you should have one **real, deployed, traced, evaluated** RAG app — that's your portfolio piece.

## What you'll have at the end

- A deployed FastAPI service running a LangGraph agent with RAG, evals, and tracing
- 2–3 portfolio projects on GitHub with READMEs explaining trade-offs
- Confidence to read job descriptions and recognize ~90% of the terminology
- Real opinions on cost vs. quality trade-offs, model selection, retrieval strategies

## What you'll still NOT have (next phase)

- Fine-tuning experience
- Multi-modal (vision/audio)
- Distributed inference / on-prem deployment
- Production scale-out (Kubernetes, multi-region)

These are job 2 problems. Get hired first.

---

## Reading list (in priority order)

1. **LangChain Academy** — free, official, finish this first
2. **AI Engineering by Chip Huyen** (2024) — the textbook for this role
3. **Hamel Husain's blog** — best practical writing on evals
4. **Eugene Yan's blog** — practical RAG and ML patterns
5. **DeepLearning.AI short courses** — bite-sized, hands-on
