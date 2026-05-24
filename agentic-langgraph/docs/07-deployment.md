# 07 · Deployment (FastAPI, Docker, Cloud)

## What it is

Taking your LangGraph agent from a Jupyter notebook to a **publicly accessible, always-on HTTP service** that real users can hit. The default path: FastAPI → Docker → cloud (Railway, Fly.io, Render, AWS, GCP).

## Why it matters

A notebook is not a product. Every "real" AI engineer role expects you can deploy. Even if your team has DevOps, you must be able to ship a service end-to-end alone.

## Key concepts

| Concept | What it is |
|---|---|
| FastAPI | Modern async Python web framework. The default for AI APIs. |
| ASGI | Async server interface. `uvicorn` is the standard ASGI server. |
| Streaming | Sending the LLM response token-by-token to the client (SSE/WebSocket). |
| Dockerfile | Recipe for building a portable container image. |
| Image | Built artifact. Push to GHCR / Docker Hub / cloud registry. |
| Health check | `/health` endpoint. Cloud load balancers check it. |
| Env vars | Secrets / config injected at runtime. Never bake API keys into images. |
| Horizontal scaling | More copies of the service behind a load balancer. |
| Cold start | First-request latency when a container spins up. Matters on serverless. |
| LangGraph Cloud / LangGraph Platform | Managed deployment from LangChain Inc. Skip FastAPI entirely. |

## The minimal production shape

```
client ──HTTPS──▶ load balancer ──▶ FastAPI (uvicorn) ──▶ LangGraph + LLM + tools
                                          │
                                          ├──▶ Postgres (checkpointer + state)
                                          ├──▶ Qdrant (vectors)
                                          └──▶ LangSmith (traces)
```

## Code patterns

### Minimal FastAPI wrapper
```python
# app/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .graph import graph     # your compiled LangGraph

app = FastAPI(title="my-agent")

class ChatRequest(BaseModel):
    thread_id: str
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    cfg = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke({"messages": req.message}, config=cfg)
    return {"reply": result["messages"][-1].content}

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    cfg = {"configurable": {"thread_id": req.thread_id}}
    def gen():
        for event in graph.stream({"messages": req.message}, config=cfg):
            for v in event.values():
                yield v["messages"][-1].content + "\n"
    return StreamingResponse(gen(), media_type="text/plain")
```

Run locally: `uvicorn app.main:app --reload`

### Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen

COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build + run:
```bash
docker build -t my-agent .
docker run -p 8000:8000 --env-file .env my-agent
```

### docker-compose for local dev
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres, qdrant]
  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: dev
    volumes: ["pgdata:/var/lib/postgresql/data"]
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes: ["qdrant:/qdrant/storage"]
volumes:
  pgdata:
  qdrant:
```

### Deploy: easiest options ranked

1. **Railway** (`railway up`) — simplest, free tier, just works.
2. **Fly.io** (`flyctl launch`) — great DX, global edge, generous free tier.
3. **Render** — also simple, similar to Railway.
4. **LangGraph Platform** — official managed runtime. Use if you don't want infra.
5. **AWS / GCP / Azure** — flexibility, but more YAML. Save for later.

### Streaming to a browser (SSE)
```python
from sse_starlette.sse import EventSourceResponse

@app.get("/sse")
async def sse(message: str, thread_id: str):
    async def gen():
        async for ev in graph.astream({"messages": message}, config={"configurable": {"thread_id": thread_id}}):
            yield {"data": list(ev.values())[0]["messages"][-1].content}
    return EventSourceResponse(gen())
```

## Production checklist

- [ ] Secrets in env vars, not source
- [ ] `/health` endpoint returns DB/vector-DB connectivity status
- [ ] LangSmith tracing enabled with `LANGSMITH_PROJECT` set
- [ ] Postgres checkpointer (not `MemorySaver`)
- [ ] Token-budget / cost-cap per request
- [ ] Rate limiting (`slowapi` or upstream)
- [ ] CORS configured for your frontend domain
- [ ] Structured logs (JSON) to stdout
- [ ] Image scanned (`trivy image my-agent`)
- [ ] CI builds + pushes image on every main commit

## What trips beginners up

- Baking API keys into Docker images → leaked secrets when image is shared.
- No `EXPOSE` / wrong port mapping → "connection refused".
- Forgetting `--host 0.0.0.0` in `uvicorn` inside Docker → unreachable from outside container.
- Synchronous blocking calls in async endpoints → blocks the event loop, kills throughput.
- Not pinning model versions → silent behavior drift in production.
- Treating logs as optional → impossible to debug prod issues.

## Mini-project

Deploy your topic-04 RAG agent:
1. Wrap it in FastAPI with `/chat`, `/chat/stream`, `/health`.
2. Add Postgres checkpointer + Qdrant via docker-compose.
3. Write a Dockerfile. Build. Run locally with compose.
4. Push to GitHub. Deploy to **Railway** or **Fly.io**.
5. Build a tiny HTML page that calls `/chat/stream` and renders tokens as they arrive.
6. Hit your live URL from your phone. Take a screenshot. Add to README.

That screenshot is now in your portfolio.

## Resources

- [FastAPI docs](https://fastapi.tiangolo.com/) — best-in-class
- [LangGraph Platform docs](https://langchain-ai.github.io/langgraph/cloud/)
- [Fly.io Python guide](https://fly.io/docs/python/)
- [Railway docs](https://docs.railway.app/)
- [12-Factor App](https://12factor.net/) — production-readiness bible
