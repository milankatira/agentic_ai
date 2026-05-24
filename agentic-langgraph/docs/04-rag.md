# 04 · Retrieval-Augmented Generation (RAG)

## What it is

RAG lets an LLM answer questions over **your private/recent data** by:
1. Chunking documents into pieces
2. Embedding each chunk into a vector
3. Storing vectors in a **vector database**
4. At query time, finding chunks most similar to the question
5. Passing those chunks to the LLM as context

The LLM "augments" its answer with **retrieved** content it never saw in training.

## Why it matters

~70% of "Applied AI Engineer" jobs in 2026 boil down to "build RAG over our docs." Customer support bots, internal knowledge bases, code search, legal/medical assistants — all RAG variants.

You will be asked about chunking, retrieval strategies, and eval metrics in interviews. Guaranteed.

## Key concepts

| Concept | What it is |
|---|---|
| Embedding | Vector representation of text. Similar text → similar vectors. |
| Chunking | Splitting docs into pieces (typically 200–1000 tokens) so they fit in retrieval + context. |
| Vector DB | Storage + ANN search over vectors. Examples: `pgvector`, `qdrant`, `weaviate`, `chroma`, `pinecone`. |
| ANN | Approximate Nearest Neighbor — fast similarity search. |
| Cosine similarity | The distance metric most embeddings use. |
| Top-K retrieval | Return the K most-similar chunks. |
| Reranking | Second-stage scoring with a smaller model to reorder Top-K. Big quality boost. |
| Hybrid search | Combine vector + keyword (BM25) search. |
| HyDE | "Hypothetical Document Embeddings" — ask LLM to draft a fake answer, embed that, retrieve against it. |
| RAG fusion / multi-query | Generate multiple query rewrites, retrieve for each, merge. |

## The minimal pipeline

```
PDF/markdown
  └─▶ chunker (split by tokens / semantic)
        └─▶ embedder (text-embedding-3-small, e.g.)
              └─▶ vector DB (qdrant, pgvector...)

query
  └─▶ embedder
        └─▶ vector DB → top-K chunks
              └─▶ reranker (optional)
                    └─▶ LLM with chunks in prompt → answer
```

## Code patterns

### Index documents (one-time)
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

docs = PyPDFLoader("manual.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

vectorstore = QdrantVectorStore.from_documents(
    chunks,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    location=":memory:",        # use real Qdrant URL in prod
    collection_name="manual",
)
```

### Retrieve + answer (RAG)
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def rag_node(state: State):
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return {"messages": [llm.invoke(prompt)]}
```

### Hybrid search (vector + keyword)
Use `EnsembleRetriever` to combine a `BM25Retriever` and a vector retriever. Often +10–20% recall.

## Chunking strategies (rank-ordered for most use cases)

1. **Recursive character splitter** with `chunk_size=600–1000`, `overlap=100`. Default.
2. **Semantic chunking** (split at semantic shifts). Slower but better for narrative docs.
3. **Document-aware** (split at markdown headers, code function boundaries). Best for structured docs.
4. **Fixed-size token chunks**. Simplest, OK fallback.

Tune chunk size empirically with **evals** (next topic).

## What trips beginners up

- Chunks too small → no context, fragmented answers.
- Chunks too big → top-K eats your context window.
- Forgetting overlap → information lost at chunk boundaries.
- Not deduping chunks → top-K returns near-duplicates, wasting slots.
- Not rerankering → top-K from pure embeddings is often noisy.
- Embedding queries with a *different* model than chunks → broken.
- Storing chunks without metadata (source filename, page) → can't cite.

## Mini-project — your portfolio piece

Build a **RAG over your personal documents**:
1. Pick a corpus: your notes (Obsidian/Notion export), or a few technical PDFs you care about.
2. Index with `qdrant` (Docker) or `pgvector`.
3. Retrieval: top-K=8, hybrid vector + BM25.
4. Add a reranker (`cohere-rerank-3` or `bge-reranker-v2-m3` local).
5. Wrap in a LangGraph agent with one tool: `search_docs(query)`.
6. **Cite sources** in answers (filename + page).
7. Evaluate retrieval with `Ragas` (next topic).
8. Deploy via FastAPI (topic 07).

This single project hits topics 4–7 and becomes your strongest interview talking point.

## Resources

- [LangChain: RAG tutorial](https://python.langchain.com/docs/tutorials/rag/) — start here
- ["Retrieval Augmented Generation: a survey"](https://arxiv.org/abs/2312.10997) — paper
- [Pinecone learning hub](https://www.pinecone.io/learn/) — best vendor-neutral writeups
- [Jina AI blog on rerankers](https://jina.ai/news/) — practical tips
- [Eugene Yan: Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)
