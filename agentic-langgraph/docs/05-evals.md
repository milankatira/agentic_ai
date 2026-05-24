# 05 · Evals — How to know your AI works

## What it is

**Evaluations** = systematic, automated tests for LLM outputs. Instead of "I ran it 3 times and it looked right", you run hundreds of test cases and get **measurable metrics**: accuracy, faithfulness, latency, cost.

## Why it matters

This is the single biggest skill gap between *hobbyist* and *Applied AI Engineer*. Anyone can wire up a LangGraph + RAG demo. Few can answer: "How do you **know** retrieval improved when you swapped chunkers?"

Hamel Husain: *"The single highest-leverage activity in AI engineering is building good evals."*

Quote it in interviews.

## Key concepts

| Concept | What it is |
|---|---|
| Dataset | Collection of inputs (+ expected outputs) you test against. |
| Eval / scorer | Function that scores one model output. Returns bool, float, or struct. |
| LLM-as-judge | Using a (stronger) LLM to grade outputs of another LLM. Mainstream now. |
| Reference-based | Compare to a known-good answer (string match, embedding sim). |
| Reference-free | Score quality without a gold answer (faithfulness, coherence). |
| Pairwise | "Is A better than B?" — used for model/prompt comparisons. |
| Ragas | OSS lib of standard RAG metrics (faithfulness, answer relevancy, context precision/recall). |
| Regression suite | Eval suite that runs in CI on every prompt change. |

## The eval mindset shift

Treat your AI like a function:
- Inputs: queries
- Outputs: answers
- Tests: did the answer satisfy criteria?

Then write **lots of tests**. 20 cases minimum. 100+ is typical. Tag failures, fix root cause, never delete failing tests.

## Three layers of eval

1. **Unit evals** — single component (retriever returned right chunks? extractor parsed correct fields?).
2. **Integration evals** — end-to-end (user query → final answer correctness, faithfulness).
3. **Production evals** — sampled from real traffic, evaluated nightly. Catches drift.

## Code patterns

### Dataset + scorer pattern (vanilla Python)
```python
dataset = [
    {"q": "When was LangGraph released?", "expected_keywords": ["2023"]},
    {"q": "Who founded LangChain?", "expected_keywords": ["Harrison Chase"]},
    # ... 50+ more
]

def keyword_scorer(answer: str, expected: list[str]) -> float:
    return sum(1 for kw in expected if kw.lower() in answer.lower()) / len(expected)

scores = []
for case in dataset:
    answer = graph.invoke({"messages": case["q"]})["messages"][-1].content
    scores.append(keyword_scorer(answer, case["expected_keywords"]))

print(f"Mean: {sum(scores)/len(scores):.2%}")
```

### LLM-as-judge
```python
from pydantic import BaseModel

class Judgement(BaseModel):
    correct: bool
    reasoning: str

judge = init_chat_model("openai:gpt-4o").with_structured_output(Judgement)

def llm_judge(question: str, answer: str, reference: str) -> bool:
    verdict = judge.invoke(
        f"Question: {question}\n"
        f"Reference: {reference}\n"
        f"Candidate answer: {answer}\n"
        f"Is the candidate answer correct? Be strict."
    )
    return verdict.correct
```

### Ragas (RAG-specific)
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset=ragas_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
print(result)
```

### LangSmith (managed)
```python
from langsmith import Client
client = Client()

ds = client.create_dataset("my-rag-evals")
for case in cases:
    client.create_example(inputs={"q": case["q"]}, outputs={"a": case["a"]}, dataset_id=ds.id)

client.run_on_dataset(
    dataset_name="my-rag-evals",
    llm_or_chain_factory=lambda: graph,
    evaluators=[my_scorer],
)
```

## What trips beginners up

- "Vibes" eval — running 3 manual queries and saying "looks good." No.
- No dataset → no baseline → can't measure improvement.
- LLM judge that's too lenient (use a stronger model than the one being evaluated).
- Treating eval as one-off → never re-running after prompt edits.
- Aggregating all metrics into one score → loses signal. Track per-metric, per-tag.

## Mini-project

Add evals to your topic-04 RAG project:
1. Create a dataset of **50 questions** over your corpus, with reference answers.
2. Implement **4 metrics**: keyword overlap, LLM-judge correctness, faithfulness (Ragas), latency.
3. Run on 3 configurations: chunk_size=400 / 800 / 1500. Plot results.
4. Run on 2 retrievers: vector-only vs. hybrid. Compare.
5. Write a README explaining: what improved, what regressed, why.

This README is what an interviewer wants to read.

## Resources

- [Hamel Husain: Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) — *read this twice*
- [Hamel Husain: A Field Guide to Rapidly Improving AI Products](https://hamel.dev/blog/posts/field-guide/)
- [Ragas docs](https://docs.ragas.io/)
- [LangSmith docs: Evaluation](https://docs.smith.langchain.com/evaluation)
- [Eugene Yan: Eval LLMs and Apps](https://eugeneyan.com/writing/evals/)
