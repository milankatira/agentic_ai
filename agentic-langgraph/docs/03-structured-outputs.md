# 03 · Structured Outputs (Pydantic, JSON Mode)

## What it is

Forcing the LLM to return a **structured, validated object** (Pydantic model / JSON schema) instead of free-form text. Modern LLM APIs support this natively via "JSON mode", "tool calling", or "structured outputs" features.

## Why it matters

Free-form LLM text is unparseable. Real applications need to:
- Extract entities (`{"name": "...", "email": "..."}`)
- Classify (`{"category": "billing", "urgency": "high"}`)
- Decide actions (`{"action": "refund", "amount": 49.99}`)

Without structured outputs you regex-parse strings and pray. With them, you get **type-safe, schema-validated Pydantic objects**.

This is THE daily-use skill. Every production LLM app uses structured outputs somewhere.

## Key concepts

| Concept | What it is |
|---|---|
| `with_structured_output(Schema)` | LangChain method: bind a Pydantic model as the output type. |
| Pydantic `BaseModel` | Type-safe class with validation. Define your schema here. |
| `Field(description="...")` | Field metadata. Becomes part of the prompt — write it well. |
| Strict mode | Provider enforces schema at decode time (OpenAI, recent Anthropic). |
| Tool calling | Same machinery under the hood — the LLM returns args matching your tool's signature. |
| Few-shot examples | When the schema isn't enough, show examples in the prompt. |

## Code patterns

### Basic extraction
```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class Contact(BaseModel):
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    phone: str | None = Field(default=None, description="Phone if mentioned")

llm = init_chat_model("groq:llama-3.3-70b-versatile")
structured_llm = llm.with_structured_output(Contact)

result = structured_llm.invoke("Hi, I'm Milan Katira, milan@example.com")
print(result.name, result.email)   # type-safe Pydantic object
```

### Classification
```python
from typing import Literal

class Ticket(BaseModel):
    category: Literal["billing", "technical", "feedback"]
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(description="One-sentence summary")

structured_llm = llm.with_structured_output(Ticket)
structured_llm.invoke("My credit card was charged twice!")
```

### Nested objects + lists
```python
class LineItem(BaseModel):
    product: str
    quantity: int
    price: float

class Order(BaseModel):
    customer: str
    items: list[LineItem]
    total: float

structured_llm = llm.with_structured_output(Order)
```

### Inside a LangGraph node
```python
def classify_node(state: State):
    classification = structured_llm.invoke(state["messages"])
    return {"classification": classification}   # State must have a 'classification' key
```

## What trips beginners up

- Vague `Field(description=...)` → garbage output. Treat descriptions like prompts.
- Using `dict` or untyped fields → no validation; you've gained nothing over plain JSON.
- Forgetting `Literal[...]` for enums → LLM invents values.
- Schemas too deep / too many fields → LLMs struggle. Keep schemas flat where possible.
- Mixing structured output with tool calling on small models → fight over the same machinery, errors.

## Mini-project

Build a **resume parser**:
1. Define Pydantic models for `Education`, `Experience`, `Skill`, `Resume` (with nested lists).
2. Feed in 5 real resumes (PDFs converted to text with `pypdf`).
3. Validate the output — count fields filled vs. missing.
4. Compare: how does parsing quality change between `llama-3.3-70b` and `llama-3.1-8b`?

## Resources

- [LangChain: Structured outputs](https://python.langchain.com/docs/concepts/structured_outputs/)
- [Pydantic docs](https://docs.pydantic.dev/latest/)
- [Instructor library](https://python.useinstructor.com/) — provider-agnostic structured outputs, retry on validation failure
- [OpenAI Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs)
