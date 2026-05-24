# 09 · Python & Software Engineering Fundamentals

## What it is

The non-AI half of "Applied AI Engineer." Type hints, async, packaging, debugging, git, testing — the skills that let you ship reliable code regardless of the framework. Studied **in parallel** with topics 01–08, not after.

## Why it matters

The bugs in this repo so far — empty function body that silently returned `None`, malformed Google-style docstring, missing positional arg to `add_node` — were not AI problems. They were Python problems. A senior engineer would have spotted all three before running. Closing this gap is leverage on **every** future topic.

## What to be fluent in

### Python language
- Type hints (`list[int]`, `str | None`, `TypedDict`, `Protocol`, `Literal`)
- Dataclasses + Pydantic (when each)
- Async/await, `asyncio.gather`, `asyncio.TaskGroup`
- Context managers (`with`, `__enter__`/`__exit__`, `contextlib`)
- Generators (`yield`, `yield from`), iterators
- Decorators (writing and reading)
- Exceptions: define your own; never bare `except:` ; never silent `pass`
- f-strings, walrus operator, structural pattern matching (`match`/`case`)
- Comprehensions vs. loops vs. `map`/`filter`

### Reading errors
- Read the traceback **bottom-up**.
- The last frame tells you *what*. The frames above tell you *how you got there*.
- Most Python errors are 5 words: parse them, don't panic.
- `Cell In[N]` is the kernel's execution counter, NOT a file location.

### Debugging
- `breakpoint()` — drop into the debugger inline (Python 3.7+)
- `pdb` commands: `n`, `s`, `c`, `l`, `p var`, `pp var`, `w`, `u`/`d`
- VSCode's notebook debugger — set breakpoints in cells
- `python -X dev` flag — surfaces hidden issues
- Print-debugging is fine; just remove prints before commit

### Tooling
- `uv` — package + venv manager (you're using this, good)
- `ruff` — linter + formatter (single tool replaces black/flake8/isort)
- `mypy` or `pyright` — static type checking
- `pytest` — test runner; learn fixtures and parametrize
- `pre-commit` — run linters before every commit

Minimum `pyproject.toml` dev-dep set:
```toml
[tool.uv.dev-dependencies]
ruff = "*"
pytest = "*"
pyright = "*"
pre-commit = "*"
```

### Git
- Atomic commits; one logical change per commit
- Conventional commit prefixes (`feat:`, `fix:`, `docs:`)
- Branch per feature; `git rebase -i` to clean history before PR
- Read `git blame` and `git log -p` to understand "why was this written this way"

### Testing
- Write a test before fixing a bug (regression coverage)
- Pytest fixtures for setup/teardown
- Parametrize for table-driven tests
- Mock external services (`pytest-mock`)
- Aim for 80% coverage on business logic; don't chase 100%

### REST + HTTP basics
- Idempotency (GET = safe; POST = not; PUT = idempotent)
- Status codes (2xx success, 4xx client error, 5xx server)
- JSON content-type, CORS, auth headers
- `httpx` for async HTTP; never use bare `requests` in async code

## Specific bug archetypes you must learn to spot

The `multiply` function bug in `chatbot.ipynb` is a great teacher. Spotting these:

### 1. Function with empty body
```python
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    # ← no return statement → silently returns None
```
**Fix**: always return; or raise `NotImplementedError`.

### 2. Wrong docstring format
```python
def f(a: int, b: int):
    """
    Args:
        a (int): first       # ← Google style, must match signature exactly
        b (int): second
    """
```
LangChain parses this. Wrong format = `ValueError: Arg ... not found in function signature.`
**Fix**: use plain Google style (`a: first`) or just don't include `Args:` section.

### 3. Missing positional arg (silent default)
```python
builder.add_node("name",)    # ← trailing comma, second arg defaults to None → RuntimeError
```
**Fix**: always pass both args; ruff catches trailing commas in function calls.

### 4. Stale Jupyter cell
- Kernel restart ≠ cell reload.
- Editor buffer is independent of file on disk.
- Always check `In[N]` matches the saved cell content.

## How to study — concrete plan

Pick **one resource per category** and finish it. Don't bounce.

| Category | Resource |
|---|---|
| Python language | [Fluent Python (2nd ed)](https://www.fluentpython.com/) — Luciano Ramalho |
| Async | [Async IO in Python: A Complete Walkthrough](https://realpython.com/async-io-python/) |
| Type hints | [mypy cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) |
| Testing | [Brian Okken: Python Testing with pytest (3rd ed)](https://pragprog.com/titles/bopytest2/python-testing-with-pytest-second-edition/) |
| Git | [Learn Git Branching (interactive)](https://learngitbranching.js.org/) |
| Software design | [A Philosophy of Software Design](https://web.stanford.edu/~ousterhout/cs190-winter18/lectures/) — John Ousterhout |
| Debugging | [Julia Evans's debugging zines](https://wizardzines.com/) |

## Daily habits

- Run `ruff check .` and `pyright` before every commit.
- Read 1 error message per day all the way through. Don't just paste into ChatGPT.
- After fixing a bug, write down (a) the symptom, (b) the root cause, (c) the lesson — one sentence each.
- Once a week, read someone else's pull request on a major OSS project (LangChain, FastAPI, Pydantic). Learn from the review comments.

## Mini-project

Pick **any 50-line function** in a public LangGraph example and:
1. Add complete type hints
2. Add a docstring
3. Add 3 pytest tests (happy path, edge case, error case)
4. Run `ruff` and `pyright` until clean
5. Open a PR upstream (yes, really — they're friendly)

That PR is portfolio gold. It signals "this candidate ships clean code, not just notebooks."
