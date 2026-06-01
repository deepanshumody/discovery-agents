# Contributing

Thanks for your interest in `discovery-agents`. This project favors small, focused,
well-tested changes.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # core is dependency-free; dev adds pytest, ruff, mypy
```

To run against a real model, install a provider extra and set its key:

```bash
pip install -e ".[anthropic]"  # or .[cohere] / .[openai]
export ANTHROPIC_API_KEY=...   # then: discovery-agents --provider anthropic
```

Everything runs **keyless by default** via the deterministic `MockLLMClient`, so you
do not need an API key to develop or run the tests.

## Quality gates (run before every commit)

```bash
ruff check src tests          # lint
ruff format src tests         # format
mypy                          # strict type check
pytest -q                     # tests (keyless, deterministic)
```

CI runs the same gates on Python 3.9–3.12.

## Conventions

- **Keyless-by-default:** no change may require an API key to run the demo or tests.
  Provider SDKs are imported lazily so the core stays dependency-free.
- **Typed:** all functions are fully annotated (`disallow_untyped_defs`).
- **Cited or flagged:** generated content cites evidence or is caught by a guardrail.
- **Observable:** new agent/tool/LLM calls emit a trace span.
- **Tested:** new behavior ships with a deterministic test against the mock backend;
  tests needing real keys are marked `@pytest.mark.live` (skipped by default).

## Commit messages

Use conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `test:`,
`refactor:`).
