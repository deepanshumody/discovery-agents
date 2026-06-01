# Adding a tool

Tools are how agents act on the world (retrieval, calculation, external APIs). A tool
is any object satisfying the `Tool` protocol in `discovery_agents/tools/base.py`:

```python
name: str
description: str
parameters: dict          # JSON schema advertised to the model
def run(self, arguments: dict) -> ToolResult: ...
```

`ToolResult(ok, data, citations, error)` carries the result plus citation ids for
grounded, auditable answers.

## Example: a units tool

```python
from discovery_agents.tools.base import ToolResult


class CelsiusToFahrenheitTool:
    name = "c_to_f"
    description = "Convert a temperature from Celsius to Fahrenheit."
    parameters = {
        "type": "object",
        "properties": {"celsius": {"type": "number"}},
        "required": ["celsius"],
    }

    def run(self, arguments: dict) -> ToolResult:
        try:
            c = float(arguments["celsius"])
        except (KeyError, TypeError, ValueError):
            return ToolResult(ok=False, error="celsius (number) is required")
        return ToolResult(ok=True, data={"fahrenheit": c * 9 / 5 + 32})
```

## Register it

```python
from discovery_agents.tools import ToolRegistry

registry = ToolRegistry([CelsiusToFahrenheitTool()])
registry.run("c_to_f", {"celsius": 100})        # -> ToolResult(ok=True, data={"fahrenheit": 212.0})
registry.specs()                                # model-facing ToolSpec list for the ReAct loop
```

The registry exposes `specs()` (the model-facing schemas) and a crash-safe `run(name,
arguments)`. Pass the registry to `runtime.agent.LLMAgent` to let a ReAct loop call it,
or to the pipeline so its agents can use it.

## Guidelines

- **Deterministic by default.** Like `WebSearchTool`, give external tools a deterministic
  default backend so demos/CI are reproducible; inject a live backend for production.
- **Never crash the loop.** Return `ToolResult(ok=False, error=...)` instead of raising;
  the registry also catches exceptions defensively.
- **Cite sources.** Populate `citations` so output guardrails and evals can verify grounding.
- **Test against the mock.** Add a deterministic unit test (see `tests/test_tools.py`).
