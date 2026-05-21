# discovery-agents

A multi-agent product-discovery pipeline. Given a product brief and a set of
customer evidence, it clusters the evidence into insights, generates several
candidate product directions, critiques and scores them, picks a winner, and
produces a coding-agent-ready handoff packet.

The demo runs entirely on the standard library so it can be executed anywhere.
Each agent is deliberately small and deterministic — the architecture is
designed so any single agent can later be backed by an LLM, retrieval index, or
multimodal model without changing the pipeline shape.

## Layout

```
src/discovery_agents/
  agent_base.py        # BaseAgent + AgentTrace
  models.py            # dataclasses: ProductBrief, EvidenceItem, …
  pipeline.py          # orchestrator
  render.py            # Markdown + HTML renderers
  sample_data.py       # demo brief + evidence
  cli.py               # `python -m discovery_agents.cli`
  agents/
    evidence_insight.py
    strategy.py
    ideation.py
    critique.py
    canvas.py
    selection.py
    handoff.py
    eval.py
    memory.py
tests/                 # pytest suite
docs/                  # additional documentation
outputs/               # generated artifacts (git-ignored)
```

## Install

```bash
pip install -e .[dev]
```

## Run the demo

```bash
python -m discovery_agents.cli --output outputs/demo
# or, after install:
discovery-agents --output outputs/demo
```

This writes five artifacts into `outputs/demo/`:

- `run_summary.md` — narrative summary of the run
- `coding_agent_handoff.md` — implementation-ready spec
- `canvas.html` — visual canvas of all candidate directions
- `agent_run.json` — full structured run artifact
- `agent_trace.md` — per-agent trace log

## Test

```bash
pytest
```
