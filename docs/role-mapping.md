# Role mapping

This repo is deliberately structured to demonstrate the requirements of an **Applied AI
Engineer — Agentic Workflows** role (building reliable, observable, evaluated LLM agents
for enterprise use). Each requirement maps to concrete, runnable code.

## Technical foundations & applied AI

| Requirement | Where it lives | Notes |
|---|---|---|
| **Production engineering** — clean, testable, observable, scalable Python | strict `mypy`, `ruff`, 56 deterministic tests, GitHub Actions CI (3.9–3.12), `observability/` | Core has **zero required dependencies**; everything heavy is an optional extra. |
| **Agentic architectures** — ReAct / Plan-and-Execute, external APIs/tools | `runtime/agent.py` (ReAct loop with step budgets), `tools/` (registry + `evidence_search`, `calculator`, `web_search`) | The same graph runs on a custom state machine or LangGraph. |
| **The LLM stack** — frontier models (Claude/GPT/Cohere), RAG, vector DBs, orchestration frameworks | `llm/` (provider-agnostic + 3 adapters), `retrieval/` (embeddings + vector store + Pinecone adapter), `runtime/state_machine.py` + `runtime/langgraph_adapter.py` | One `LLMClient` protocol; keyless mock default. |
| **Rigorous evaluation** — beyond trial-and-error; accuracy, safety, latency | `eval/` — deterministic metrics + LLM-as-judge + committed baseline + **CI regression gate** | Latency/cost reported as ungated ops metrics. |

## Reliability, safety & observability

| Requirement | Where it lives |
|---|---|
| Reliable & observable from day one | `observability/` — per-span latency, token usage, and cost, with Markdown + HTML timelines |
| Safe & auditable | `guardrails/` — PII redaction, prompt-injection blocking, citation-required & groundedness checks; every decision recorded to the trace |
| Debuggable agent behavior | full `Trace` of every LLM call, tool call, and guardrail event; `agent_run.json` + `agent_trace.md` artifacts |
| Evidence-grounded, not hallucinated | retrieval-backed citations; `GroundednessGuard` rejects citations that aren't real evidence ids |

## Leadership & communication

| Requirement | Where it lives |
|---|---|
| Translate ambiguous problems into well-framed specs | `docs/superpowers/specs/` (design spec) and `docs/superpowers/plans/` (staged implementation plan) |
| Shared frameworks & patterns | a reusable agent runtime, tool registry, guardrail and eval frameworks — not one-off features |
| Clear written communication | this doc set: [`architecture.md`](architecture.md), [`evals.md`](evals.md), [`adding-a-tool.md`](adding-a-tool.md), [`mcp.md`](mcp.md) |

## Ecosystem fluency (bonus)

An **MCP server** (`mcp_server.py`, `discovery-agents-mcp`) exposes the workflow as tools
to Claude Desktop / Claude Code — see [`mcp.md`](mcp.md).

## Try it in 30 seconds (no API key)

```bash
pip install -e ".[dev]"
discovery-agents --output outputs/demo
discovery-agents --eval
pytest -q
```
