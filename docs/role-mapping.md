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

---

## ML-infrastructure role mapping

The `mlinfra/` platform ([`ml-platform.md`](ml-platform.md)) targets ML-infrastructure-at-scale
roles. Each requirement maps to runnable, tested code.

| Requirement | Where it lives |
|---|---|
| **Measured result on real data** | **Banking77 retrieval** ([`../benchmark/RESULTS.md`](../benchmark/RESULTS.md)): trained embedder beats the lexical baseline — hit@1 0.830 vs 0.769, **mAP 0.775 vs 0.503** |
| PyTorch: custom training loops, distributed training, low-level perf | `mlinfra/train/loop.py`, `mlinfra/train/distributed.py` (DDP, gloo/nccl); SupCon training in `retrieval_eval.py` |
| Reliability/continuity of large training runs | `mlinfra/train/checkpoint.py` — atomic, fsync-durable, resumable, SIGTERM-safe; resume reproduces the trajectory (tested) |
| GPU-native data I/O; Zarr/HDF5/TensorStore; multi-dim tensors | `mlinfra/store/` (Numpy/Zarr/HDF5), `mlinfra/data/loader.py` |
| I/O performance benchmarking at scale | `mlinfra/bench/io_benchmark.py` (samples/s, MB/s, p50/p95) |
| Distributed computing (Spark/Dask/Ray) | `mlinfra/curation/executors.py` (Local + Dask; Ray = future work) |
| Containerization & orchestration (Docker/K8s) | `deploy/Dockerfile`, `deploy/docker-compose.yml`, `deploy/k8s/train-job.yaml` |
| MLOps / full lifecycle / artifact tracking / monitoring | `mlinfra/tracking/` (MLflow + JSON), `mlinfra/bench/profile.py`, dedicated CI job |
| Abstractions other engineers depend on | `ArrayStore` / `Executor` / `Tracker` protocols; `TorchEmbedder` implements the existing `Embedder` |
| AI agent frameworks (a plus) | the agentic pipeline (above) consumes the trained embedder |

```bash
pip install -e ".[ml,dask,benchmark,st]"
python -m discovery_agents.mlinfra.cli benchmark --full --with-st   # the headline result
python -m discovery_agents.mlinfra.cli train --smoke               # curate -> train -> export (CPU)
python -m discovery_agents.mlinfra.cli io-bench                     # Numpy vs Zarr vs HDF5 I/O
```

Deferred to future work (documented in the spec): TensorStore backend, Ray executor, FSDP, and a
K8s serving Deployment.
