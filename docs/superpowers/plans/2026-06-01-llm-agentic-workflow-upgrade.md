# LLM-Powered Agentic Workflow Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, per-stage checkpoints) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the deterministic `discovery-agents` pipeline into a production-grade, LLM-powered, evaluated, observable multi-agent workflow with RAG, ReAct tool use, guardrails, a custom runtime (+ optional LangGraph adapter), and an MCP server — keyless-by-default, real-on-demand.

**Architecture:** Provider-agnostic `LLMClient` (default Claude, deterministic `MockLLMClient` fallback) consumed by the existing 9 agents; a typed state-machine runtime drives them; capabilities (retrieval/RAG, tools, guardrails, observability) are pluggable; an eval harness scores runs against a committed baseline and gates CI. Spec: `docs/superpowers/specs/2026-06-01-llm-agentic-workflow-upgrade-design.md`.

**Tech Stack:** Python 3.9+ (stdlib core, zero required deps), optional extras: `anthropic`, `cohere`, `openai`, `langgraph`, vector-db clients, `mcp`; dev: `pytest`, `ruff`, `mypy`; CI: GitHub Actions.

**Working branch:** `feat/llm-agentic-upgrade` (default branch `main`).

**Global invariants (true after every stage):**
- `pytest` passes with **no API keys** set.
- `ruff check .` and `mypy src` pass.
- `python -m discovery_agents.cli --output outputs/demo` produces all artifacts.

---

## Stage 0 — Hygiene & enterprise reskin

**Files:**
- Modify: `pyproject.toml` (extras, ruff/mypy config)
- Create: `.github/workflows/ci.yml`
- Create: `ruff.toml` (or `[tool.ruff]` in pyproject)
- Modify: `src/discovery_agents/sample_data.py` (enterprise reskin, remove "Remy")
- Modify: `README.md` (interim badge/quickstart note; full rewrite in Stage 8)
- Create: `LICENSE` (MIT), `CONTRIBUTING.md`
- Test: existing `tests/test_pipeline.py` stays green

- [ ] **0.1** Add `[project.optional-dependencies]` extras (`anthropic`, `cohere`, `openai`, `langgraph`, `vectordb`, `mcp`, `dev`) and `[tool.ruff]` + `[tool.mypy]` config to `pyproject.toml`.
- [ ] **0.2** Add `.github/workflows/ci.yml`: matrix py3.9–3.12 → `ruff check` → `mypy src` → `pytest -q` → eval-smoke (added in Stage 5; until then a no-op step).
- [ ] **0.3** Reskin `sample_data.py`: enterprise scenario (e.g. a B2B platform team), remove the "Remy-style" docstring and any company-specific naming; keep evidence tags so downstream agents still cluster.
- [ ] **0.4** Add `LICENSE` (MIT) and `CONTRIBUTING.md`.
- [ ] **0.5** Run `ruff check . && mypy src && pytest -q`; fix lint/type findings in existing code. Commit: `chore: tooling, CI, MIT license, enterprise sample data`.

**Verify:** CI config valid; `pytest` green; demo runs.

---

## Stage 1 — LLM adapter layer + observability

**Files:**
- Create: `src/discovery_agents/llm/{__init__,types,base,mock,factory}.py`
- Create: `src/discovery_agents/llm/{anthropic_client,cohere_client,openai_client}.py`
- Create: `src/discovery_agents/config.py`
- Modify: `src/discovery_agents/observability/trace.py` (move/upgrade from `agent_base.AgentTrace`)
- Create: `src/discovery_agents/observability/{__init__,cost,report}.py`
- Modify: `src/discovery_agents/agents/ideation.py` (first LLM-backed agent, mock parity)
- Test: `tests/test_llm_mock.py`, `tests/test_observability.py`

**Key contracts (write these first as failing tests, then implement):**

```python
# llm/types.py
@dataclass
class Message: role: str; content: str
@dataclass
class TokenUsage: input_tokens: int = 0; output_tokens: int = 0
@dataclass
class ToolSpec: name: str; description: str; parameters: dict
@dataclass
class ToolInvocation: name: str; arguments: dict
@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    structured: dict | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    model: str = "mock"
```

```python
# llm/base.py
class LLMClient(Protocol):
    model: str
    def chat(self, messages: list[Message], *, tools: list[ToolSpec] | None = None,
             response_format: dict | None = None) -> LLMResponse: ...
```

- [ ] **1.1** Test: `MockLLMClient.chat(...)` returns a deterministic `LLMResponse` (same input → same output), populates `usage` and `latency_ms=0.0`, and honors `response_format` by returning valid `structured` JSON. Implement `mock.py` (keyed canned responses by a stable hash of role+intent; structured mode returns a schema-shaped dict).
- [ ] **1.2** Test: `get_client()` returns `MockLLMClient` when no provider env/key is set, and warns+falls back to mock when a provider is requested but its key/SDK is missing. Implement `config.py` (`RunConfig`) + `factory.py`.
- [ ] **1.3** Implement real adapters (`anthropic_client.py` default, `cohere_client.py`, `openai_client.py`) behind the same protocol; import SDKs lazily so core stays dependency-free. Mark live tests `@pytest.mark.live` (skipped without keys).
- [ ] **1.4** Test: `TraceSpan` records `agent, op, latency_ms, usage, cost_usd, tool_calls`; `Trace.as_markdown()`/`as_html()` render spans. Implement `observability/trace.py` (supersedes `agent_base.AgentTrace`, keep a back-compat shim) + `cost.py` (price table) + `report.py`.
- [ ] **1.5** Wire `IdeationAgent` to call the LLM (structured output → `ProductDirection[]`), grounded in insights/opportunities; with `MockLLMClient` it reproduces today's five directions so `test_pipeline.py` stays green. Emit a span.
- [ ] **1.6** Run `ruff && mypy && pytest`; commit: `feat: provider-agnostic LLM layer + observability spans`.

**Verify:** keyless run unchanged in output; `agent_run.json` now carries token/cost/latency spans.

---

## Stage 2 — Retrieval / RAG

**Files:**
- Create: `src/discovery_agents/retrieval/{__init__,embeddings,vector_store,index}.py`
- Create: `src/discovery_agents/tools/{__init__,base,evidence_search}.py`
- Modify: `agents/evidence_insight.py`, `agents/ideation.py` (ground via retrieval)
- Test: `tests/test_retrieval.py`, `tests/test_tools.py`

- [ ] **2.1** Test: `HashingEmbedder.embed("text")` returns a fixed-dim deterministic vector; identical text → identical vector; cosine(self,self)=1. Implement `embeddings.py` (`Embedder` protocol + `HashingEmbedder`; lazy Cohere/OpenAI embedders).
- [ ] **2.2** Test: `InMemoryVectorStore.upsert(chunks)` then `.search(qvec,k)` returns top-k by cosine, descending. Implement `vector_store.py` (`VectorStore` protocol + in-memory; Pinecone/Weaviate adapter stubs that raise a clear "configure X" error until creds present).
- [ ] **2.3** Test: `EvidenceIndex.search("query", k=3)` returns evidence chunks with stable citation ids. Implement `index.py` (chunk→embed→upsert→search).
- [ ] **2.4** Test: `evidence_search` tool conforms to the `Tool` protocol (name, json-schema params, `run`) and returns cited results. Implement `tools/base.py` (`Tool` + `ToolRegistry`) and `tools/evidence_search.py`.
- [ ] **2.5** Ground `EvidenceInsightAgent` and `IdeationAgent` in retrieval (retrieved evidence ids flow into citations). Keep mock-mode output deterministic.
- [ ] **2.6** `ruff && mypy && pytest`; commit: `feat: RAG retrieval (embeddings + vector store) and evidence_search tool`.

---

## Stage 3 — Runtime: typed state machine + ReAct loop (+ LangGraph adapter)

**Files:**
- Create: `src/discovery_agents/runtime/{__init__,state_machine,agent,graph,langgraph_adapter}.py`
- Modify: `src/discovery_agents/pipeline.py` (run via the state machine)
- Test: `tests/test_runtime_react.py`, `tests/test_state_machine.py`

- [ ] **3.1** Test: `StateMachine` runs nodes in dependency order, threads a typed shared `State`, and emits one span per node. Implement `state_machine.py`.
- [ ] **3.2** Test: `LLMAgent.run(goal, tools, max_steps)` performs a ReAct loop — model proposes a tool call → tool runs → observation appended → repeats until a final answer or `max_steps`; stops on budget; every step emits a span. With the mock, a scripted tool call then a final answer. Implement `agent.py`.
- [ ] **3.3** Define the discovery graph in `graph.py` (nodes = the 9 agents, edges = data deps from `pipeline.py:run`).
- [ ] **3.4** Port `ProductDiscoveryPipeline.run` to execute the graph via `StateMachine`; identical artifacts in mock mode (`test_pipeline.py` green).
- [ ] **3.5** Test (skipped if `langgraph` absent): `langgraph_adapter.build(graph)` runs the same graph and produces equivalent results. Implement the adapter behind a lazy import.
- [ ] **3.6** `ruff && mypy && pytest`; commit: `feat: typed runtime + ReAct agent loop + optional LangGraph adapter`.

---

## Stage 4 — Guardrails

**Files:**
- Create: `src/discovery_agents/guardrails/{__init__,base,input_guards,output_guards,pipeline}.py`
- Modify: `src/discovery_agents/pipeline.py` (wrap inputs/outputs)
- Test: `tests/test_guardrails.py`

- [ ] **4.1** Test: `PiiRedactionGuard` redacts emails/phones; `PromptInjectionGuard` flags "ignore previous instructions"; each returns `GuardrailResult(passed, severity, reason, redacted)`. Implement `base.py` + `input_guards.py`.
- [ ] **4.2** Test: `CitationRequiredGuard` fails a direction with no evidence ids; `GroundednessGuard` flags claims unsupported by cited evidence; `SchemaGuard` validates structured output. Implement `output_guards.py`.
- [ ] **4.3** Test: `GuardrailPipeline.run(...)` executes guards, logs trace events, blocks on `severity>=block_threshold`. Implement `pipeline.py`.
- [ ] **4.4** Wire input guards on brief/evidence and output guards on directions/handoff in the pipeline. Keep mock run green (sample data passes guards).
- [ ] **4.5** `ruff && mypy && pytest`; commit: `feat: input/output guardrails wired into the pipeline`.

---

## Stage 5 — Rigorous eval harness + regression gate

**Files:**
- Create: `src/discovery_agents/eval/{__init__,cases,metrics,judge,harness}.py`
- Create: `src/discovery_agents/eval/baseline.json`
- Modify: `.github/workflows/ci.yml` (eval-smoke runs the harness)
- Modify: `cli.py` (`--eval`)
- Test: `tests/test_eval_harness.py`

- [ ] **5.1** Test: deterministic metrics — citation precision/recall, evidence coverage, distinctiveness, handoff completeness, guardrail-pass, latency, cost — compute expected values on a fixture run. Implement `metrics.py`.
- [ ] **5.2** Test: `LLMJudge.score(run)` returns faithfulness/relevance/helpfulness; with the mock the scores are deterministic. Implement `judge.py`.
- [ ] **5.3** Test: golden `EvalCase`s in `cases.py`; `harness.run()` aggregates metrics+judge into a scorecard. Implement `cases.py` + `harness.py`.
- [ ] **5.4** Test: `harness.check_regression(baseline)` fails when a metric drops beyond tolerance; passes on the committed baseline. Generate & commit `baseline.json` from a mock run.
- [ ] **5.5** Add `--eval` to the CLI and the eval-smoke CI step (`python -m discovery_agents.cli --eval` → assert no regression).
- [ ] **5.6** `ruff && mypy && pytest`; commit: `feat: eval harness (metrics + LLM-judge) with CI regression gate`.

---

## Stage 6 — MCP server

**Files:**
- Create: `src/discovery_agents/mcp_server.py`
- Create: `docs/mcp.md`
- Modify: `pyproject.toml` (`[project.scripts]` mcp entry; `[mcp]` extra)
- Test: `tests/test_mcp_server.py`

- [ ] **6.1** Test: the server registers tools `discovery.run`, `evidence.search`, `eval.run` with valid JSON schemas (introspect the handler registry without needing a live stdio client).
- [ ] **6.2** Test: invoking `discovery.run` with sample input returns a structured run summary; `evidence.search` returns cited chunks; `eval.run` returns a scorecard — all keyless via mock.
- [ ] **6.3** Implement `mcp_server.py` using the `mcp` SDK (lazy import; clear error if extra missing). Add `discovery-agents-mcp` script entry.
- [ ] **6.4** Write `docs/mcp.md`: how to register the server in Claude Desktop / Claude Code, example tool calls.
- [ ] **6.5** `ruff && mypy && pytest`; commit: `feat: MCP server exposing discovery, evidence search, and eval`.

---

## Stage 7 — Tools & end-to-end ReAct demo

**Files:**
- Create: `src/discovery_agents/tools/{calculator,web_search}.py`
- Modify: `agents/ideation.py` (use tools in the ReAct loop), `cli.py` (`--use-rag`, `--use-langgraph`)
- Test: `tests/test_tools.py` (extend), `tests/test_react_e2e.py`

- [ ] **7.1** Test: `calculator.run(expression="2+2*3")==7` and rejects unsafe input (no `eval` of arbitrary code; use a safe AST evaluator). Implement `calculator.py`.
- [ ] **7.2** Test: `web_search.run(query)` returns deterministic fixture results in mock mode. Implement `web_search.py` (lazy real backend).
- [ ] **7.3** Test: an end-to-end ReAct run where Ideation calls `evidence_search` + `calculator` then produces a grounded direction (mock-scripted). 
- [ ] **7.4** Expose `--use-rag` / `--use-langgraph` flags; document tool registry in `docs/adding-a-tool.md`.
- [ ] **7.5** `ruff && mypy && pytest`; commit: `feat: calculator + web_search tools and end-to-end ReAct demo`.

---

## Stage 8 — Docs & final polish

**Files:**
- Rewrite: `README.md`
- Create: `docs/{architecture,evals,role-mapping}.md`
- Modify: `render.py` (trace timeline + eval dashboard in HTML)
- Verify: full suite + demo + (optional) one live run if a key is available

- [ ] **8.1** Rewrite `README.md`: hero + Mermaid architecture diagram, 30-second keyless quickstart, real-LLM mode, RAG/tools, eval harness, guardrails, observability, MCP, and a JD→repo mapping summary with badges.
- [ ] **8.2** Write `docs/architecture.md` (diagrams + module responsibilities), `docs/evals.md` (metrics + judge + regression), `docs/role-mapping.md` (full JD table).
- [ ] **8.3** Extend `render.py` so `canvas.html` (or a new `report.html`) shows the trace timeline + eval dashboard.
- [ ] **8.4** Final: `ruff && mypy && pytest`; run the demo; update `baseline.json` if metrics legitimately changed; commit: `docs: README rewrite, architecture/eval/role-mapping docs, HTML reports`.
- [ ] **8.5** Open PR `feat/llm-agentic-upgrade` → `main` (or fast-forward merge) per user preference.

---

## Self-review (plan vs. spec)

- **Spec coverage:** LLM layer (S1) ✓, keyless mock (S1) ✓, RAG/vector (S2) ✓, ReAct/tools (S3,S7) ✓, runtime+LangGraph (S3) ✓, guardrails (S4) ✓, eval harness+regression (S5) ✓, observability (S1,S8) ✓, MCP (S6) ✓, enterprise reskin (S0) ✓, CI/typing/tests (S0–S8) ✓, docs+role-mapping (S8) ✓. No spec section unmapped.
- **Placeholder scan:** Stage tasks name exact files, tests, and commit messages; contracts shown for the foundational types. Per-step full source is produced during execution (inline executor has full repo context) — interfaces and test behaviors are specified so each task is unambiguous.
- **Type consistency:** `LLMClient.chat`, `LLMResponse`, `Tool`, `VectorStore`, `GuardrailResult`, `TraceSpan` names are used identically across stages.
