# Design: FastAPI live demo with real LLM-driven agents (Gemini)

- **Date:** 2026-06-01
- **Status:** Approved
- **Author:** Deepanshu Mody

## 1. Goal

A deployable **FastAPI** web app that runs the product-discovery pipeline with **all
reasoning agents calling a real LLM** (default **Google Gemini**), so a visitor sees a
genuinely model-driven run. Bring-your-own-key by default, a small number of free
server-key runs, and the keyless mock as a fallback. Packaged in a Dockerfile; runs
locally via uvicorn.

## 2. Non-goals
- No GPU / no torch in the web image (the agentic pipeline uses the hashing embedder).
- Not a multi-tenant product: a single in-memory rate limiter is sufficient for a demo.
- Canvas (layout) and Selection (argmax) stay deterministic — they don't need an LLM.

## 3. Components

### 3.1 Gemini provider (`llm/gemini_client.py`)
- `GeminiClient` implements `LLMClient`; lazy `from google import genai` (the `google-genai`
  SDK), `[gemini]` extra. Structured output via `response_mime_type="application/json"` +
  `response_schema`; parse with the existing `extract_json`. Capture `usage_metadata`
  (prompt/candidates tokens) and latency.
- Config: provider `"gemini"` → default model `gemini-2.0-flash`; key env `GEMINI_API_KEY`
  (fallback `GOOGLE_API_KEY`). Cost table entry for the model (Gemini Flash is cheap/free-tier).

### 3.2 Per-request API key (refactor)
- `RunConfig` gains `api_key: str | None`. The provider adapters accept an explicit
  `api_key` (falling back to env) instead of only reading global env — so a web request can
  carry its own key with **no global env mutation and no cross-request leakage**.
- `get_client(config)` passes `config.api_key` through; missing key/SDK still falls back to mock.

### 3.3 Reasoning agents → real LLM
Wire **EvidenceInsight (cluster summaries), Strategy, Critique, Handoff, Memory** to the LLM
(Ideation already is). Each follows the established pattern: build a prompt + JSON schema,
call `BaseAgent._chat`, parse the structured result into domain objects, and **fall back to
the current deterministic logic** when the model returns nothing valid (mock mode or parse
failure). Output-guardrails still apply. The **keyless mock path stays byte-identical**, so
the existing tests and the committed eval `baseline.json` remain green.

### 3.4 FastAPI app (`webapp.py`)
- `GET /` — minimal HTML page (no template engine): a form for the brief + evidence (prefilled
  from the sample or Banking77), a provider selector, an optional API-key field, and a Run
  button; renders the selected direction, all directions, the eval scorecard, and the trace
  totals (spans / tokens / cost). Links to the generated canvas.
- `POST /api/run` — JSON in `{provider?, api_key?, dataset?, brief?, evidence?}`, JSON out
  `{selected, directions, evals, trace, provider_used}`.
- `GET /healthz` — `{"status": "ok"}`.
- **Key resolution per request:** request `api_key` → BYO (no rate limit); else a server key
  (`GEMINI_API_KEY` in the server env) **under a global in-memory rate limit** → a free run;
  else → keyless mock. Keys are never logged or echoed.
- Rate limiter: a small in-memory fixed-window counter (e.g. N free server-key runs/hour).

### 3.5 Packaging
- `[web]` extra (`fastapi`, `uvicorn[standard]`). `deploy/Dockerfile.web` installs
  `.[web,gemini]`; `uvicorn discovery_agents.webapp:app` entry. Local run documented;
  deploy notes for Render/Fly. `.env` is gitignored; the server reads `GEMINI_API_KEY` from env.

## 4. Testing
- Recorded-response tests per newly-LLM agent: a scripted structured response yields
  **model-derived** output (not the deterministic fallback); invalid/empty → fallback.
- FastAPI via Starlette `TestClient`: `GET /` 200 + form present; `POST /api/run` (mock) returns
  ≥5 directions; BYO-key path selects the provider; rate-limit returns the documented behavior;
  `/healthz` ok.
- Gemini adapter: a `@pytest.mark.live` test skipped without a key; one real smoke run is done
  manually with the provided key (not in CI).
- All keyless tests + the eval gate stay green; mypy strict + ruff clean.

## 5. Security
- Per-request keys live only for the request's `RunConfig`; never written to disk, logged, or
  returned. `.env` is gitignored. The server key is an env var on the host, never committed.

## 6. Staged build
1. Gemini adapter + `[gemini]`/`[web]` extras + per-request `api_key` (config/adapters/factory) + tests; real-Gemini smoke.
2. Wire the 5 reasoning agents to the LLM (deterministic fallbacks preserved) + recorded tests.
3. FastAPI app (UI + `/api/run` + `/healthz` + BYO/server-key/rate-limit/mock) + TestClient tests.
4. Dockerfile.web + local run + docs/deploy notes; final verify; local smoke (mock + real Gemini).

## 7. Risks
- **Real Gemini API shape** can't be CI-tested → write to the SDK spec + one manual real smoke;
  graceful mock fallback means a wrong key never crashes the app.
- **Cost/abuse** on a public URL → BYO-key default + rate-limited free runs + small max-tokens.
- **More LLM agents = more latency** on a real run → fine for a demo; document expected ~10–30s.
