# Live demo (FastAPI)

A small FastAPI app (`discovery_agents.webapp:app`) runs the full multi-agent
product-discovery pipeline behind a web form and a JSON API. The reasoning agents call a
real LLM (default **Google Gemini**); Canvas layout and Selection stay deterministic.
With no key it runs the keyless mock, so the app always works.

## Run locally

```bash
pip install -e ".[web,gemini]"
uvicorn discovery_agents.webapp:app --reload        # http://localhost:8000
```

Optional — enable free server-side runs on a real model:

```bash
export GEMINI_API_KEY=...        # host env only; never commit it (.env is gitignored)
uvicorn discovery_agents.webapp:app --reload
```

Or with the Makefile: `make -f deploy/Makefile web-run` (set `PORT` to override 8000).

## Endpoints

| Route          | Method | Purpose |
|----------------|--------|---------|
| `/`            | GET    | HTML form: dataset, provider, optional API key, goal override. |
| `/`            | POST   | Runs the form and renders results inline. |
| `/api/run`     | POST   | JSON in/out (below). |
| `/healthz`     | GET    | `{"status": "ok"}` liveness probe. |

### `POST /api/run`

Request (all fields optional):

```json
{
  "provider": "gemini",
  "api_key": "your-key",
  "dataset": "banking",
  "goal": "Deflect repetitive contacts without hurting trust",
  "brief": { "company": "...", "goal": "..." },
  "evidence": [ { "id": "E1", "text": "...", "tags": ["blank_state"] } ]
}
```

`brief`/`evidence` override the named `dataset` when supplied. Response:

```json
{
  "provider_used": { "mode": "byok", "provider": "gemini", "model": "gemini-2.0-flash" },
  "selected": { "id": "D2", "title": "...", "score": 4.1, "evidence_ids": ["E1"] },
  "directions": [ { "id": "D1", "title": "...", "score": 3.8, "evidence_ids": ["..."] } ],
  "evals": [ { "metric": "citation_precision", "score": 1.0, "explanation": "..." } ],
  "decision_log": ["..."],
  "trace": { "spans": 18, "total_tokens": 5234, "cost_usd": 0.0021, "latency_ms": 14200 }
}
```

## Key resolution (per request)

1. **Request key** → bring-your-own-key, no rate limit (`mode: "byok"`).
2. **Host server key** (`GEMINI_API_KEY` etc. in the server env) → a free run under a global
   in-memory sliding-window limit (`mode: "server"`). Default 20 runs/hour; override with
   `DISCOVERY_WEB_FREE_RUNS`.
3. **Otherwise** → the keyless deterministic mock (`mode: "mock"`).

Keys live only inside the request's `RunConfig` (or are read from the host env by the
adapter). They are never written to disk, logged, or echoed in any response. `.env` is
gitignored; a server key is a host env var, never committed.

A real Gemini run with all reasoning agents takes roughly **10–30s** (several model calls).
A wrong or rate-limited key never crashes a run — each agent degrades to its deterministic
fallback.

## Container

```bash
docker build -f deploy/Dockerfile.web -t discovery-agents-web .
docker run --rm -p 8000:8000 discovery-agents-web                          # keyless mock
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=$GEMINI_API_KEY discovery-agents-web
```

The image is CPU-only and torch-free (the pipeline uses the hashing embedder), so it stays
small. It honors `$PORT` (set by most hosts).

## Deploy notes

The app is a standard ASGI app; any host that runs `uvicorn` works.

- **Render** — New → Web Service from the repo. Runtime: Docker, Dockerfile path
  `deploy/Dockerfile.web`. Add `GEMINI_API_KEY` as a secret env var. Render sets `$PORT`.
- **Fly.io** — `fly launch --dockerfile deploy/Dockerfile.web`; `fly secrets set
  GEMINI_API_KEY=...`. Set the internal port to 8000 in `fly.toml`.
- **Cloud Run** — `gcloud run deploy --source .` (or build the image and push), set
  `GEMINI_API_KEY` as a secret; Cloud Run injects `$PORT`.

For a public URL, leave the default keyless/BYO behavior on and keep `DISCOVERY_WEB_FREE_RUNS`
modest so a host key can't be drained. The in-memory limiter is per-process — fine for a
single-instance demo; a multi-instance deployment would need a shared store (out of scope).
