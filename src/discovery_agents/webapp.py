"""FastAPI live demo for the product-discovery pipeline.

Runs the full multi-agent workflow with all reasoning agents calling a real LLM
(default Google Gemini). Key resolution per request:

1. a key supplied in the request -> bring-your-own-key (no rate limit);
2. else a server key in the host env -> a free run, under a global in-memory rate limit;
3. else -> the keyless deterministic mock.

Keys are read from the request or the host env only; they are never logged, written
to disk, or echoed back in any response. Install with ``pip install '.[web,gemini]'``
and run ``uvicorn discovery_agents.webapp:app``.
"""

from __future__ import annotations

import html
import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .config import PROVIDER_KEY_ENV, RunConfig
from .models import AgentRun, EvidenceItem, ProductBrief
from .observability.trace import Trace
from .pipeline import ProductDiscoveryPipeline
from .webdata import DATASETS

# Providers offered in the UI; "mock" is always safe and keyless.
PROVIDERS = ["mock", "gemini", "anthropic", "openai", "cohere"]

# Defensive caps so a public endpoint can't be handed an unbounded payload.
MAX_EVIDENCE = 50
MAX_TEXT = 4000


class _RateLimiter:
    """Tiny in-memory sliding-window limiter for free server-key runs."""

    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: list[float] = []

    def allow(self, now: float) -> bool:
        cutoff = now - self.window_s
        self._hits = [t for t in self._hits if t >= cutoff]
        if len(self._hits) >= self.limit:
            return False
        self._hits.append(now)
        return True


_FREE_RUNS = int(os.environ.get("DISCOVERY_WEB_FREE_RUNS", "20"))
_LIMITER = _RateLimiter(limit=_FREE_RUNS, window_s=3600.0)

app = FastAPI(title="Discovery Agents — live demo", version="0.2.0")


# --- request/response models ------------------------------------------------


class RunRequest(BaseModel):
    provider: str = "mock"
    api_key: str | None = None
    dataset: str = "sample"
    goal: str | None = None
    brief: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] | None = None


# --- key resolution ---------------------------------------------------------


def resolve_config(provider: str, api_key: str | None) -> tuple[RunConfig, str]:
    """Pick a RunConfig and a mode label without ever mutating global env.

    Returns (config, mode) where mode is one of "byok", "server", or "mock".
    """
    provider = (provider or "mock").lower()
    if provider not in PROVIDERS or provider == "mock":
        return RunConfig(provider="mock"), "mock"
    if api_key:
        return RunConfig(provider=provider, api_key=api_key), "byok"
    env_var = PROVIDER_KEY_ENV.get(provider)
    server_key = os.environ.get(env_var) if env_var else None
    if server_key and _LIMITER.allow(time.time()):
        # Adapter reads the key from the host env; we do not copy it into the config.
        return RunConfig(provider=provider), "server"
    return RunConfig(provider="mock"), "mock"


# --- input building ---------------------------------------------------------


def _truncate(text: str) -> str:
    return text[:MAX_TEXT]


def build_inputs(req: RunRequest) -> tuple[ProductBrief, list[EvidenceItem]]:
    """Resolve brief + evidence from an explicit payload or the named dataset."""
    _, dataset_brief, dataset_evidence = DATASETS.get(req.dataset, DATASETS["sample"])

    if req.brief:
        b = req.brief
        brief = ProductBrief(
            company=str(b.get("company", dataset_brief.company)),
            product=str(b.get("product", dataset_brief.product)),
            target_user=str(b.get("target_user", dataset_brief.target_user)),
            goal=_truncate(str(b.get("goal", dataset_brief.goal))),
            constraints=[_truncate(str(c)) for c in b.get("constraints", dataset_brief.constraints)],
            strategic_themes=[str(t) for t in b.get("strategic_themes", dataset_brief.strategic_themes)],
        )
    else:
        brief = dataset_brief
    if req.goal:  # a lightweight goal override from the HTML form
        brief = ProductBrief(
            company=brief.company,
            product=brief.product,
            target_user=brief.target_user,
            goal=_truncate(req.goal),
            constraints=list(brief.constraints),
            strategic_themes=list(brief.strategic_themes),
        )

    if req.evidence:
        evidence = [
            EvidenceItem(
                id=str(e.get("id", f"E{i + 1}")),
                source=str(e.get("source", "user")),
                text=_truncate(str(e.get("text", ""))),
                user_segment=str(e.get("user_segment", "unknown")),
                severity=int(e.get("severity", 3)),
                tags=[str(t) for t in e.get("tags", [])],
            )
            for i, e in enumerate(req.evidence[:MAX_EVIDENCE])
        ]
    else:
        evidence = list(dataset_evidence)
    return brief, evidence


# --- run + shaping ----------------------------------------------------------


def _direction_dict(direction: Any, scores: dict[str, float]) -> dict[str, Any]:
    return {
        "id": direction.id,
        "title": direction.title,
        "one_liner": direction.one_liner,
        "differentiator": direction.differentiator,
        "evidence_ids": list(direction.evidence_ids),
        "score": scores.get(direction.id),
    }


def summarize_run(run: AgentRun, trace: Trace, mode: str, config: RunConfig) -> dict[str, Any]:
    scores = {c.direction_id: c.weighted_score for c in run.critiques}
    selected = next((d for d in run.directions if d.id == run.selected_direction_id), None)
    usage = trace.total_usage
    return {
        "provider_used": {
            "mode": mode,
            "provider": config.provider,
            "model": config.resolved_model(),
        },
        "selected": _direction_dict(selected, scores) if selected else None,
        "directions": [_direction_dict(d, scores) for d in run.directions],
        "evals": [
            {"metric": e.metric, "score": e.score, "explanation": e.explanation} for e in run.evals
        ],
        "decision_log": list(run.decision_log),
        "trace": {
            "spans": len(trace.spans),
            "total_tokens": usage.total_tokens,
            "cost_usd": trace.total_cost_usd,
            "latency_ms": trace.total_latency_ms,
        },
    }


def execute(req: RunRequest) -> dict[str, Any]:
    config, mode = resolve_config(req.provider, req.api_key)
    brief, evidence = build_inputs(req)
    pipeline = ProductDiscoveryPipeline(config)
    run = pipeline.run(brief, evidence)
    return summarize_run(run, pipeline.trace, mode, config)


# --- routes -----------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/run")
def api_run(req: RunRequest) -> JSONResponse:
    return JSONResponse(execute(req))


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(render_page())


@app.post("/", response_class=HTMLResponse)
async def index_run(request: Request) -> HTMLResponse:
    form = await request.form()
    req = RunRequest(
        provider=str(form.get("provider", "mock")),
        api_key=(str(form.get("api_key")) or None) if form.get("api_key") else None,
        dataset=str(form.get("dataset", "sample")),
        goal=(str(form.get("goal")) or None) if form.get("goal") else None,
    )
    result = execute(req)
    return HTMLResponse(
        render_page(
            selected_dataset=req.dataset,
            selected_provider=req.provider,
            result_html=render_result(result),
        )
    )


# --- HTML rendering ---------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value))


def render_result(result: dict[str, Any]) -> str:
    used = result["provider_used"]
    note = {
        "byok": "Ran with your API key.",
        "server": "Ran on a free server-key run (rate-limited).",
        "mock": "Ran on the keyless deterministic mock.",
    }.get(used["mode"], "")
    trace = result["trace"]
    parts = [
        "<section class='result'>",
        f"<div class='banner'>{_esc(note)} "
        f"<span class='muted'>provider={_esc(used['provider'])} · model={_esc(used['model'])} · "
        f"mode={_esc(used['mode'])}</span></div>",
    ]

    selected = result["selected"]
    if selected:
        parts.append(
            "<h2>Selected direction</h2>"
            f"<div class='card selected'><h3>{_esc(selected['title'])} "
            f"<span class='score'>score {_esc(selected['score'])}</span></h3>"
            f"<p>{_esc(selected['one_liner'])}</p>"
            f"<p class='muted'>Why it wins: {_esc(selected['differentiator'])}</p>"
            f"<p class='muted'>Evidence: {_esc(', '.join(selected['evidence_ids']))}</p></div>"
        )

    parts.append("<h2>All directions</h2><div class='grid'>")
    for d in result["directions"]:
        parts.append(
            f"<div class='card'><h3>{_esc(d['title'])} "
            f"<span class='score'>{_esc(d['score'])}</span></h3>"
            f"<p>{_esc(d['one_liner'])}</p>"
            f"<p class='muted'>Evidence: {_esc(', '.join(d['evidence_ids']))}</p></div>"
        )
    parts.append("</div>")

    parts.append("<h2>Eval scorecard</h2><table class='evals'>")
    for e in result["evals"]:
        parts.append(
            f"<tr><td>{_esc(e['metric'])}</td><td>{_esc(round(e['score'], 4))}</td>"
            f"<td class='muted'>{_esc(e['explanation'])}</td></tr>"
        )
    parts.append("</table>")

    if result["decision_log"]:
        parts.append("<h2>Decision memory</h2><ul>")
        for line in result["decision_log"]:
            parts.append(f"<li>{_esc(line)}</li>")
        parts.append("</ul>")

    cost_str = "{:.6f}".format(trace["cost_usd"])
    parts.append(
        "<h2>Trace totals</h2>"
        f"<p class='muted'>{_esc(trace['spans'])} spans · {_esc(trace['total_tokens'])} tokens · "
        f"${_esc(cost_str)} · {_esc(round(trace['latency_ms'], 1))} ms</p>"
    )
    parts.append("</section>")
    return "".join(parts)


def render_page(
    selected_dataset: str = "sample",
    selected_provider: str = "mock",
    result_html: str = "",
) -> str:
    dataset_options = "".join(
        f"<option value='{_esc(name)}'{' selected' if name == selected_dataset else ''}>"
        f"{_esc(label)}</option>"
        for name, (label, _, _) in DATASETS.items()
    )
    provider_options = "".join(
        f"<option value='{_esc(p)}'{' selected' if p == selected_provider else ''}>{_esc(p)}</option>"
        for p in PROVIDERS
    )
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Discovery Agents — live demo</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font-family: ui-sans-serif, system-ui, sans-serif; max-width: 920px; margin: 2rem auto;
        padding: 0 1rem; line-height: 1.5; }}
 h1 {{ margin-bottom: .25rem; }}
 form {{ display: grid; gap: .75rem; padding: 1rem; border: 1px solid #8884; border-radius: 12px; }}
 label {{ font-weight: 600; font-size: .9rem; }}
 input, select, textarea {{ font: inherit; padding: .5rem; border-radius: 8px;
        border: 1px solid #8886; width: 100%; box-sizing: border-box; }}
 .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }}
 button {{ font: inherit; font-weight: 600; padding: .6rem 1rem; border-radius: 8px;
        border: 0; background: #2563eb; color: #fff; cursor: pointer; }}
 .muted {{ color: #6b7280; font-size: .88rem; }}
 .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }}
 .card {{ border: 1px solid #8884; border-radius: 10px; padding: .75rem; }}
 .card.selected {{ border-color: #2563eb; }}
 .score {{ float: right; color: #2563eb; font-size: .9rem; }}
 .banner {{ background: #2563eb1a; padding: .6rem .8rem; border-radius: 8px; margin: 1rem 0; }}
 table.evals {{ width: 100%; border-collapse: collapse; }}
 table.evals td {{ border-bottom: 1px solid #8883; padding: .35rem .5rem; vertical-align: top; }}
</style></head>
<body>
 <h1>Discovery Agents</h1>
 <p class='muted'>A multi-agent product-discovery workflow: evidence → insights → strategy →
   ideation (RAG) → critique → selection → coding handoff, with evals, guardrails, and tracing.
   Pick a dataset and provider, then run a genuinely model-driven pass.</p>
 <form method='post' action='/'>
   <div class='row'>
     <div><label>Dataset</label><select name='dataset'>{dataset_options}</select></div>
     <div><label>Provider</label><select name='provider'>{provider_options}</select></div>
   </div>
   <div>
     <label>API key <span class='muted'>(optional — your key is used for this request only,
       never stored or logged)</span></label>
     <input type='password' name='api_key' autocomplete='off'
       placeholder='Leave blank to use a free server run or the keyless mock'>
   </div>
   <div>
     <label>Goal override <span class='muted'>(optional)</span></label>
     <input type='text' name='goal' placeholder='Leave blank to use the dataset goal'>
   </div>
   <div><button type='submit'>Run discovery</button></div>
 </form>
 {result_html}
 <p class='muted' style='margin-top:2rem'>JSON API: <code>POST /api/run</code> ·
   Health: <code>GET /healthz</code></p>
</body></html>"""
