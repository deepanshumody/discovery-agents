"""Deterministic evaluation metrics computed from a run and its trace.

Quality metrics are deterministic under the mock provider, so they can gate CI.
Ops metrics (latency, cost, tokens) are reported but not gated, because they vary
with the provider and machine.
"""

from __future__ import annotations

import itertools

from ..agents._utils import tokenize
from ..models import AgentRun
from ..observability.trace import Trace


def citation_precision(run: AgentRun) -> float:
    """Fraction of citations across directions that point to real evidence."""
    valid = {item.id for item in run.evidence}
    cited = [eid for d in run.directions for eid in d.evidence_ids]
    if not cited:
        return 0.0
    return round(sum(1 for c in cited if c in valid) / len(cited), 4)


def evidence_coverage(run: AgentRun) -> float:
    """Fraction of evidence items cited by at least one direction."""
    valid = {item.id for item in run.evidence}
    cited = {eid for d in run.directions for eid in d.evidence_ids}
    return round(len(cited & valid) / max(1, len(valid)), 4)


def direction_count_score(run: AgentRun) -> float:
    return round(min(1.0, len(run.directions) / 5), 4)


def distinctiveness(run: AgentRun) -> float:
    """1 - mean pairwise Jaccard similarity of direction text."""
    if len(run.directions) < 2:
        return 0.0
    token_sets = [
        set(tokenize(f"{d.title} {d.one_liner} {d.differentiator}")) for d in run.directions
    ]
    sims = [len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(token_sets, 2)]
    return round(max(0.0, 1.0 - sum(sims) / len(sims)), 4)


def handoff_completeness(run: AgentRun) -> float:
    spec = run.coding_spec
    if spec is None:
        return 0.0
    sections = [
        spec.user_story,
        spec.functional_requirements,
        spec.non_functional_requirements,
        spec.data_contract,
        spec.acceptance_criteria,
        spec.analytics_events,
        spec.open_questions,
    ]
    return round(sum(1 for s in sections if s) / len(sections), 4)


def selection_validity(run: AgentRun) -> float:
    ids = {d.id for d in run.directions}
    return 1.0 if run.selected_direction_id in ids else 0.0


def guardrail_pass_rate(trace: Trace) -> float:
    """Fraction of guardrail checks (across guardrail spans) that passed."""
    events = [e for span in trace.spans if span.op == "guardrail" for e in span.guardrail_events]
    if not events:
        return 1.0
    return round(sum(1 for e in events if e.get("passed")) / len(events), 4)


def quality_metrics(run: AgentRun, trace: Trace) -> dict[str, float]:
    """Deterministic, CI-gateable quality metrics."""
    return {
        "citation_precision": citation_precision(run),
        "evidence_coverage": evidence_coverage(run),
        "direction_count": direction_count_score(run),
        "distinctiveness": distinctiveness(run),
        "handoff_completeness": handoff_completeness(run),
        "selection_validity": selection_validity(run),
        "guardrail_pass_rate": guardrail_pass_rate(trace),
    }


def ops_metrics(trace: Trace) -> dict[str, float]:
    """Operational metrics: reported, not gated (provider/machine dependent)."""
    usage = trace.total_usage
    return {
        "total_tokens": float(usage.total_tokens),
        "cost_usd": trace.total_cost_usd,
        "latency_ms": trace.total_latency_ms,
    }
