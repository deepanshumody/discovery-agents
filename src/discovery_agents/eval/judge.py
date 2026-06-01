"""LLM-as-judge scoring with a deterministic keyless fallback.

With a real provider, the judge asks the model to rate the selected direction on
faithfulness, relevance, and helpfulness (0..1) as JSON. With the mock (or if the
model returns invalid JSON), it computes transparent, deterministic heuristics so
CI stays stable. Either way it records a trace span.
"""

from __future__ import annotations

from typing import Any

from ..llm.base import LLMClient
from ..llm.types import Message
from ..models import AgentRun, ProductDirection
from ..observability.trace import Trace
from ..retrieval.embeddings import HashingEmbedder
from ..retrieval.vector_store import cosine_similarity

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number"},
        "relevance": {"type": "number"},
        "helpfulness": {"type": "number"},
    },
    "required": ["faithfulness", "relevance", "helpfulness"],
}

_METRICS = ("faithfulness", "relevance", "helpfulness")


class LLMJudge:
    name = "LLMJudge"

    def __init__(self, llm: LLMClient, trace: Trace | None = None) -> None:
        self.llm = llm
        self.trace = trace

    def score(self, run: AgentRun) -> dict[str, float]:
        selected = self._selected_direction(run)
        if selected is None:
            return dict.fromkeys(_METRICS, 0.0)

        response = self.llm.chat(
            [
                Message(role="system", content=self._system()),
                Message(role="user", content=self._prompt(run, selected)),
            ],
            response_format=JUDGE_SCHEMA,
        )
        if self.trace is not None:
            self.trace.record_llm(self.name, "judge.score", response, message="LLM-as-judge")

        if response.structured and all(k in response.structured for k in _METRICS):
            return {m: _clamp(response.structured[m]) for m in _METRICS}
        return self._heuristic(run, selected)

    # -- helpers -------------------------------------------------------------
    def _selected_direction(self, run: AgentRun) -> ProductDirection | None:
        return next((d for d in run.directions if d.id == run.selected_direction_id), None)

    def _system(self) -> str:
        return (
            "You are a strict evaluation judge. Rate the selected product direction on "
            "faithfulness (claims supported by cited evidence), relevance (addresses the goal), "
            "and helpfulness (clear, actionable). Return JSON with three numbers in [0,1]."
        )

    def _prompt(self, run: AgentRun, selected: ProductDirection) -> str:
        evidence_by_id = {e.id: e.text for e in run.evidence}
        cited = "\n".join(
            f"- {eid}: {evidence_by_id.get(eid, '(missing)')}" for eid in selected.evidence_ids
        )
        return (
            f"Goal: {run.brief.goal}\n\n"
            f"Selected direction: {selected.title}\n"
            f"One-liner: {selected.one_liner}\n"
            f"Core loop: {selected.core_loop}\n"
            f"Differentiator: {selected.differentiator}\n\n"
            f"Cited evidence:\n{cited}\n"
        )

    def _heuristic(self, run: AgentRun, selected: ProductDirection) -> dict[str, float]:
        valid = {e.id for e in run.evidence}
        cited = selected.evidence_ids
        faithfulness = sum(1 for c in cited if c in valid) / len(cited) if cited else 0.0

        # Relevance via embedding cosine similarity between the goal/themes and the
        # direction text — a semantic, deterministic proxy that reuses the RAG embedder.
        embedder = HashingEmbedder()
        goal_text = f"{run.brief.goal} {' '.join(run.brief.strategic_themes)}"
        direction_text = (
            f"{selected.title} {selected.one_liner} {selected.core_loop} {selected.differentiator}"
        )
        relevance = cosine_similarity(embedder.embed(goal_text), embedder.embed(direction_text))

        parts = [
            selected.core_loop,
            selected.differentiator,
            selected.risks,
            selected.implementation_notes,
        ]
        helpfulness = sum(1 for p in parts if p) / len(parts)

        return {
            "faithfulness": round(_clamp(faithfulness), 4),
            "relevance": round(_clamp(relevance), 4),
            "helpfulness": round(_clamp(helpfulness), 4),
        }


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
