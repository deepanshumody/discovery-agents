"""Critique agent — scores directions with transparent criteria.

With a real provider the model returns a structured 1-5 scorecard per direction; with
the keyless mock (or invalid/incomplete output) it falls back to the deterministic
heuristic. Selection consumes the resulting weighted scores either way.
"""

from __future__ import annotations

from typing import Any

from ..agent_base import BaseAgent
from ..models import CritiqueScore, EvidenceItem, ProductDirection

_SCORE_FIELDS = (
    "customer_alignment",
    "novelty",
    "feasibility",
    "strategic_fit",
    "clarity",
    "risk_level",
)
CRITIQUES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "critiques": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "direction_id": {"type": "string"},
                    **{f: {"type": "integer"} for f in _SCORE_FIELDS},
                    "summary": {"type": "string"},
                    "recommended_next_step": {"type": "string"},
                },
                "required": ["direction_id", *_SCORE_FIELDS],
            },
        }
    },
    "required": ["critiques"],
}


class CritiqueAgent(BaseAgent):
    """Scores directions with transparent, product-specific criteria."""

    name = "CritiqueAgent"

    def run(
        self, directions: list[ProductDirection], evidence: list[EvidenceItem]
    ) -> list[CritiqueScore]:
        system, user = self._prompt(directions)
        response = self._chat(
            op="critique.score",
            system=system,
            user=user,
            schema=CRITIQUES_SCHEMA,
            message="score directions",
        )
        critiques = self._parse(response.structured, directions)
        source = "llm"
        if critiques is None:
            critiques = self._deterministic(directions, evidence)
            source = "deterministic"
        self.log("Critiqued candidate directions", critique_count=len(critiques), source=source)
        return critiques

    def _prompt(self, directions: list[ProductDirection]) -> tuple[str, str]:
        system = (
            "You are a rigorous product critic. Score EVERY direction on customer_alignment, "
            "novelty, feasibility, strategic_fit, clarity, and risk_level — each an integer 1-5 "
            "(higher risk_level = riskier). Add a one-line summary and a recommended_next_step. "
            "Return JSON with a 'critiques' array covering every direction id."
        )
        lines = "\n".join(
            f"- {d.id} {d.title}: {d.one_liner} | differentiator: {d.differentiator} | "
            f"risks: {'; '.join(d.risks)}"
            for d in directions
        )
        user = f"Directions:\n{lines}\n\nReturn JSON {{'critiques': [...]}} with one entry per id."
        return system, user

    def _parse(
        self, value: dict[str, Any] | None, directions: list[ProductDirection]
    ) -> list[CritiqueScore] | None:
        if not value:
            return None
        raw = value.get("critiques")
        if not isinstance(raw, list):
            return None
        by_id = {str(item.get("direction_id")): item for item in raw if isinstance(item, dict)}
        critiques: list[CritiqueScore] = []
        for direction in directions:  # require a critique for every direction, else fall back
            item = by_id.get(direction.id)
            if item is None or not all(_is_int(item.get(f)) for f in _SCORE_FIELDS):
                return None
            critiques.append(
                CritiqueScore(
                    direction_id=direction.id,
                    customer_alignment=_clamp(item["customer_alignment"]),
                    novelty=_clamp(item["novelty"]),
                    feasibility=_clamp(item["feasibility"]),
                    strategic_fit=_clamp(item["strategic_fit"]),
                    clarity=_clamp(item["clarity"]),
                    risk_level=_clamp(item["risk_level"]),
                    summary=str(item.get("summary") or f"Critique of {direction.title}."),
                    recommended_next_step=str(
                        item.get("recommended_next_step") or _next_step_for(direction.id)
                    ),
                )
            )
        return critiques

    def _deterministic(
        self, directions: list[ProductDirection], evidence: list[EvidenceItem]
    ) -> list[CritiqueScore]:
        evidence_ids = {item.id for item in evidence}
        critiques: list[CritiqueScore] = []
        for direction in directions:
            coverage = len(set(direction.evidence_ids) & evidence_ids)
            customer_alignment = min(5, 2 + coverage)
            novelty = (
                5
                if "not" in direction.differentiator.lower()
                or "judgment" in direction.differentiator.lower()
                else 4
            )
            feasibility = 4 if len(direction.implementation_notes) >= 3 else 3
            strategic_fit = (
                5
                if any(
                    word in direction.one_liner.lower()
                    for word in ["product", "decision", "handoff", "memory", "evaluate"]
                )
                else 4
            )
            clarity = 5 if len(direction.one_liner) < 140 else 4
            risk_level = min(5, max(2, len(direction.risks)))
            critiques.append(
                CritiqueScore(
                    direction_id=direction.id,
                    customer_alignment=customer_alignment,
                    novelty=novelty,
                    feasibility=feasibility,
                    strategic_fit=strategic_fit,
                    clarity=clarity,
                    risk_level=risk_level,
                    summary=(
                        f"{direction.title} is strong on customer evidence and strategic fit. "
                        f"Primary risk: {direction.risks[0]}"
                    ),
                    recommended_next_step=_next_step_for(direction.id),
                )
            )
        return critiques


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) or (
        isinstance(value, float) and value.is_integer()
    )


def _clamp(value: Any) -> int:
    return max(1, min(5, int(value)))


def _next_step_for(direction_id: str) -> str:
    return {
        "D1": "Prototype a canvas with 5–8 opportunity cards and test whether teams can choose a direction faster.",
        "D2": "Run the critique panel against three real customer problems and compare to founder judgment.",
        "D3": "Build a tiny memory graph over evidence, decisions, and rejected ideas; measure context recall quality.",
        "D4": "Generate a handoff packet for one selected direction and ask an engineer/coding agent to implement from it.",
        "D5": "Instrument user behavior and create an eval dashboard for selection, edit depth, and build-through.",
    }.get(direction_id, "Run a small user test and collect qualitative feedback.")
