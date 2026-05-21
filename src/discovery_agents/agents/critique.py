"""Critique agent — scores directions with transparent criteria."""

from __future__ import annotations

from typing import List

from ..agent_base import BaseAgent
from ..models import CritiqueScore, EvidenceItem, ProductDirection


class CritiqueAgent(BaseAgent):
    """Scores directions with transparent, product-specific criteria."""

    name = "CritiqueAgent"

    def run(self, directions: List[ProductDirection], evidence: List[EvidenceItem]) -> List[CritiqueScore]:
        evidence_ids = {item.id for item in evidence}
        critiques: List[CritiqueScore] = []
        for direction in directions:
            coverage = len(set(direction.evidence_ids) & evidence_ids)
            customer_alignment = min(5, 2 + coverage)
            novelty = 5 if "not" in direction.differentiator.lower() or "judgment" in direction.differentiator.lower() else 4
            feasibility = 4 if len(direction.implementation_notes) >= 3 else 3
            strategic_fit = 5 if any(word in direction.one_liner.lower() for word in ["product", "decision", "handoff", "memory", "evaluate"]) else 4
            clarity = 5 if len(direction.one_liner) < 140 else 4
            risk_level = min(5, max(2, len(direction.risks)))
            next_step = _next_step_for(direction.id)
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
                    recommended_next_step=next_step,
                )
            )
        self.log("Critiqued candidate directions", critique_count=len(critiques))
        return critiques


def _next_step_for(direction_id: str) -> str:
    return {
        "D1": "Prototype a canvas with 5–8 opportunity cards and test whether teams can choose a direction faster.",
        "D2": "Run the critique panel against three real customer problems and compare to founder judgment.",
        "D3": "Build a tiny memory graph over evidence, decisions, and rejected ideas; measure context recall quality.",
        "D4": "Generate a handoff packet for one selected direction and ask an engineer/coding agent to implement from it.",
        "D5": "Instrument user behavior and create an eval dashboard for selection, edit depth, and build-through.",
    }.get(direction_id, "Run a small user test and collect qualitative feedback.")
