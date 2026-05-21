"""Canvas agent — maps directions to visual canvas cards."""

from __future__ import annotations

from typing import List

from ..agent_base import BaseAgent
from ..models import CanvasCard, CritiqueScore, ProductDirection


class CanvasAgent(BaseAgent):
    """Maps directions into a simple visual canvas structure."""

    name = "CanvasAgent"

    def run(self, directions: List[ProductDirection], critiques: List[CritiqueScore]) -> List[CanvasCard]:
        score_by_id = {c.direction_id: c.weighted_score for c in critiques}
        cards = []
        for index, direction in enumerate(directions, start=1):
            body = (
                f"{direction.one_liner}\n\n"
                f"Core loop: {direction.core_loop}\n\n"
                f"Differentiator: {direction.differentiator}"
            )
            cards.append(
                CanvasCard(
                    id=direction.id,
                    title=direction.title,
                    column=direction.canvas_column,
                    row=direction.canvas_row or index,
                    body=body,
                    evidence=direction.evidence_ids,
                    score=score_by_id.get(direction.id, 0.0),
                    tags=[direction.canvas_column.lower(), "agent-generated"],
                )
            )
        self.log("Mapped directions to canvas cards", card_count=len(cards))
        return cards
