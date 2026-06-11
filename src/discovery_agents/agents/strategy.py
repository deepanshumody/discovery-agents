"""Strategy agent — turns insights into opportunity statements.

With a real provider the model proposes opportunities from the insights; with the
keyless mock (or invalid output) it falls back to the deterministic rule set. Either
way the stated goal is always the first opportunity.
"""

from __future__ import annotations

from typing import Any

from ..agent_base import BaseAgent
from ..models import Insight, ProductBrief

OPPORTUNITIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"opportunities": {"type": "array", "items": {"type": "string"}}},
    "required": ["opportunities"],
}


class ProductStrategyAgent(BaseAgent):
    """Converts insights into strategic opportunity areas."""

    name = "ProductStrategyAgent"

    def run(self, brief: ProductBrief, insights: list[Insight]) -> list[str]:
        system, user = self._prompt(brief, insights)
        response = self._chat(
            op="strategy.opportunities",
            system=system,
            user=user,
            schema=OPPORTUNITIES_SCHEMA,
            message="map opportunities",
        )
        opportunities = self._parse(response.structured)
        source = "llm"
        if opportunities is None:
            opportunities = self._deterministic(insights)
            source = "deterministic"

        goal_line = f"Primary goal: {brief.goal}"  # always tie back to the stated goal
        if goal_line not in opportunities:
            opportunities = [goal_line, *opportunities]
        self.log(
            "Generated strategic opportunity map", opportunities=len(opportunities), source=source
        )
        return opportunities

    def _prompt(self, brief: ProductBrief, insights: list[Insight]) -> tuple[str, str]:
        system = (
            "You are a product strategist. From customer insights, propose concise, distinct "
            "strategic opportunities (one sentence each) that serve the stated goal. Return JSON."
        )
        insight_lines = "\n".join(f"- {i.title}: {i.summary}" for i in insights)
        user = (
            f"Goal: {brief.goal}\n\nInsights:\n{insight_lines}\n\n"
            "Return JSON {\"opportunities\": [...]}."
        )
        return system, user

    def _parse(self, value: dict[str, Any] | None) -> list[str] | None:
        if not value:
            return None
        raw = value.get("opportunities")
        if not isinstance(raw, list):
            return None
        out = [str(x).strip() for x in raw if isinstance(x, (str, int, float)) and str(x).strip()]
        return out or None

    def _deterministic(self, insights: list[Insight]) -> list[str]:
        opportunities = []
        if any("blank_state" in i.tags or "ideation" in i.tags for i in insights):
            opportunities.append(
                "Reduce blank-state anxiety by showing several concrete paths immediately."
            )
        if any("alignment" in i.tags or "decision" in i.tags for i in insights):
            opportunities.append(
                "Turn abstract debate into a visual decision workflow with explicit tradeoffs."
            )
        if any("handoff" in i.tags or "implementation" in i.tags for i in insights):
            opportunities.append(
                "Convert selected ideas into implementation-ready specs for engineers or coding agents."
            )
        if any("brand_fit" in i.tags or "context" in i.tags for i in insights):
            opportunities.append(
                "Use product memory so generated directions reflect brand, taste, and prior decisions."
            )
        if any("feedback" in i.tags or "learning_loop" in i.tags for i in insights):
            opportunities.append("Learn from what users select, edit, reject, and share.")
        return opportunities
