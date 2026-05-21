"""Strategy agent — turns insights into opportunity statements."""

from __future__ import annotations

from typing import List

from ..agent_base import BaseAgent
from ..models import Insight, ProductBrief


class ProductStrategyAgent(BaseAgent):
    """Converts insights into strategic opportunity areas."""

    name = "ProductStrategyAgent"

    def run(self, brief: ProductBrief, insights: List[Insight]) -> List[str]:
        opportunities = []
        if any("blank_state" in i.tags or "ideation" in i.tags for i in insights):
            opportunities.append("Reduce blank-state anxiety by showing several concrete paths immediately.")
        if any("alignment" in i.tags or "decision" in i.tags for i in insights):
            opportunities.append("Turn abstract debate into a visual decision workflow with explicit tradeoffs.")
        if any("handoff" in i.tags or "implementation" in i.tags for i in insights):
            opportunities.append("Convert selected ideas into implementation-ready specs for engineers or coding agents.")
        if any("brand_fit" in i.tags or "context" in i.tags for i in insights):
            opportunities.append("Use product memory so generated directions reflect brand, taste, and prior decisions.")
        if any("feedback" in i.tags or "learning_loop" in i.tags for i in insights):
            opportunities.append("Learn from what users select, edit, reject, and share.")

        # Always tie back to the stated goal.
        opportunities.insert(0, f"Primary goal: {brief.goal}")
        self.log("Generated strategic opportunity map", opportunities=len(opportunities))
        return opportunities
