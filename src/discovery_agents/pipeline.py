"""Pipeline orchestration for the Product Discovery Agent System."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .agent_base import AgentTrace
from .agents import (
    CanvasAgent,
    CritiqueAgent,
    DecisionMemoryAgent,
    EvalAgent,
    EvidenceInsightAgent,
    HandoffAgent,
    IdeationAgent,
    ProductStrategyAgent,
    SelectionAgent,
)
from .models import AgentRun, EvidenceItem, ProductBrief


class ProductDiscoveryPipeline:
    """Orchestrates a multi-agent product discovery workflow."""

    def __init__(self) -> None:
        self.trace = AgentTrace()
        self.evidence_agent = EvidenceInsightAgent(self.trace)
        self.strategy_agent = ProductStrategyAgent(self.trace)
        self.ideation_agent = IdeationAgent(self.trace)
        self.critique_agent = CritiqueAgent(self.trace)
        self.canvas_agent = CanvasAgent(self.trace)
        self.selection_agent = SelectionAgent(self.trace)
        self.handoff_agent = HandoffAgent(self.trace)
        self.eval_agent = EvalAgent(self.trace)
        self.memory_agent = DecisionMemoryAgent(self.trace)

    def run(self, brief: ProductBrief, evidence: List[EvidenceItem]) -> AgentRun:
        insights = self.evidence_agent.run(evidence)
        opportunities = self.strategy_agent.run(brief, insights)
        directions = self.ideation_agent.run(brief, insights, opportunities)
        critiques = self.critique_agent.run(directions, evidence)
        canvas_cards = self.canvas_agent.run(directions, critiques)
        selected_direction_id = self.selection_agent.run(critiques)
        selected_direction = next(d for d in directions if d.id == selected_direction_id)
        coding_spec = self.handoff_agent.run(selected_direction)
        evals = self.eval_agent.run(brief, evidence, directions, critiques, coding_spec)
        decision_log = self.memory_agent.run(selected_direction_id, directions, critiques)
        return AgentRun(
            brief=brief,
            evidence=evidence,
            insights=insights,
            directions=directions,
            critiques=critiques,
            canvas_cards=canvas_cards,
            selected_direction_id=selected_direction_id,
            coding_spec=coding_spec,
            evals=evals,
            decision_log=decision_log,
        )

    def write_outputs(self, run: AgentRun, output_dir: str | Path) -> None:
        from .render import render_canvas_html, render_handoff_markdown, render_run_markdown

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "agent_run.json").write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        (output / "run_summary.md").write_text(render_run_markdown(run), encoding="utf-8")
        (output / "coding_agent_handoff.md").write_text(render_handoff_markdown(run), encoding="utf-8")
        (output / "canvas.html").write_text(render_canvas_html(run), encoding="utf-8")
        (output / "agent_trace.md").write_text(self.trace.as_markdown(), encoding="utf-8")
