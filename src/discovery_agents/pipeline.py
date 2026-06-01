"""Pipeline orchestration for the Product Discovery Agent System."""

from __future__ import annotations

import json
from pathlib import Path

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
from .config import RunConfig
from .llm import get_client
from .models import AgentRun, EvidenceItem, ProductBrief
from .observability.trace import Trace
from .retrieval.index import EvidenceIndex
from .tools import EvidenceSearchTool, ToolRegistry


class ProductDiscoveryPipeline:
    """Orchestrates a multi-agent product discovery workflow.

    Construct with a `RunConfig` to choose the provider/model; with no config it
    resolves from the environment and defaults to the keyless mock client.
    """

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config or RunConfig.from_env()
        self.trace = Trace()
        self.llm = get_client(self.config)
        self.evidence_agent = EvidenceInsightAgent(self.trace, self.llm)
        self.strategy_agent = ProductStrategyAgent(self.trace, self.llm)
        self.ideation_agent = IdeationAgent(self.trace, self.llm)
        self.critique_agent = CritiqueAgent(self.trace, self.llm)
        self.canvas_agent = CanvasAgent(self.trace, self.llm)
        self.selection_agent = SelectionAgent(self.trace, self.llm)
        self.handoff_agent = HandoffAgent(self.trace, self.llm)
        self.eval_agent = EvalAgent(self.trace, self.llm)
        self.memory_agent = DecisionMemoryAgent(self.trace, self.llm)

    def run(self, brief: ProductBrief, evidence: list[EvidenceItem]) -> AgentRun:
        # Build the RAG index + tool registry once per run, then thread them in.
        self.index = EvidenceIndex.from_evidence(evidence)
        self.tools = ToolRegistry([EvidenceSearchTool(self.index)])

        insights = self.evidence_agent.run(evidence, index=self.index)
        opportunities = self.strategy_agent.run(brief, insights)
        directions = self.ideation_agent.run(brief, insights, opportunities, index=self.index)
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
        (output / "agent_run.json").write_text(
            json.dumps(run.to_dict(), indent=2), encoding="utf-8"
        )
        (output / "run_summary.md").write_text(render_run_markdown(run), encoding="utf-8")
        (output / "coding_agent_handoff.md").write_text(
            render_handoff_markdown(run), encoding="utf-8"
        )
        (output / "canvas.html").write_text(render_canvas_html(run), encoding="utf-8")
        (output / "agent_trace.md").write_text(self.trace.as_markdown(), encoding="utf-8")
