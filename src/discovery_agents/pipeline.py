"""Pipeline orchestration for the Product Discovery Agent System."""

from __future__ import annotations

import json
import logging
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
from .guardrails import (
    CitationRequiredGuard,
    GroundednessGuard,
    GuardrailInput,
    GuardrailPipeline,
    PiiRedactionGuard,
    PromptInjectionGuard,
)
from .llm import get_client
from .models import AgentRun, EvidenceItem, ProductBrief
from .observability.trace import Trace
from .retrieval.index import EvidenceIndex
from .runtime.graph import build_discovery_graph
from .runtime.state_machine import State, StateMachine
from .tools import CalculatorTool, EvidenceSearchTool, ToolRegistry, WebSearchTool

logger = logging.getLogger("discovery_agents.pipeline")


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
        self.tools = ToolRegistry(
            [EvidenceSearchTool(self.index), CalculatorTool(), WebSearchTool()]
        )

        graph = build_discovery_graph(
            trace=self.trace,
            evidence_agent=self.evidence_agent,
            strategy_agent=self.strategy_agent,
            ideation_agent=self.ideation_agent,
            critique_agent=self.critique_agent,
            canvas_agent=self.canvas_agent,
            selection_agent=self.selection_agent,
            handoff_agent=self.handoff_agent,
            eval_agent=self.eval_agent,
            memory_agent=self.memory_agent,
        )
        self._guard_inputs(brief, evidence)

        initial: State = {"brief": brief, "evidence": evidence, "index": self.index}
        state = self._execute(graph, initial)

        self._guard_outputs(evidence, state)

        return AgentRun(
            brief=brief,
            evidence=evidence,
            insights=state["insights"],
            directions=state["directions"],
            critiques=state["critiques"],
            canvas_cards=state["canvas_cards"],
            selected_direction_id=state["selected_direction_id"],
            coding_spec=state["coding_spec"],
            evals=state["evals"],
            decision_log=state["decision_log"],
        )

    def _execute(self, graph: StateMachine, initial: State) -> State:
        """Run the graph via the custom state machine, or LangGraph if requested."""
        if self.config.use_langgraph:
            try:
                from .runtime.langgraph_adapter import run_via_langgraph

                return run_via_langgraph(graph, initial)
            except ImportError as exc:
                logger.warning("LangGraph unavailable (%s); using the built-in state machine.", exc)
        return graph.run(initial)

    def _guard_inputs(self, brief: ProductBrief, evidence: list[EvidenceItem]) -> None:
        """Screen brief + evidence for PII and prompt injection (records spans)."""
        guards = GuardrailPipeline([PiiRedactionGuard(), PromptInjectionGuard()], trace=self.trace)
        text = " ".join([brief.goal, *brief.constraints, *(item.text for item in evidence)])
        results = guards.run(GuardrailInput(text=text), stage="input")
        if guards.is_blocked(results):
            blocked = [r.name for r in results if not r.passed]
            logger.warning("Input guardrails flagged a blocking issue: %s", blocked)

    def _guard_outputs(self, evidence: list[EvidenceItem], state: State) -> None:
        """Verify each generated direction cites real evidence (records spans)."""
        available = [item.id for item in evidence]
        guards = GuardrailPipeline([CitationRequiredGuard(), GroundednessGuard()], trace=self.trace)
        for direction in state.get("directions", []):
            results = guards.run(
                GuardrailInput(
                    text=direction.one_liner,
                    citations=list(direction.evidence_ids),
                    available_evidence=available,
                ),
                stage=f"output:{direction.id}",
            )
            if guards.is_blocked(results):
                logger.warning(
                    "Direction %s failed an output guardrail: %s",
                    direction.id,
                    [r.reason for r in results if not r.passed],
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
