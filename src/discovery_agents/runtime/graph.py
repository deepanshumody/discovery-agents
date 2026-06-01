"""The product-discovery graph: the nine agents wired as state-machine nodes.

Each node reads its inputs from and writes its outputs to the shared state, with
`requires`/`provides` capturing the data dependencies the pipeline used to encode
imperatively. The same graph can be executed by the custom StateMachine or by the
LangGraph adapter.
"""

from __future__ import annotations

from ..agents import (
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
from ..models import ProductDirection
from ..observability.trace import Trace
from .state_machine import State, StateMachine


def build_discovery_graph(
    *,
    trace: Trace,
    evidence_agent: EvidenceInsightAgent,
    strategy_agent: ProductStrategyAgent,
    ideation_agent: IdeationAgent,
    critique_agent: CritiqueAgent,
    canvas_agent: CanvasAgent,
    selection_agent: SelectionAgent,
    handoff_agent: HandoffAgent,
    eval_agent: EvalAgent,
    memory_agent: DecisionMemoryAgent,
) -> StateMachine:
    machine = StateMachine(trace=trace)

    def insights_node(state: State) -> None:
        state["insights"] = evidence_agent.run(state["evidence"], index=state["index"])

    def opportunities_node(state: State) -> None:
        state["opportunities"] = strategy_agent.run(state["brief"], state["insights"])

    def directions_node(state: State) -> None:
        state["directions"] = ideation_agent.run(
            state["brief"], state["insights"], state["opportunities"], index=state["index"]
        )

    def critiques_node(state: State) -> None:
        state["critiques"] = critique_agent.run(state["directions"], state["evidence"])

    def canvas_node(state: State) -> None:
        state["canvas_cards"] = canvas_agent.run(state["directions"], state["critiques"])

    def selection_node(state: State) -> None:
        state["selected_direction_id"] = selection_agent.run(state["critiques"])

    def handoff_node(state: State) -> None:
        directions: list[ProductDirection] = state["directions"]
        selected_id = state["selected_direction_id"]
        selected = next(d for d in directions if d.id == selected_id)
        state["coding_spec"] = handoff_agent.run(selected)

    def evals_node(state: State) -> None:
        state["evals"] = eval_agent.run(
            state["brief"],
            state["evidence"],
            state["directions"],
            state["critiques"],
            state["coding_spec"],
        )

    def memory_node(state: State) -> None:
        state["decision_log"] = memory_agent.run(
            state["selected_direction_id"], state["directions"], state["critiques"]
        )

    return (
        machine.add(
            "EvidenceInsightAgent",
            insights_node,
            requires=["evidence", "index"],
            provides=["insights"],
        )
        .add(
            "ProductStrategyAgent",
            opportunities_node,
            requires=["brief", "insights"],
            provides=["opportunities"],
        )
        .add(
            "IdeationAgent",
            directions_node,
            requires=["brief", "insights", "opportunities", "index"],
            provides=["directions"],
        )
        .add(
            "CritiqueAgent",
            critiques_node,
            requires=["directions", "evidence"],
            provides=["critiques"],
        )
        .add(
            "CanvasAgent",
            canvas_node,
            requires=["directions", "critiques"],
            provides=["canvas_cards"],
        )
        .add(
            "SelectionAgent",
            selection_node,
            requires=["critiques"],
            provides=["selected_direction_id"],
        )
        .add(
            "HandoffAgent",
            handoff_node,
            requires=["directions", "selected_direction_id"],
            provides=["coding_spec"],
        )
        .add(
            "EvalAgent",
            evals_node,
            requires=["brief", "evidence", "directions", "critiques", "coding_spec"],
            provides=["evals"],
        )
        .add(
            "DecisionMemoryAgent",
            memory_node,
            requires=["selected_direction_id", "directions", "critiques"],
            provides=["decision_log"],
        )
    )
