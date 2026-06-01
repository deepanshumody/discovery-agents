"""Optional LangGraph adapter.

Runs the SAME discovery graph through LangGraph instead of the built-in
StateMachine, demonstrating framework fluency without coupling the core to it.
Lazy-imported; raises a clear error if the `langgraph` extra is not installed.
"""

from __future__ import annotations

from collections.abc import Callable

from .state_machine import Node, State, StateMachine


def run_via_langgraph(machine: StateMachine, initial_state: State) -> State:
    """Build a LangGraph StateGraph from the machine's nodes and invoke it."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - only when extra is absent
        raise ImportError(
            "The LangGraph adapter requires the 'langgraph' extra: "
            "pip install 'discovery-agents[langgraph]'"
        ) from exc

    ordered = machine.ordered_nodes(set(initial_state))
    if not ordered:  # pragma: no cover - empty graph
        return dict(initial_state)

    builder = StateGraph(dict)

    def _wrap(node: Node) -> Callable[[State], State]:  # pragma: no cover - langgraph only
        def _fn(state: State) -> State:
            node.run(state)
            return state

        return _fn

    for node in ordered:  # pragma: no cover
        builder.add_node(node.name, _wrap(node))

    builder.set_entry_point(ordered[0].name)  # pragma: no cover
    for current, nxt in zip(ordered, ordered[1:]):  # pragma: no cover
        builder.add_edge(current.name, nxt.name)
    builder.add_edge(ordered[-1].name, END)  # pragma: no cover

    app = builder.compile()  # pragma: no cover
    result = app.invoke(dict(initial_state))  # pragma: no cover
    return dict(result)  # pragma: no cover
