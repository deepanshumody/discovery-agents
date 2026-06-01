"""Test the LangGraph adapter's wiring via a stub module (no real dependency).

The real-langgraph parity test in test_runtime_react.py is skipped without the extra;
this test injects a minimal fake `langgraph.graph` so the adapter's node/edge wiring is
always exercised deterministically in CI.
"""

from __future__ import annotations

import sys
import types

from discovery_agents.runtime.langgraph_adapter import run_via_langgraph
from discovery_agents.runtime.state_machine import StateMachine


def _install_fake_langgraph(monkeypatch) -> None:
    fake = types.ModuleType("langgraph")
    fake_graph = types.ModuleType("langgraph.graph")

    class FakeStateGraph:
        def __init__(self, schema: object) -> None:
            self.nodes: dict[str, object] = {}
            self.order: list[str] = []

        def add_node(self, name, fn):  # noqa: ANN001 - stub
            self.nodes[name] = fn
            self.order.append(name)

        def set_entry_point(self, name):  # noqa: ANN001 - stub
            self.entry = name

        def add_edge(self, a, b):  # noqa: ANN001 - stub
            pass

        def compile(self):  # noqa: ANN201 - stub
            order, nodes = self.order, self.nodes

            class App:
                def invoke(self, state):  # noqa: ANN001, ANN202 - stub
                    for name in order:
                        state = nodes[name](state)
                    return state

            return App()

    fake_graph.StateGraph = FakeStateGraph
    fake_graph.END = "__end__"
    monkeypatch.setitem(sys.modules, "langgraph", fake)
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_graph)


def test_adapter_runs_nodes_in_order_via_stub(monkeypatch) -> None:
    _install_fake_langgraph(monkeypatch)
    machine = StateMachine()
    machine.add("double", lambda s: s.__setitem__("y", s["x"] * 2), requires=["x"], provides=["y"])
    machine.add("plus", lambda s: s.__setitem__("z", s["y"] + 1), requires=["y"], provides=["z"])

    result = run_via_langgraph(machine, {"x": 21})
    assert result["z"] == 43
