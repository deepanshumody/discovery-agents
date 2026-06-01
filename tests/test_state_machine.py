"""Tests for the typed state machine."""

from __future__ import annotations

import pytest

from discovery_agents.runtime.state_machine import State, StateMachine


def test_runs_nodes_in_dependency_order_even_if_added_out_of_order() -> None:
    order: list[str] = []

    def make(name: str, key: str):
        def _run(state: State) -> None:
            order.append(name)
            state[key] = True

        return _run

    machine = StateMachine()
    # Added c, b, a but dependencies force a -> b -> c.
    machine.add("c", make("c", "c"), requires=["b"], provides=["c"])
    machine.add("b", make("b", "b"), requires=["a"], provides=["b"])
    machine.add("a", make("a", "a"), requires=["seed"], provides=["a"])

    final = machine.run({"seed": True})
    assert order == ["a", "b", "c"]
    assert final["a"] and final["b"] and final["c"]


def test_threads_state_between_nodes() -> None:
    machine = StateMachine()
    machine.add("double", lambda s: s.__setitem__("y", s["x"] * 2), requires=["x"], provides=["y"])
    machine.add("plus", lambda s: s.__setitem__("z", s["y"] + 1), requires=["y"], provides=["z"])
    final = machine.run({"x": 21})
    assert final["z"] == 43


def test_missing_provider_raises() -> None:
    machine = StateMachine()
    machine.add("needs_missing", lambda s: None, requires=["nope"], provides=["out"])
    with pytest.raises(ValueError, match="no node provides required key 'nope'"):
        machine.run({})


def test_cycle_is_detected() -> None:
    machine = StateMachine()
    machine.add("a", lambda s: None, requires=["b"], provides=["a"])
    machine.add("b", lambda s: None, requires=["a"], provides=["b"])
    with pytest.raises(ValueError, match="cycle"):
        machine.run({})


def test_records_one_span_per_node() -> None:
    machine = StateMachine()
    machine.add("a", lambda s: s.__setitem__("a", 1), requires=["seed"], provides=["a"])
    machine.add("b", lambda s: s.__setitem__("b", 2), requires=["a"], provides=["b"])
    machine.run({"seed": True})
    node_spans = [s for s in machine.trace.spans if s.op == "node"]
    assert [s.agent for s in node_spans] == ["a", "b"]
