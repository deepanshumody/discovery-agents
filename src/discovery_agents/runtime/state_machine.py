"""A small, typed state machine for orchestrating agents.

Nodes declare the state keys they `require` and `provide`; the machine computes a
topological order, validates that every dependency is satisfied, runs each node
against a shared state blackboard, and records a trace span (with latency) per
node. This is the auditable, dependency-light core the JD calls a "custom state
machine"; a LangGraph adapter can run the same graph (see langgraph_adapter).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from ..observability.trace import Trace, TraceSpan

State = dict[str, Any]
NodeFn = Callable[[State], None]


@dataclass
class Node:
    name: str
    run: NodeFn
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)


class StateMachine:
    """Executes nodes in dependency order over a shared state dict."""

    def __init__(self, trace: Trace | None = None) -> None:
        self.nodes: list[Node] = []
        self.trace: Trace = trace or Trace()

    def add(
        self,
        name: str,
        run: NodeFn,
        *,
        requires: list[str] | None = None,
        provides: list[str] | None = None,
    ) -> StateMachine:
        self.nodes.append(Node(name, run, list(requires or []), list(provides or [])))
        return self

    def ordered_nodes(self, available: set[str]) -> list[Node]:
        """Topologically sort nodes; `available` keys are pre-satisfied inputs."""
        provider: dict[str, Node] = {}
        for node in self.nodes:
            for key in node.provides:
                provider[key] = node

        ordered: list[Node] = []
        visited: set[str] = set()
        in_progress: set[str] = set()

        def visit(node: Node) -> None:
            if node.name in visited:
                return
            if node.name in in_progress:
                raise ValueError(f"dependency cycle detected at node '{node.name}'")
            in_progress.add(node.name)
            for requirement in node.requires:
                if requirement in available:
                    continue
                dependency = provider.get(requirement)
                if dependency is None:
                    raise ValueError(
                        f"no node provides required key '{requirement}' for node '{node.name}'"
                    )
                visit(dependency)
            in_progress.discard(node.name)
            visited.add(node.name)
            ordered.append(node)

        for node in self.nodes:
            visit(node)
        return ordered

    def run(self, initial_state: State) -> State:
        state: State = dict(initial_state)
        for node in self.ordered_nodes(set(initial_state)):
            missing = [r for r in node.requires if r not in state]
            if missing:
                raise ValueError(f"node '{node.name}' is missing required state: {missing}")
            start = perf_counter()
            node.run(state)
            latency_ms = (perf_counter() - start) * 1000.0
            self.trace.record(
                TraceSpan(
                    agent=node.name,
                    op="node",
                    message="state-machine node",
                    latency_ms=latency_ms,
                    payload={"provides": node.provides},
                )
            )
        return state
