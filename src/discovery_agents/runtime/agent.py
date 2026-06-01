"""A ReAct / plan-execute agent loop.

The agent calls the model with the available tool specs; if the model requests a
tool, the loop runs it, appends the observation, and iterates; otherwise the
model's text is the final answer. It stops on a final answer, on `max_steps`, or
when an optional token budget is exhausted. Every model call and tool call is
recorded as a trace span, so the whole reasoning trajectory is auditable.

With the keyless mock the model returns no tool calls, so the loop terminates in
one step. Scripted mock responses (`MockLLMClient(script=[...])`) drive
deterministic multi-step tool-use tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMClient
from ..llm.types import Message
from ..observability.trace import Trace, TraceSpan
from ..tools.base import ToolRegistry


@dataclass
class AgentStep:
    tool: str
    arguments: dict[str, Any]
    observation: Any
    citations: list[str] = field(default_factory=list)
    ok: bool = True


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    structured: dict[str, Any] | None = None
    stop_reason: str = "final_answer"

    @property
    def citations(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            for cid in step.citations:
                if cid not in seen:
                    seen.append(cid)
        return seen


class LLMAgent:
    """A tool-using agent that reasons and acts over multiple steps."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        *,
        trace: Trace | None = None,
        name: str = "ReActAgent",
        max_steps: int = 6,
        token_budget: int | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.trace = trace or Trace()
        self.name = name
        self.max_steps = max_steps
        self.token_budget = token_budget

    def run(self, goal: str, *, system: str | None = None) -> AgentResult:
        system_prompt = system or self._default_system()
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=goal),
        ]
        tool_specs = self.tools.specs() or None
        steps: list[AgentStep] = []
        tokens_used = 0

        for _ in range(self.max_steps):
            response = self.llm.chat(messages, tools=tool_specs)
            self.trace.record_llm(self.name, "react.step", response)
            tokens_used += response.usage.total_tokens

            if not response.tool_calls:
                return AgentResult(
                    answer=response.text,
                    steps=steps,
                    structured=response.structured,
                    stop_reason="final_answer",
                )

            messages.append(Message(role="assistant", content=response.text))
            for call in response.tool_calls:
                result = self.tools.run(call.name, call.arguments)
                steps.append(
                    AgentStep(
                        tool=call.name,
                        arguments=call.arguments,
                        observation=result.data if result.ok else result.error,
                        citations=result.citations,
                        ok=result.ok,
                    )
                )
                self.trace.record(
                    TraceSpan(
                        agent=self.name,
                        op="tool",
                        message=f"tool:{call.name}",
                        tool_calls=[call.name],
                        payload={"ok": result.ok, "citations": result.citations},
                    )
                )
                messages.append(
                    Message(
                        role="user",
                        content=f"Observation from {call.name}: {result.data if result.ok else result.error}",
                    )
                )

            if self.token_budget is not None and tokens_used >= self.token_budget:
                return AgentResult(answer="", steps=steps, stop_reason="budget_exhausted")

        return AgentResult(answer="", steps=steps, stop_reason="max_steps")

    def _default_system(self) -> str:
        return (
            "You are a careful research agent. Use the available tools to gather grounded "
            "evidence before answering. Cite the evidence ids returned by tools. When you have "
            "enough information, stop calling tools and give a final answer."
        )
