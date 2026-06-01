"""Tests for the deterministic mock LLM client and the provider factory."""

from __future__ import annotations

from discovery_agents.config import RunConfig
from discovery_agents.llm import LLMResponse, Message, MockLLMClient, ToolInvocation, get_client


def _messages() -> list[Message]:
    return [Message("system", "be helpful"), Message("user", "cluster this evidence")]


def test_mock_is_deterministic() -> None:
    a = MockLLMClient().chat(_messages())
    b = MockLLMClient().chat(_messages())
    assert a.text == b.text
    assert a.usage.input_tokens == b.usage.input_tokens
    assert a.usage.output_tokens == b.usage.output_tokens


def test_mock_populates_usage_and_model() -> None:
    resp = MockLLMClient().chat(_messages())
    assert resp.model == "mock-1"
    assert resp.usage.total_tokens > 0
    assert resp.latency_ms == 0.0


def test_mock_structured_is_none_by_default() -> None:
    # The mock never fabricates domain answers; agents fall back to their baseline.
    resp = MockLLMClient().chat(_messages(), response_format={"type": "object"})
    assert resp.structured is None


def test_mock_script_returns_responses_in_order() -> None:
    script = [
        LLMResponse(tool_calls=[ToolInvocation(name="calculator", arguments={"x": 1})]),
        LLMResponse(text="final answer"),
    ]
    client = MockLLMClient(script=script)
    first = client.chat(_messages())
    second = client.chat(_messages())
    assert first.tool_calls[0].name == "calculator"
    assert second.text == "final answer"


def test_factory_defaults_to_mock() -> None:
    client = get_client(RunConfig(provider="mock"))
    assert isinstance(client, MockLLMClient)


def test_factory_falls_back_to_mock_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = get_client(RunConfig(provider="anthropic"))
    assert isinstance(client, MockLLMClient)


def test_factory_unknown_provider_falls_back_to_mock() -> None:
    client = get_client(RunConfig(provider="does-not-exist"))
    assert isinstance(client, MockLLMClient)
