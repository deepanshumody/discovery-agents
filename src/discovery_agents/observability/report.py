"""Render a Trace as an HTML timeline (used by the canvas/report output)."""

from __future__ import annotations

import html

from .trace import Trace


def render_trace_html(trace: Trace) -> str:
    """A compact HTML fragment showing per-span latency, tokens, and cost."""
    usage = trace.total_usage
    rows = []
    for span in trace.spans:
        rows.append(
            "<tr>"
            f"<td>{html.escape(span.agent)}</td>"
            f"<td>{html.escape(span.op)}</td>"
            f"<td>{html.escape(span.model or '—')}</td>"
            f"<td>{span.usage.total_tokens}</td>"
            f"<td>${span.cost_usd:.6f}</td>"
            f"<td>{span.latency_ms:.1f}</td>"
            f"<td>{html.escape(span.message)}</td>"
            "</tr>"
        )
    return (
        '<section class="trace">'
        "<h2>Run trace</h2>"
        f"<p>{len(trace.spans)} spans · {usage.total_tokens} tokens · "
        f"${trace.total_cost_usd:.6f} · {trace.total_latency_ms:.1f} ms</p>"
        "<table><thead><tr>"
        "<th>Agent</th><th>Op</th><th>Model</th><th>Tokens</th><th>Cost</th>"
        "<th>Latency (ms)</th><th>Message</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )
