"""Renderers for Markdown and HTML outputs."""

from __future__ import annotations

import html
from typing import Dict

from .models import AgentRun


def render_run_markdown(run: AgentRun) -> str:
    critique_by_id = {c.direction_id: c for c in run.critiques}
    evidence_by_id = {e.id: e for e in run.evidence}

    lines = [
        "# Product Discovery Agent Run",
        "",
        "## Brief",
        f"- Company: **{run.brief.company}**",
        f"- Product: **{run.brief.product}**",
        f"- Target user: **{run.brief.target_user}**",
        f"- Goal: {run.brief.goal}",
        "",
        "## Insights",
    ]
    for insight in run.insights:
        lines.extend(
            [
                f"### {insight.id}. {insight.title}",
                f"{insight.summary}",
                f"Evidence: {', '.join(insight.evidence_ids)} | Confidence: {insight.confidence}",
                "",
            ]
        )

    lines.append("## Candidate Product Directions")
    for direction in run.directions:
        critique = critique_by_id[direction.id]
        evidence_quotes = [f"{eid}: {evidence_by_id[eid].text}" for eid in direction.evidence_ids if eid in evidence_by_id]
        lines.extend(
            [
                f"### {direction.id}. {direction.title}",
                f"**One-liner:** {direction.one_liner}",
                f"**Core loop:** {direction.core_loop}",
                f"**Differentiator:** {direction.differentiator}",
                f"**Score:** {critique.weighted_score} / 5",
                f"**Recommended next step:** {critique.recommended_next_step}",
                "**Evidence:**",
            ]
        )
        for quote in evidence_quotes:
            lines.append(f"- {quote}")
        lines.extend(["**Risks:**"] + [f"- {risk}" for risk in direction.risks] + [""])

    lines.extend(["## Selected Direction", f"Selected: **{run.selected_direction_id}**", ""])
    lines.extend(["## Evals"])
    for result in run.evals:
        lines.append(f"- **{result.metric}: {result.score}** — {result.explanation}")

    lines.extend(["", "## Decision Memory"])
    for entry in run.decision_log:
        lines.append(f"- {entry}")

    return "\n".join(lines)


def render_handoff_markdown(run: AgentRun) -> str:
    if not run.coding_spec:
        return "# Coding Handoff\n\nNo selected direction."
    spec = run.coding_spec
    direction = next(d for d in run.directions if d.id == spec.direction_id)
    lines = [
        "# Coding-Agent Handoff Packet",
        "",
        f"## Feature: {spec.feature_name}",
        f"Selected direction: `{spec.direction_id}`",
        "",
        "## Product Rationale",
        direction.differentiator,
        "",
        "## User Story",
        spec.user_story,
        "",
        "## Functional Requirements",
    ]
    lines.extend(f"- {item}" for item in spec.functional_requirements)
    lines.extend(["", "## Non-Functional Requirements"])
    lines.extend(f"- {item}" for item in spec.non_functional_requirements)
    lines.extend(["", "## Data Contract"])
    for entity, fields in spec.data_contract.items():
        lines.append(f"- **{entity}:** {', '.join(fields)}")
    lines.extend(["", "## Acceptance Criteria"])
    lines.extend(f"- {item}" for item in spec.acceptance_criteria)
    lines.extend(["", "## Analytics Events"])
    lines.extend(f"- `{event}`" for event in spec.analytics_events)
    lines.extend(["", "## Open Questions"])
    lines.extend(f"- {item}" for item in spec.open_questions)
    return "\n".join(lines)


def render_canvas_html(run: AgentRun) -> str:
    critique_by_id: Dict[str, float] = {c.direction_id: c.weighted_score for c in run.critiques}
    selected = run.selected_direction_id
    cards_html = []
    for card in run.canvas_cards:
        is_selected = " selected" if card.id == selected else ""
        body = html.escape(card.body).replace("\n", "<br>")
        evidence = ", ".join(card.evidence)
        cards_html.append(
            f"""
            <article class="card{is_selected}">
              <div class="card-meta"><span>{html.escape(card.column)}</span><span>Score {critique_by_id.get(card.id, 0):.2f}</span></div>
              <h2>{html.escape(card.title)}</h2>
              <p>{body}</p>
              <div class="evidence">Evidence: {html.escape(evidence)}</div>
            </article>
            """
        )

    eval_rows = "".join(
        f"<tr><td>{html.escape(e.metric)}</td><td>{e.score:.2f}</td><td>{html.escape(e.explanation)}</td></tr>"
        for e in run.evals
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Product Discovery Agent Canvas</title>
  <style>
    :root {{
      --bg: #f7f5ef;
      --ink: #181713;
      --muted: #6f6a5f;
      --line: #d8d0bf;
      --card: #fffdf8;
      --selected: #fff2be;
    }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ padding: 42px 48px 24px; max-width: 1180px; margin: auto; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: .12em; font-size: 12px; color: var(--muted); font-weight: 700; }}
    h1 {{ font-size: 44px; line-height: 1; margin: 10px 0 14px; }}
    .sub {{ font-size: 18px; line-height: 1.5; color: var(--muted); max-width: 880px; }}
    .canvas {{ max-width: 1180px; margin: 18px auto 42px; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 16px; padding: 0 48px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 22px; padding: 18px; box-shadow: 0 8px 24px rgba(0,0,0,.05); min-height: 320px; }}
    .card.selected {{ background: var(--selected); border-color: #bd9f2f; }}
    .card-meta {{ display: flex; justify-content: space-between; gap: 12px; font-size: 12px; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }}
    h2 {{ font-size: 20px; margin: 18px 0 10px; line-height: 1.15; }}
    p {{ font-size: 14px; line-height: 1.48; }}
    .evidence {{ margin-top: 16px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 12px; }}
    section {{ max-width: 1180px; margin: 0 auto 56px; padding: 0 48px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 18px; overflow: hidden; border: 1px solid var(--line); }}
    th, td {{ text-align: left; padding: 14px 16px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ background: #eee7d8; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    @media (max-width: 1000px) {{ .canvas {{ grid-template-columns: 1fr; }} h1 {{ font-size: 34px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Agentic Product Discovery Demo</div>
    <h1>From evidence to product direction</h1>
    <div class="sub">This canvas demonstrates a product-discovery agent system: cluster evidence, generate multiple directions, critique tradeoffs, select a path, and produce a coding-agent handoff.</div>
  </header>
  <main class="canvas">
    {''.join(cards_html)}
  </main>
  <section>
    <h2>Pipeline evals</h2>
    <table>
      <thead><tr><th>Metric</th><th>Score</th><th>Explanation</th></tr></thead>
      <tbody>{eval_rows}</tbody>
    </table>
  </section>
</body>
</html>"""
