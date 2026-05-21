"""Sample inputs for a Remy-style product discovery demo.

This file avoids hard-coding a proprietary company. It uses a fictional product
team building a collaborative AI workspace, which is close enough to demonstrate
product discovery without claiming access to private Remy data.
"""

from __future__ import annotations

from .models import EvidenceItem, ProductBrief


SAMPLE_BRIEF = ProductBrief(
    company="Northstar Labs",
    product="multiplayer AI workspace for founders and product teams",
    target_user="early-stage founders, PMs, designers, and product engineers",
    goal=(
        "Help teams discover which product ideas are worth building before they "
        "commit design or engineering time."
    ),
    constraints=[
        "The output must produce multiple directions, not a single design.",
        "Every direction should cite customer or product evidence.",
        "The final handoff should be usable by a coding agent or engineer.",
        "The workflow should support multiplayer decision-making.",
    ],
    strategic_themes=[
        "product discovery",
        "visual reasoning",
        "agentic workflows",
        "decision memory",
        "coding-agent handoff",
    ],
)


SAMPLE_EVIDENCE = [
    EvidenceItem(
        id="E1",
        source="customer_interview",
        user_segment="seed founder",
        severity=5,
        tags=["blank_state", "ideation", "speed"],
        text=(
            "I usually know the customer problem but I struggle to turn it into "
            "several concrete product directions quickly. I want to see options, "
            "not just read a long answer."
        ),
    ),
    EvidenceItem(
        id="E2",
        source="sales_call",
        user_segment="product lead",
        severity=4,
        tags=["alignment", "stakeholders", "decision"],
        text=(
            "Our team wastes too much time debating abstract ideas in docs. We need "
            "a way to compare alternatives visually and agree on what to test next."
        ),
    ),
    EvidenceItem(
        id="E3",
        source="support_ticket",
        user_segment="designer",
        severity=3,
        tags=["visual_quality", "brand_fit", "context"],
        text=(
            "Generated mockups often look polished but miss our product context, "
            "brand tone, and previous design decisions."
        ),
    ),
    EvidenceItem(
        id="E4",
        source="founder_note",
        user_segment="founder",
        severity=5,
        tags=["handoff", "implementation", "engineering"],
        text=(
            "The biggest gap is the handoff. An idea can look good, but engineering "
            "still needs requirements, edge cases, analytics events, and acceptance criteria."
        ),
    ),
    EvidenceItem(
        id="E5",
        source="product_analytics",
        user_segment="mixed",
        severity=4,
        tags=["selection", "feedback", "learning_loop"],
        text=(
            "Users tend to select one of the first three generated ideas, but often "
            "make heavy edits before sharing with their team."
        ),
    ),
    EvidenceItem(
        id="E6",
        source="competitor_review",
        user_segment="PM",
        severity=3,
        tags=["differentiation", "generic_outputs", "critique"],
        text=(
            "Prompt-to-prototype tools are useful, but they often jump to a single "
            "artifact without explaining tradeoffs or why that direction is worth building."
        ),
    ),
]
