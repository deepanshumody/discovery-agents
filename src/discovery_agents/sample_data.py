"""Sample inputs for an enterprise agentic-workflow discovery demo.

The scenario is fictional and self-contained: an enterprise software vendor's
platform team is deciding which agentic-AI features to ship to its large,
regulated enterprise customers. It exercises the full pipeline without claiming
access to any real company's private data.

Evidence tags are chosen so the EvidenceInsightAgent clusters them into the
themes the downstream agents reason about (exploration, alignment, context,
handoff, feedback loop, and differentiation).
"""

from __future__ import annotations

from .models import EvidenceItem, ProductBrief

SAMPLE_BRIEF = ProductBrief(
    company="Atlas Industries (Enterprise Platform Group)",
    product="internal agentic-workflow workspace for the enterprise product organization",
    target_user="enterprise PMs, staff engineers, and solutions architects",
    goal=(
        "Decide which agentic-AI features are worth building for large, regulated "
        "enterprise customers before committing design or engineering time."
    ),
    constraints=[
        "The output must produce multiple evidence-backed directions, not a single design.",
        "Every direction must cite customer or product evidence (auditable provenance).",
        "The final handoff must be implementation-ready for an engineer or coding agent.",
        "Outputs must be safe and reviewable: no unsupported claims, no leaked PII.",
        "The workflow must support multi-stakeholder decision-making across teams.",
    ],
    strategic_themes=[
        "enterprise agentic workflows",
        "evidence-grounded reasoning",
        "rigorous evaluation",
        "auditability and safety",
        "coding-agent handoff",
    ],
)


SAMPLE_EVIDENCE = [
    EvidenceItem(
        id="E1",
        source="customer_advisory_board",
        user_segment="enterprise product lead",
        severity=5,
        tags=["blank_state", "ideation", "speed"],
        text=(
            "We know the business problem, but turning it into several concrete, "
            "buildable agent workflows quickly is hard. We want to compare options, "
            "not read one long recommendation."
        ),
    ),
    EvidenceItem(
        id="E2",
        source="enterprise_sales_call",
        user_segment="VP of engineering",
        severity=4,
        tags=["alignment", "stakeholders", "decision"],
        text=(
            "Our teams burn weeks debating approaches in slide decks. We need to "
            "compare alternatives side by side and align stakeholders on what to "
            "pilot next."
        ),
    ),
    EvidenceItem(
        id="E3",
        source="support_escalation",
        user_segment="solutions architect",
        severity=3,
        tags=["visual_quality", "brand_fit", "context"],
        text=(
            "Generated proposals often look polished but miss our domain context, "
            "compliance constraints, and prior architectural decisions."
        ),
    ),
    EvidenceItem(
        id="E4",
        source="solutions_architect_note",
        user_segment="staff engineer",
        severity=5,
        tags=["handoff", "implementation", "engineering"],
        text=(
            "The real gap is the handoff. A direction can look good, but engineering "
            "still needs requirements, data contracts, edge cases, analytics events, "
            "and acceptance criteria before anyone can build it."
        ),
    ),
    EvidenceItem(
        id="E5",
        source="product_telemetry",
        user_segment="mixed",
        severity=4,
        tags=["selection", "feedback", "learning_loop"],
        text=(
            "Teams usually pick one of the first few proposed directions, but heavily "
            "edit it before circulating it for sign-off. We have no loop that learns "
            "from those edits."
        ),
    ),
    EvidenceItem(
        id="E6",
        source="win_loss_analysis",
        user_segment="product manager",
        severity=3,
        tags=["differentiation", "generic_outputs", "critique"],
        text=(
            "Off-the-shelf agent tools produce a single generic artifact without "
            "explaining tradeoffs, risks, or why a direction is worth funding over "
            "the alternatives."
        ),
    ),
]
