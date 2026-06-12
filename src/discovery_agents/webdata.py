"""Prefilled datasets for the web demo.

Two self-contained scenarios the visitor can run without uploading anything:

- ``sample``   — the enterprise agentic-workflow scenario (reused from ``sample_data``).
- ``banking``  — a banking customer-support discovery scenario seeded from the kinds of
  intents in the Banking77 dataset (card issues, transfers, top-ups, verification...).
  It is static text (no ``datasets`` download, no network) so the web image stays light;
  the tags are chosen so the EvidenceInsightAgent clusters it into the usual themes.
"""

from __future__ import annotations

from .models import EvidenceItem, ProductBrief
from .sample_data import SAMPLE_BRIEF, SAMPLE_EVIDENCE

BANKING_BRIEF = ProductBrief(
    company="Northwind Bank (Digital Servicing)",
    product="AI assistant for the mobile banking app's customer-support experience",
    target_user="retail banking customers and the support agents who back them up",
    goal=(
        "Decide which AI-assisted support features to build for the mobile banking app "
        "to deflect repetitive contacts without hurting trust on sensitive money actions."
    ),
    constraints=[
        "Every proposed direction must cite real customer evidence (auditable provenance).",
        "Money-movement and identity actions must stay safe and reviewable — no unsupported claims.",
        "The handoff must be implementation-ready for an engineer or coding agent.",
        "Directions must be distinct, not one design dressed up several ways.",
    ],
    strategic_themes=[
        "contact deflection",
        "trust on sensitive actions",
        "evidence-grounded reasoning",
        "agent assist",
        "coding-agent handoff",
    ],
)

BANKING_EVIDENCE: list[EvidenceItem] = [
    EvidenceItem(
        id="B1",
        source="support transcript",
        text=(
            "Customers repeatedly ask how to freeze or replace a lost or stolen card and can't "
            "find where to start in the app."
        ),
        user_segment="retail customer",
        severity=5,
        tags=["blank_state", "ideation"],
    ),
    EvidenceItem(
        id="B2",
        source="support transcript",
        text=(
            "People want an instant answer on why a card payment was declined instead of waiting "
            "in a queue to ask an agent."
        ),
        user_segment="retail customer",
        severity=4,
        tags=["speed", "ideation"],
    ),
    EvidenceItem(
        id="B3",
        source="agent interview",
        text=(
            "Support agents and the fraud team disagree on when an AI assistant may act on a "
            "suspicious transfer versus escalate to a human."
        ),
        user_segment="support agent",
        severity=4,
        tags=["alignment", "stakeholders", "decision"],
    ),
    EvidenceItem(
        id="B4",
        source="compliance review",
        text=(
            "Risk and compliance stakeholders need to sign off on any automated money-movement "
            "flow before it ships."
        ),
        user_segment="compliance",
        severity=5,
        tags=["alignment", "decision"],
    ),
    EvidenceItem(
        id="B5",
        source="brand guidelines",
        text=(
            "Assistant replies must match the bank's calm, plain-language tone and never invent "
            "fees, limits, or policy details."
        ),
        user_segment="brand",
        severity=4,
        tags=["brand_fit", "context"],
    ),
    EvidenceItem(
        id="B6",
        source="user research",
        text=(
            "Customers trust answers more when the assistant shows the specific transaction or "
            "statement it is referring to."
        ),
        user_segment="retail customer",
        severity=4,
        tags=["visual_quality", "context"],
    ),
    EvidenceItem(
        id="B7",
        source="engineering notes",
        text=(
            "Any new assistant feature must hand off to the existing card-services and transfers "
            "APIs with clear request/response contracts."
        ),
        user_segment="engineering",
        severity=4,
        tags=["handoff", "implementation", "engineering"],
    ),
    EvidenceItem(
        id="B8",
        source="product analytics",
        text=(
            "Top-up and balance questions are the highest-volume contacts, so deflecting them "
            "first would free the most agent time."
        ),
        user_segment="product",
        severity=5,
        tags=["selection", "feedback"],
    ),
    EvidenceItem(
        id="B9",
        source="post-contact survey",
        text=(
            "Customers who rate a self-serve answer poorly almost always reopen the same issue "
            "with an agent within a day."
        ),
        user_segment="retail customer",
        severity=3,
        tags=["feedback", "learning_loop"],
    ),
    EvidenceItem(
        id="B10",
        source="competitive teardown",
        text=(
            "Rival banking assistants give generic FAQ answers; customers say they want help tied "
            "to their own account, not a help-center article."
        ),
        user_segment="retail customer",
        severity=4,
        tags=["differentiation", "generic_outputs", "critique"],
    ),
]

# name -> (label, brief, evidence). The web form's dataset selector reads this registry.
DATASETS: dict[str, tuple[str, ProductBrief, list[EvidenceItem]]] = {
    "sample": ("Enterprise agentic workflows (sample)", SAMPLE_BRIEF, SAMPLE_EVIDENCE),
    "banking": ("Banking customer support (Banking77-style)", BANKING_BRIEF, BANKING_EVIDENCE),
}
