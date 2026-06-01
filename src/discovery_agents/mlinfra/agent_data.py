"""Build agent inputs from real Banking77 support messages (torch-free).

Lets `discovery-agents --dataset banking77` run the discovery pipeline on real
customer-support utterances instead of the toy sample, with each message tagged by
its intent so the evidence agent can cluster real themes.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..models import EvidenceItem, ProductBrief
from .datasets import load_banking77

SAMPLE_BRIEF = ProductBrief(
    company="NeoBank",
    product="consumer digital banking app",
    target_user="retail banking customers",
    goal=(
        "Decide which product features to build next from real customer support "
        "messages, before committing design or engineering time."
    ),
    constraints=[
        "The output must produce multiple evidence-backed directions, not a single design.",
        "Every direction must cite the customer messages that support it.",
        "The final handoff must be implementation-ready.",
    ],
    strategic_themes=["self-service", "trust and safety", "reduce support load", "onboarding"],
)


def evidence_from_banking77(
    intents: int = 6, per_intent: int = 3, *, seed: int = 7, full: bool = False
) -> tuple[ProductBrief, list[EvidenceItem]]:
    """Return (brief, evidence) sampled from real Banking77 utterances.

    Picks `intents` distinct intents and `per_intent` messages each, tagging every
    EvidenceItem with its intent so themes have several supporting messages.
    """
    data = load_banking77(full=full)
    split = data.test if len(data.test) else data.train

    by_intent: dict[int, list[str]] = defaultdict(list)
    for text, label in zip(split.texts, split.labels):
        by_intent[label].append(text)

    rng = random.Random(seed)
    chosen = rng.sample(sorted(by_intent), min(intents, len(by_intent)))

    evidence: list[EvidenceItem] = []
    counter = 1
    for label in chosen:
        messages = by_intent[label]
        picks = rng.sample(messages, min(per_intent, len(messages)))
        intent_name = data.name_for(label)  # id->name map (label ids may be sparse)
        for text in picks:
            evidence.append(
                EvidenceItem(
                    id=f"E{counter}",
                    source="support_message",
                    user_segment="customer",
                    severity=3,
                    tags=[intent_name],
                    text=text,
                )
            )
            counter += 1
    return SAMPLE_BRIEF, evidence
