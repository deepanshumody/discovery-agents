"""Golden evaluation cases.

Each case is a labeled input the harness runs end-to-end. Extra enterprise cases
can be appended here; the harness aggregates metrics across all of them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import EvidenceItem, ProductBrief
from ..sample_data import SAMPLE_BRIEF, SAMPLE_EVIDENCE


@dataclass
class EvalCase:
    id: str
    brief: ProductBrief
    evidence: list[EvidenceItem]


GOLDEN_CASES: list[EvalCase] = [
    EvalCase(
        id="enterprise-discovery",
        brief=SAMPLE_BRIEF,
        evidence=SAMPLE_EVIDENCE,
    ),
]
