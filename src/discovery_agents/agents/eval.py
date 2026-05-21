"""Eval agent — scores the whole pipeline run."""

from __future__ import annotations

import itertools
from typing import List

from ..agent_base import BaseAgent
from ..models import (
    CodingSpec,
    CritiqueScore,
    EvalResult,
    EvidenceItem,
    ProductBrief,
    ProductDirection,
)
from ._utils import tokenize


class EvalAgent(BaseAgent):
    """Evaluates the overall agent run."""

    name = "EvalAgent"

    def run(
        self,
        brief: ProductBrief,
        evidence: List[EvidenceItem],
        directions: List[ProductDirection],
        critiques: List[CritiqueScore],
        spec: CodingSpec,
    ) -> List[EvalResult]:
        evals = [
            self._direction_count_eval(directions),
            self._evidence_coverage_eval(evidence, directions),
            self._distinctiveness_eval(directions),
            self._constraint_coverage_eval(brief, directions, spec),
            self._handoff_completeness_eval(spec),
            self._risk_visibility_eval(directions, critiques),
        ]
        self.log("Evaluated agent run", eval_count=len(evals))
        return evals

    def _direction_count_eval(self, directions: List[ProductDirection]) -> EvalResult:
        score = min(1.0, len(directions) / 5)
        return EvalResult("direction_diversity_count", score, f"Generated {len(directions)} directions; target is 5+.")

    def _evidence_coverage_eval(self, evidence: List[EvidenceItem], directions: List[ProductDirection]) -> EvalResult:
        cited = set(itertools.chain.from_iterable(d.evidence_ids for d in directions))
        available = {e.id for e in evidence}
        score = len(cited & available) / max(1, len(available))
        return EvalResult("evidence_coverage", round(score, 2), f"Cited {len(cited & available)} of {len(available)} evidence items.")

    def _distinctiveness_eval(self, directions: List[ProductDirection]) -> EvalResult:
        if len(directions) < 2:
            return EvalResult("semantic_distinctiveness", 0.0, "Need at least two directions to compare.")
        token_sets = [set(tokenize(d.title + " " + d.one_liner + " " + d.differentiator)) for d in directions]
        similarities = []
        for a, b in itertools.combinations(token_sets, 2):
            similarities.append(len(a & b) / max(1, len(a | b)))
        avg_similarity = sum(similarities) / len(similarities)
        score = round(max(0.0, 1.0 - avg_similarity), 2)
        return EvalResult("semantic_distinctiveness", score, f"Average pairwise Jaccard similarity is {avg_similarity:.2f}.")

    def _constraint_coverage_eval(self, brief: ProductBrief, directions: List[ProductDirection], spec: CodingSpec) -> EvalResult:
        combined = " ".join(
            [d.one_liner + " " + d.core_loop + " " + d.differentiator for d in directions]
            + spec.functional_requirements
            + spec.non_functional_requirements
        ).lower()
        hits = 0
        for constraint in brief.constraints:
            keywords = [w for w in tokenize(constraint) if len(w) > 4]
            if any(k in combined for k in keywords):
                hits += 1
        score = hits / max(1, len(brief.constraints))
        return EvalResult("constraint_coverage", round(score, 2), f"Covered {hits} of {len(brief.constraints)} constraints.")

    def _handoff_completeness_eval(self, spec: CodingSpec) -> EvalResult:
        sections = [
            spec.user_story,
            spec.functional_requirements,
            spec.non_functional_requirements,
            spec.data_contract,
            spec.acceptance_criteria,
            spec.analytics_events,
            spec.open_questions,
        ]
        complete = sum(1 for section in sections if section)
        score = complete / len(sections)
        return EvalResult("handoff_completeness", round(score, 2), f"Completed {complete} of {len(sections)} handoff sections.")

    def _risk_visibility_eval(self, directions: List[ProductDirection], critiques: List[CritiqueScore]) -> EvalResult:
        directions_with_risks = sum(1 for d in directions if d.risks)
        critiques_with_next_steps = sum(1 for c in critiques if c.recommended_next_step)
        score = (directions_with_risks + critiques_with_next_steps) / max(1, len(directions) + len(critiques))
        return EvalResult("risk_visibility", round(score, 2), "Checks whether risks and next steps are explicit.")
