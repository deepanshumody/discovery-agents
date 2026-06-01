# Evaluation

The eval harness moves quality measurement beyond trial-and-error: it scores a run with
deterministic metrics **and** an LLM-as-judge, aggregates a scorecard across golden cases,
and **gates regressions in CI** against a committed baseline.

```bash
discovery-agents --eval                    # score + regression gate (exit 1 on regression)
discovery-agents --eval --update-baseline  # accept the current scores as the new baseline
```

## Metrics (`eval/metrics.py`)

Deterministic under the mock provider, so they are safe to gate in CI:

| Metric | Meaning |
|---|---|
| `citation_precision` | Fraction of direction citations that point to real evidence ids |
| `evidence_coverage` | Fraction of evidence items cited by at least one direction |
| `direction_count` | Coverage of the "5+ divergent directions" target |
| `distinctiveness` | 1 − mean pairwise Jaccard similarity of directions |
| `handoff_completeness` | Fraction of handoff-spec sections populated |
| `selection_validity` | Selected direction is a real direction |
| `guardrail_pass_rate` | Fraction of guardrail checks that passed |

**Ops metrics** (`total_tokens`, `cost_usd`, `latency_ms`) are reported but **not gated**,
because they depend on the provider and machine.

## LLM-as-judge (`eval/judge.py`)

Scores the selected direction on **faithfulness** (claims supported by cited evidence),
**relevance** (addresses the goal), and **helpfulness** (clear, actionable). With a real
provider it asks the model for JSON scores in `[0,1]`. With the mock — or if the model
returns invalid JSON — it computes transparent deterministic heuristics (relevance via
embedding cosine similarity, reusing the RAG embedder), so CI stays stable.

## Regression gate (`eval/harness.py`)

`EvalHarness.evaluate()` runs the pipeline over the golden cases in `eval/cases.py` and
aggregates a `Scorecard`. `check_regression(scorecard, baseline, tolerance=0.02)` flags
any gated metric that drops more than the tolerance below the committed
`eval/baseline.json`. CI runs `discovery-agents --eval` and fails on a real regression.

## Extending

- Add golden cases to `eval/cases.py` (the harness averages metrics across them).
- Add a metric in `eval/metrics.py` and include it in `quality_metrics`.
- Regenerate the baseline with `--update-baseline` and commit it whenever metrics
  legitimately change.
