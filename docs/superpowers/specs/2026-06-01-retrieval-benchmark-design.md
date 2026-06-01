# Design: Real Retrieval Benchmark + Real-Data Agent (substance upgrade)

- **Date:** 2026-06-01
- **Status:** Approved
- **Author:** Deepanshu Mody
- **Why:** Move the repo from "demo" to "substance" — a real dataset, a real task, and a
  reproducible, committed result that proves the trained embedder earns its place.

## 1. Context & motivation

The platform's plumbing is solid but it runs on toy inputs (6 hand-written evidence items, a
34-word synthetic corpus), so nothing proves a component is *load-bearing*. This adds a real
benchmark: train the embedder on a real dataset and **measure** that it beats the lexical
`HashingEmbedder` at retrieval, with the numbers committed to the repo. It also runs the agent
pipeline on real data and finally tests the real LLM parsing path.

## 2. Dataset

**Banking77** (`mteb/banking77` on HuggingFace; original Casanueva et al. 2020, **CC-BY-4.0**):
9,993 train / 3,076 test customer-support utterances across **77 intents** (`text`, `label`,
`label_text`). Same intent = relevant; intents are phrased many ways, so a model that learns
intent semantics can beat a purely lexical embedder — a fair win condition.

- `mlinfra/datasets.py` → `load_banking77(full=False)`:
  - `full=False` (default): reads a committed stratified slice `benchmark/data/banking77_sample.csv`
    (~8 train / ~4 test per intent) via the stdlib `csv` module — **offline, CI-safe, no extra deps**.
  - `full=True`: loads the complete split via HuggingFace `datasets` (the `[benchmark]` extra).
  - Returns `(train, test)` as lists of `(text: str, label: int)` plus `label_names: list[str]`.
- Attribution + license recorded in `benchmark/data/README.md`.

## 3. Goals / non-goals

### Goals
- A `benchmark` command that trains a supervised-contrastive embedder on Banking77 and reports
  **recall@{1,5,10}, MRR, mAP** for **HashingEmbedder vs trained TorchEmbedder vs (optional)
  sentence-transformers**, written to `benchmark/RESULTS.md` + `benchmark/results.json`.
- The committed headline numbers come from a **`--full` run executed locally**; CI/tests run the
  same code on the committed sample.
- Run the **agent pipeline on real Banking77 utterances** (`discovery-agents --dataset banking77`).
- **Recorded-response tests** that exercise the real LLM parsing path (not the mock fallback).
- Everything stays typed, ruff/mypy-clean, and CPU-runnable.

### Non-goals
- Beating SOTA. We report whatever the numbers are; sentence-transformers is a reference upper
  bound, not a target. A near-tie with the lexical baseline is an honest, analyzed result.
- GPU training; new heavy required deps (the core/agent paths keep their current deps).

## 4. Components

```
src/discovery_agents/mlinfra/
  datasets.py            # load_banking77(full); sample CSV reader (stdlib) + full via `datasets`
  model/losses.py        # + supervised_contrastive(z, labels, temperature)  (SupCon)
  data/sampler.py        # LabelBatchSampler: classes_per_batch x samples_per_class index batches
  train/loop.py          # + Trainer.fit_supervised(tokens, labels, sampler_cfg, steps)
  retrieval_eval.py      # embed_texts, retrieval_metrics (recall@k/MRR/mAP), evaluate_embedder,
                         #   SentenceTransformerEmbedder (optional, lazy), run_benchmark(...)
  agent_data.py          # evidence_from_banking77(n) -> (ProductBrief, list[EvidenceItem])
  cli.py                 # + `benchmark` subcommand; rename I/O `bench` -> `io-bench`
src/discovery_agents/cli.py          # + `--dataset banking77` for the agent demo
src/discovery_agents/agents/evidence_insight.py   # data-driven clustering fallback (see §6)
benchmark/
  RESULTS.md             # committed real numbers (table) + reproduction command
  results.json           # machine-readable metrics
  data/banking77_sample.csv, data/README.md       # committed slice + attribution/license
```

## 5. Training & evaluation

- **SupCon loss** (`supervised_contrastive`): for L2-normalized batch embeddings `z` and integer
  `labels`, each anchor's positives are the other same-label rows; standard supervised contrastive
  (InfoNCE generalized to multiple positives). Returns a scalar; degrades gracefully when a label
  has no in-batch positive.
- **LabelBatchSampler**: each batch draws `classes_per_batch` intents × `samples_per_class`
  examples so positives always exist (e.g. 16 × 4 = batch 64).
- **`Trainer.fit_supervised(tokens, labels, ...)`**: reuses the platform's optimizer / warmup
  scheduler / checkpointing; one step = encode batch → SupCon → clip → step. Keeps SimCSE
  `fit`/`train_step` intact.
- **Retrieval eval** (`retrieval_eval.py`): pool = train split, queries = test split; embed both;
  for each query rank the pool by cosine; **recall@k** = a same-intent item appears in top-k;
  **MRR** = reciprocal rank of the first same-intent hit; **mAP** over same-intent hits in top-k.
  Compare `HashingEmbedder`, trained `TorchEmbedder`, and (if installed and `--with-st`) a
  `sentence-transformers` model.

## 6. Real-data agent

- `evidence_from_banking77(n)` builds a `ProductBrief` (a digital bank's product team deciding
  what to build from support messages) and `n` `EvidenceItem`s from real utterances, tagging each
  with its `label_text` (intent).
- `EvidenceInsightAgent` gains a **data-driven clustering fallback**: when the predefined `THEMES`
  match too little of the evidence (as with arbitrary real tags), it clusters by the most frequent
  tags actually present (top-N → themes). The original sample (whose tags match `THEMES`) is
  unaffected. This lets the pipeline produce real insights from real, arbitrarily-tagged evidence.

## 7. Proven LLM path (recorded-response tests)

A `RecordedLLMClient` (a thin wrapper over the scripted mock) replays realistic captured model
JSON. Tests assert: (a) `_parse_directions` + `extract_json` correctly parse realistic
structured output; (b) an `IdeationAgent` driven by a recorded structured response yields
**model-derived** directions (`source="llm"`, parsed citations) rather than the deterministic
fallback; (c) a full pipeline run on that client uses the model output. This tests the code the
deterministic mock skips — key-free.

## 8. Dependencies, CI, determinism

- New extra `[benchmark] = ["datasets>=2.18", "scikit-learn>=1.3"]`; `sentence-transformers` is an
  optional reference (`[st]`), lazy + skipped if absent. Sample-based benchmark needs only
  torch + numpy + stdlib csv → runs in the existing `ml` CI job.
- Training is seeded and deterministic on CPU. The committed `RESULTS.md` is from a `--full` run;
  CI runs `benchmark` on the committed sample and asserts it produces valid metrics (structure +
  ranges), not a specific score (small-sample scores are noisy).

## 9. Staged build (each stays green)

1. `datasets.py` (sample loader + full loader) + committed sample CSV + attribution + tests
2. SupCon loss + `LabelBatchSampler` + `Trainer.fit_supervised` + tests
3. `retrieval_eval.py` + `benchmark` CLI (+ rename I/O `bench`→`io-bench`) + run `--full` locally,
   commit `RESULTS.md`/`results.json` + Makefile/docs
4. `evidence_from_banking77` + `--dataset banking77` + data-driven clustering fallback +
   recorded-response LLM-path tests
5. Docs (headline number in README + role-mapping), adversarial review, final verify, merge

## 10. Risks & mitigations

- **Neural model may only tie the lexical baseline.** → Honest reporting; supervised contrastive +
  adequate steps give it the best shot; sentence-transformers reference contextualizes the gap.
- **HF download unavailable in CI.** → Committed sample + stdlib loader make CI offline; `--full`
  is local-only for the headline number.
- **Real tags break tag-based clustering.** → Data-driven clustering fallback (§6).
- **Scope creep into agent core.** → The clustering fallback is additive and guarded; the sample
  path is unchanged.
