# ML-Infrastructure Platform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline, per-stage checkpoints). Steps use `- [ ]` checkboxes.

**Goal:** Add a production ML-infra platform (`mlinfra/`) that trains a neural retrieval embedder and serves it through the existing `Embedder` protocol — PyTorch DDP + resumable checkpointing, Zarr/HDF5/Numpy tensor archive + GPU-native loader + I/O benchmark, Dask/Local curation, MLflow/JSON tracking, Docker + K8s.

**Architecture:** `mlinfra/` is additive and optional behind a `[ml]` extra; the keyless agentic core is unchanged. Spec: `docs/superpowers/specs/2026-06-01-ml-infra-platform-design.md`.

**Tech stack:** PyTorch 2.x (CPU `gloo`), numpy, zarr, h5py, dask, mlflow (lazy), psutil; Docker + K8s manifests.

**Branch:** `feat/ml-infra-platform` (default `main`).

**Global invariants (after every stage):** existing keyless suite stays green (`pytest -q` with no extras assumptions for the agentic tests); `ruff check` + `ruff format --check` + `mypy` pass; new `mlinfra` tests pass under `.[ml,dask]`.

---

## Stage 0 — Extras, CI, skeleton, config
**Files:** `pyproject.toml` (extras + mypy override for mlinfra optional imports), `.github/workflows/ci.yml` (add `ml` job), `src/discovery_agents/mlinfra/__init__.py`, `src/discovery_agents/mlinfra/config.py`
- [ ] Install light deps locally: `pip install zarr h5py "dask[distributed]"` (torch/numpy/psutil present).
- [ ] Add `[ml]`/`[dask]`/`[mlflow]`/`[ray]`/`[tensorstore]` extras; add `[tool.mypy]` overrides so `mlinfra.*` may import torch/zarr/etc. (already `ignore_missing_imports`).
- [ ] Add CI `ml` job: `pip install -e ".[dev,ml,dask]"` → `ruff` → `mypy` → `pytest -q tests/mlinfra`.
- [ ] `config.py`: `ModelConfig`, `DataConfig`, `TrainConfig` dataclasses + `from_env`.
- [ ] Verify: `ruff && mypy && pytest -q`; commit `chore(ml): extras, CI job, mlinfra skeleton + config`.

## Stage 1 — Tokenizer + ArrayStore (Numpy/Zarr/HDF5)
**Files:** `mlinfra/tokenizer.py`, `mlinfra/store/{base,numpy_store,zarr_store,hdf5_store,factory}.py`; `tests/mlinfra/test_tokenizer.py`, `test_store.py`
- [ ] `WordVocab.build(corpus)` deterministic; `encode/decode/pad` to fixed seq_len; tests.
- [ ] `ArrayStore` protocol; `NumpyStore` (npz-backed, append via memmap/concat); `ZarrStore`, `HDF5Store` (chunked, resizable); `open_store(backend, path, mode)`.
- [ ] Test: round-trip + slice parity across backends (write rows, read `[start:end]`, assert equal to source) — Zarr/HDF5 `skipif` absent.
- [ ] Verify + commit `feat(ml): tokenizer + Array​Store (numpy/zarr/hdf5)`.

## Stage 2 — Curation (Local + Dask) → archive
**Files:** `mlinfra/curation/{base,executors,synthetic}.py`; `tests/mlinfra/test_curation.py`
- [ ] `Executor` protocol; `LocalExecutor.map` (sequential); `DaskExecutor.map` (lazy `dask`).
- [ ] `synthetic.generate_corpus(n, seed)` deterministic; `CorpusCurator.run(docs, store, vocab)` shards docs → token rows → `store.append`.
- [ ] Test: Local vs Dask produce identical archive contents (`skipif` no dask).
- [ ] Verify + commit `feat(ml): distributed corpus curation (local + dask)`.

## Stage 3 — Dataset + GPU-native DataLoader
**Files:** `mlinfra/data/{dataset,loader}.py`; `tests/mlinfra/test_dataset_loader.py`
- [ ] `ArrayStoreDataset(Dataset)` reads token rows by index from a store; `__len__/__getitem__`.
- [ ] `build_dataloader(dataset, batch_size, device, ...)`: collate to `LongTensor`, `pin_memory` when CUDA, prefetch, `.to(device, non_blocking=True)` helper.
- [ ] Test: deterministic batches; shapes `(B, seq_len)`; device-move helper no-op on CPU.
- [ ] Verify + commit `feat(ml): array-store dataset + gpu-native dataloader`.

## Stage 4 — Model (encoder) + InfoNCE
**Files:** `mlinfra/model/{encoder,losses}.py`; `tests/mlinfra/test_model.py`, `test_losses.py`
- [ ] `TextEncoder(ModelConfig)`: embedding + positional + `TransformerEncoder` + masked mean-pool + L2-norm; forward `(B,T)->(B,dim)`.
- [ ] `info_nce(z1, z2, temperature)` SimCSE in-batch contrastive; returns loss (+ accuracy).
- [ ] Test: forward shapes + seeded determinism; InfoNCE of identical vs shuffled embeddings (loss lower when aligned); gradient flows.
- [ ] Verify + commit `feat(ml): transformer encoder + InfoNCE (SimCSE) loss`.

## Stage 5 — Trainer + checkpoint/resume + DDP
**Files:** `mlinfra/train/{distributed,checkpoint,loop}.py`; `tests/mlinfra/test_checkpoint.py`, `test_trainer.py`, `test_ddp.py`
- [ ] `distributed.py`: `setup(rank,world,backend)`, `cleanup()`, `is_main()`, `maybe_ddp(model)`, `spawn(fn,world)`.
- [ ] `checkpoint.py`: atomic `save(state, dir)` (tmp+rename), `load_latest(dir)`, captures `{model,opt,sched,step,rng(torch/np/py)}`; SIGTERM handler that checkpoints.
- [ ] `loop.py`: `Trainer.fit(steps)` — batch → dropout positives → InfoNCE → clip → step → metrics → periodic ckpt → tracker.
- [ ] Test: loss decreases on tiny overfit corpus; resume from ckpt reproduces next-step loss exactly; `test_ddp` spawns 2 `gloo` procs for a few steps and asserts parameters are identical across ranks (grad sync). Keep tiny.
- [ ] Verify + commit `feat(ml): DDP trainer + fault-tolerant resumable checkpointing`.

## Stage 6 — Tracking + I/O benchmark + profiler
**Files:** `mlinfra/tracking/{base,mlflow_tracker}.py`, `mlinfra/bench/{io_benchmark,profile}.py`; `tests/mlinfra/test_tracking.py`, `test_io_benchmark.py`
- [ ] `Tracker` protocol; `JSONTracker` (writes `runs/<id>/params.json`, `metrics.jsonl`, copies artifacts); `MLflowTracker` (lazy).
- [ ] `io_benchmark.run(store_paths)`: read throughput (MB/s), samples/s, p50/p95 latency per backend → report dict.
- [ ] `profile.ResourceSampler` (psutil cpu/mem; optional torch.profiler).
- [ ] Test: JSONTracker persists params/metrics/artifact; benchmark returns metrics for numpy (+zarr/hdf5 if present).
- [ ] Verify + commit `feat(ml): experiment tracking (json/mlflow) + i/o benchmark + profiler`.

## Stage 7 — TorchEmbedder + retrieval/pipeline integration
**Files:** `mlinfra/embedder.py`, `mlinfra/cli.py`, `pyproject.toml` (scripts), `src/discovery_agents/retrieval/embeddings.py` (+`get_embedder`), `src/discovery_agents/config.py` (`embedder` field), `src/discovery_agents/pipeline.py`; `tests/mlinfra/test_torch_embedder.py`
- [ ] `TorchEmbedder(checkpoint, vocab)`: implements `Embedder` (`dim/embed/embed_batch`), eval + `no_grad`, L2-norm; deterministic.
- [ ] `cli.py`: `curate|train|bench|export`; add `discovery-agents-train`/`-curate`/`-bench` scripts.
- [ ] `get_embedder(config)` in retrieval → `HashingEmbedder` default / `TorchEmbedder` when `embedder="torch"` + ckpt present; thread `RunConfig.embedder` through pipeline (default unchanged).
- [ ] Test: `TorchEmbedder` conforms to `Embedder`, stable in eval mode, and an `EvidenceIndex` built on it returns sensible top-k; end-to-end curate→train(2 steps)→export→embed smoke.
- [ ] Verify + commit `feat(ml): TorchEmbedder + opt-in retrieval/pipeline integration`.

## Stage 8 — Deploy + docs + review
**Files:** `deploy/{Dockerfile,docker-compose.yml,Makefile,k8s/train-job.yaml}`, `docs/ml-platform.md`, `README.md` (section), `docs/role-mapping.md` (Biohub table)
- [ ] Dockerfile (CPU `.[ml,dask]`), docker-compose (trainer + mlflow), K8s indexed Job (torchrun/DDP env), Makefile targets.
- [ ] `docs/ml-platform.md` (architecture, commands, scaling notes), README "ML platform" section + diagram, extend `docs/role-mapping.md` with the Biohub mapping.
- [ ] Adversarial multi-agent review of `mlinfra/`; fix confirmed findings with regression tests.
- [ ] Final: `ruff && mypy && pytest` (keyless + ml), demo, curate→train→bench smoke; commit; open PR / merge per user.

## Self-review (plan vs spec)
- Coverage: tokenizer/store (S1), curation (S2), loader (S3), model/loss (S4), DDP+ckpt (S5), tracking+bench (S6), embedder+integration (S7), deploy+docs (S8) — every spec section mapped.
- Deferred items (TensorStore, Ray, FSDP, serving) intentionally absent; documented as future work.
- Type names consistent: `ArrayStore`, `Executor`, `Tracker`, `TextEncoder`, `TorchEmbedder`, `info_nce`.
