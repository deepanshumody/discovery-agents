# Design: ML-Infrastructure Platform (trainable RAG embedder)

- **Date:** 2026-06-01
- **Status:** Approved (leaner first cut)
- **Author:** Deepanshu Mody
- **Target role this supports:** Biohub — *Machine Learning Engineer, AI* (ML infrastructure at scale)

## 1. Context & motivation

`discovery-agents` is a production-grade agentic RAG workflow. Its retrieval layer uses a
deterministic toy `HashingEmbedder`. To make the repo a credible centerpiece for an **ML
infrastructure** role (PyTorch/distributed training, GPU-native I/O over scientific tensor
formats, Ray/Dask, MLOps, Docker/K8s), we add a real ML workload where one authentically
belongs: **a trainable neural retrieval embedder** plus the **production ML-infra platform**
that curates data, trains the model at scale, tracks experiments, benchmarks I/O, and deploys.

The trained model plugs into the existing `Embedder` protocol, so the platform actually
improves the product rather than bolting on unrelated code.

## 2. Goals / non-goals

### Goals
- A reusable ML-infra platform under `mlinfra/` that **trains a sentence embedder** and a
  `TorchEmbedder` that implements the existing `retrieval.embeddings.Embedder` protocol.
- **PyTorch** custom training loop with **DDP** (`gloo` on CPU, `nccl` on GPU) and
  **fault-tolerant, resumable checkpointing**.
- A multi-dimensional **tensor archive** with pluggable backends — **NumpyStore** (default,
  light), **ZarrStore**, **HDF5Store** — and a **GPU-native data loader**.
- A **distributed data-curation** pipeline (Local + **Dask** executors) that writes the archive.
- An **I/O benchmark** comparing backend read throughput/latency, plus a resource profiler.
- **Experiment/artifact tracking** — a light `JSONTracker` (default) + an `MLflowTracker`.
- **Docker** image + **Kubernetes** training **Job** manifest + a local `docker-compose`.
- **Keyless/CPU-first**: the agentic core stays dependency-free; the platform runs on the
  `[ml]` extra, trains tiny on CPU, and is architected for multi-GPU/petabyte scale.
- Full **MLOps hygiene**: typed, tested (incl. a CPU DDP test + a checkpoint-resume
  reproducibility test), and a dedicated CI job.

### Non-goals / deferred to future work (leaner first cut)
- **TensorStore** (Google) backend — provided as a documented extension point, not built.
- **Ray** executor — documented + lazy interface, but only Local + Dask are built/tested.
- **FSDP** — DDP is implemented; FSDP is documented as the multi-GPU sharding path.
- **K8s serving Deployment** and a **RayCluster** manifest — deferred; we ship the training Job.
- No real GPU, real cluster, or real petabyte data (synthetic, CPU, small — scale-ready code).

## 3. Guiding principles

1. **Additive & optional.** The agentic pipeline still runs with zero ML deps; the platform
   lives behind `[ml]` and never changes the keyless default path.
2. **Light/depless defaults, real backends optional.** NumpyStore + JSONTracker + LocalExecutor
   need only torch+numpy; Zarr/HDF5/Dask/MLflow are real alternatives, lazily imported.
3. **Deterministic & reproducible.** Seed everything; CPU-deterministic; resume reproduces the
   training trajectory (a tested invariant).
4. **Abstractions others depend on.** `ArrayStore`, `Executor`, `Tracker` are small protocols;
   `TorchEmbedder` implements the existing `Embedder`.
5. **Observability & reliability first.** Throughput/loss/resource metrics; atomic, signal-safe,
   resumable checkpoints — the "continuity of large training runs" the JD calls for.

## 4. Module layout

```
src/discovery_agents/mlinfra/
  __init__.py
  config.py            # ModelConfig, DataConfig, TrainConfig (env + CLI driven)
  tokenizer.py         # WordVocab: deterministic vocab built from a corpus; encode/decode/pad
  store/
    base.py            # ArrayStore protocol (create/append/read slices + metadata) + StoreInfo
    numpy_store.py     # NumpyStore (default; .npy/.npz via numpy)
    zarr_store.py      # ZarrStore (lazy zarr)
    hdf5_store.py      # HDF5Store (lazy h5py)
    factory.py         # open_store(backend, path, mode)
  curation/
    base.py            # Executor protocol; CorpusCurator (docs -> token rows -> ArrayStore)
    executors.py       # LocalExecutor (sequential), DaskExecutor (lazy)
    synthetic.py       # deterministic synthetic-corpus generator (scale without real data)
  data/
    dataset.py         # ArrayStoreDataset (map-style over token rows)
    loader.py          # build_dataloader (collate, pin_memory, prefetch, device move)
  model/
    encoder.py         # TextEncoder: token emb + TransformerEncoder + masked mean-pool -> vec
    losses.py          # info_nce (SimCSE-style in-batch contrastive)
  train/
    distributed.py     # init/teardown process group (gloo/nccl), DDP wrap, rank/world helpers, spawn launcher
    checkpoint.py      # atomic save/load {model,opt,sched,step,rng}; latest(); SIGTERM handler
    loop.py            # Trainer: dataloader -> dropout-positive pairs -> InfoNCE -> step; metrics; ckpt; tracker
  tracking/
    base.py            # Tracker protocol; JSONTracker (default; writes runs/*.json + artifacts)
    mlflow_tracker.py  # MLflowTracker (lazy)
  bench/
    io_benchmark.py    # read throughput/latency (MB/s, samples/s, p50/p95) across backends
    profile.py         # psutil/timer resource sampler; optional torch.profiler hook
  embedder.py          # TorchEmbedder(Embedder): load encoder+vocab, embed/embed_batch (eval, no_grad)
  cli.py               # subcommands: curate | train | bench | export
deploy/
  Dockerfile           # CPU training image (installs .[ml,dask])
  docker-compose.yml   # local stack: trainer + mlflow
  k8s/train-job.yaml   # indexed Job running torchrun/DDP across N workers
  Makefile             # curate/train/bench/docker targets
```

## 5. Key interfaces

```python
# store/base.py
class ArrayStore(Protocol):
    def create(self, name: str, shape: tuple[int, ...], dtype: str) -> None: ...
    def append(self, name: str, rows: "np.ndarray") -> None: ...
    def read(self, name: str, start: int, end: int) -> "np.ndarray": ...
    def length(self, name: str) -> int: ...
    def close(self) -> None: ...

# curation/base.py
class Executor(Protocol):
    def map(self, fn: Callable[[T], R], items: list[T]) -> list[R]: ...

# tracking/base.py
class Tracker(Protocol):
    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict, step: int) -> None: ...
    def log_artifact(self, path: str) -> None: ...

# embedder.py — implements retrieval.embeddings.Embedder
class TorchEmbedder:
    dim: int
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

## 6. Model & training

- **Encoder**: token embedding → `nn.TransformerEncoder` (small: e.g. dim 128, 2 layers, 4
  heads) → masked mean-pool → L2-normalized embedding. Config-driven; tiny by default.
- **Objective**: unsupervised **SimCSE** — encode each text twice under independent dropout to
  form positive pairs; **InfoNCE** with in-batch negatives. No labels needed.
- **Trainer**: custom loop — batch from the GPU-native loader → build dropout positives →
  InfoNCE → backward → grad-clip → optimizer/scheduler step → log metrics → periodic checkpoint.
- **Distributed**: `torch.distributed` with `gloo` (CPU) / `nccl` (GPU); `DDP`-wrapped model; a
  `spawn` launcher for local multi-process and `torchrun`-compatible env for K8s.
- **Reliability**: `checkpoint.py` writes `{model, optimizer, scheduler, step, torch/numpy/py
  RNG}` atomically (temp file + rename), keeps `latest`, installs a SIGTERM handler to checkpoint
  on preemption, and resumes exactly.

## 7. Data flow

`curate` (Executor maps docs → tokenized rows → `ArrayStore.append`) → `train` (DDP workers
read row ranges via `ArrayStoreDataset` + loader → SimCSE/InfoNCE → checkpoints → `Tracker`) →
`export` (best checkpoint → `TorchEmbedder` weights + vocab) → the agent pipeline's
`EvidenceIndex` uses `TorchEmbedder` when configured. `bench` compares Numpy/Zarr/HDF5 read perf.

## 8. Integration with the agent pipeline

`retrieval/embeddings.py` already defines `Embedder` and `HashingEmbedder`. Add a
`get_embedder(config)` factory returning `HashingEmbedder` (default, keyless) or `TorchEmbedder`
(when `[ml]` is installed and a trained checkpoint path is configured). `RunConfig` gains an
`embedder` field (`"hashing"` default / `"torch"`); the pipeline passes the chosen embedder to
`EvidenceIndex`. Default behavior is unchanged.

## 9. Dependencies, runtime & CI

- Extras: `[ml]` = `torch, numpy, zarr, h5py, psutil`; `[dask]` = `dask[distributed]`;
  `[mlflow]`; `[ray]`, `[tensorstore]` (declared for future work).
- Defaults (NumpyStore, JSONTracker, LocalExecutor) run on `[ml]` alone. Zarr/HDF5/Dask are
  exercised in CI; MLflow/Ray/TensorStore are lazy + `skipif`.
- New CI job **ml**: install `.[ml,dask]`, run the `mlinfra` tests (incl. a 2-process `gloo`
  DDP smoke test and a checkpoint-resume reproducibility test) on Python 3.11–3.12, CPU. The
  existing keyless job is unchanged.

## 10. Testing strategy

- `test_store.py` (numpy/zarr/hdf5 round-trip + slicing parity), `test_tokenizer.py`,
  `test_curation.py` (Local vs Dask produce identical archives), `test_dataset_loader.py`,
  `test_model.py` (shapes, determinism), `test_losses.py` (InfoNCE properties),
  `test_trainer.py` (loss decreases on tiny overfit data), `test_checkpoint.py` (resume
  reproduces the next-step loss exactly), `test_ddp.py` (2-proc gloo: ranks converge / grads
  sync; tiny), `test_torch_embedder.py` (Embedder conformance + EvidenceIndex integration),
  `test_tracking.py` (JSONTracker), `test_io_benchmark.py` (runs, returns metrics).
- All run CPU, seeded, small. Heavy backends (`tensorstore`, `ray`, `mlflow`) are `skipif`.

## 11. Biohub JD → repo mapping (extends `docs/role-mapping.md`)

| JD requirement | Where |
|---|---|
| PyTorch, custom loops, distributed, low-level perf | `train/loop.py`, `train/distributed.py` |
| Reliability/continuity of large training runs | `train/checkpoint.py` (atomic, resumable, SIGTERM-safe) |
| GPU-native data I/O; Zarr/HDF5/TensorStore; multi-dim tensors | `store/`, `data/loader.py` |
| I/O performance benchmarking at scale | `bench/io_benchmark.py` |
| Distributed computing (Spark/Dask/Ray) | `curation/executors.py` (Dask + Local; Ray future) |
| Docker/Kubernetes | `deploy/Dockerfile`, `deploy/k8s/train-job.yaml` |
| MLOps / lifecycle / artifact tracking / monitoring | `tracking/`, `bench/profile.py`, CI |
| Abstractions others depend on | `ArrayStore` / `Executor` / `Tracker` / `Embedder` |
| AI agent frameworks (plus) | the agentic pipeline consumes the trained embedder |

## 12. Staged build (each stage stays green)

0. `[ml]` extras + install light deps + ML CI job + `mlinfra` skeleton + config
1. Tokenizer + ArrayStore (Numpy/Zarr/HDF5) + tests
2. Curation (Local + Dask, synthetic corpus) → writes archive + tests
3. Dataset + GPU-native DataLoader + tests
4. Model (encoder + InfoNCE) + forward/loss tests
5. Trainer + checkpoint/resume + DDP (gloo) + tests
6. Tracking (JSON + MLflow) + I/O benchmark + profiler + tests
7. `TorchEmbedder` → wire into retrieval/pipeline (opt-in) + tests
8. Docker + K8s Job + Makefile + docs (`docs/ml-platform.md`, README section, role-mapping) +
   adversarial review + final verification

## 13. Risks & mitigations

- **No GPU / CI flakiness** → CPU `gloo`, tiny models/data, generous tolerances; DDP test uses
  2 procs + few steps; timing metrics reported, never gated.
- **Heavy deps** → defaults are light; Zarr/HDF5/Dask in CI; Ray/TensorStore/MLflow lazy + skipif.
- **Non-determinism breaking tests** → global seeding + torch deterministic; checkpoint-resume
  reproducibility is asserted on next-step loss, not wall-clock.
- **Scope vs readability** → small focused modules; `docs/ml-platform.md` + README keep the
  narrative legible.

## 14. Future work

TensorStore backend; Ray executor + RayCluster manifest; FSDP sharding; K8s serving Deployment +
autoscaling; real tokenizer (BPE) and larger corpora; mixed-precision/AMP on GPU; Prometheus
metrics endpoint.
