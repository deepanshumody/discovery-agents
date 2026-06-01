"""CLI for the ML-infra platform: curate | train | bench | export.

python -m discovery_agents.mlinfra.cli curate
python -m discovery_agents.mlinfra.cli train
python -m discovery_agents.mlinfra.cli train --smoke     # tiny end-to-end (curate+train+export)
python -m discovery_agents.mlinfra.cli bench
python -m discovery_agents.mlinfra.cli export
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from .config import DataConfig, ModelConfig, TrainConfig
from .curation import CorpusCurator, DaskExecutor, Executor, LocalExecutor, generate_corpus
from .store import open_store
from .tokenizer import WordVocab


def _paths(base: str) -> tuple[str, str]:
    """Return (store_path, checkpoint_dir) under the artifact base directory."""
    return str(Path(base) / "corpus"), str(Path(base) / "checkpoints")


def cmd_curate(data: DataConfig, base: str) -> int:
    store_path, ckpt_dir = _paths(base)
    docs = generate_corpus(data.corpus_size, seed=data.seed)
    vocab = WordVocab.build(docs)
    # cast: the ternary's join widens to `object`; both branches satisfy Executor.
    executor = cast("Executor", DaskExecutor() if data.executor == "dask" else LocalExecutor())
    store = open_store(data.store_backend, store_path, mode="w")
    rows = CorpusCurator(vocab, data.seq_len, executor).run(docs, store)
    store.close()
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    vocab.save(Path(ckpt_dir) / "vocab.json")
    print(f"Curated {rows} docs -> {data.store_backend} store at {store_path} (vocab {len(vocab)})")
    return 0


def cmd_train(data: DataConfig, model: ModelConfig, train: TrainConfig, base: str) -> int:
    import torch  # lazy

    from .data import ArrayStoreDataset, build_dataloader
    from .model import TextEncoder
    from .tracking import JSONTracker
    from .train import Trainer, has_checkpoint, load_checkpoint, set_seed

    store_path, ckpt_dir = _paths(base)
    vocab = WordVocab.load(Path(ckpt_dir) / "vocab.json")
    store = open_store(data.store_backend, store_path, mode="r")
    loader = build_dataloader(ArrayStoreDataset(store), batch_size=data.batch_size, seed=data.seed)

    model.vocab_size = len(vocab)
    model.max_seq_len = max(model.max_seq_len, data.seq_len)
    set_seed(train.seed)
    encoder = TextEncoder(model)
    train.checkpoint_dir = ckpt_dir
    tracker = JSONTracker(run_dir=str(Path(base) / "runs"), run_name="train")
    tracker.log_params({**asdict(train), **asdict(model), "vocab_size": len(vocab)})

    trainer = Trainer(
        encoder, train, device="cuda" if torch.cuda.is_available() else "cpu", tracker=tracker
    )
    if has_checkpoint(ckpt_dir):
        trainer.step = load_checkpoint(
            ckpt_dir, model=trainer.module, optimizer=trainer.optimizer, scheduler=trainer.scheduler
        )
        print(f"Resumed from step {trainer.step}")
    result = trainer.fit(loader)
    trainer.save()
    config_json = {**asdict(model), "seq_len": data.seq_len}
    (Path(ckpt_dir) / "model_config.json").write_text(json.dumps(config_json), encoding="utf-8")
    print(f"Trained to step {result['step']}, final loss {result['final_loss']:.4f}")
    return 0


def cmd_bench(data: DataConfig, base: str) -> int:
    from .bench import run

    backends = ["numpy"]
    for name, backend in [("zarr", "zarr"), ("h5py", "hdf5")]:
        try:
            __import__(name)
            backends.append(backend)
        except ImportError:
            pass
    results = run(backends, str(Path(base) / "bench"), rows=4096, cols=data.seq_len)
    print(f"{'backend':8} {'write_s':>8} {'samples/s':>12} {'MB/s':>8} {'p50_ms':>8} {'p95_ms':>8}")
    for r in results:
        if "error" in r:
            print(f"{r['backend']:8} ERROR: {r['error']}")
        else:
            print(
                f"{r['backend']:8} {r['write_s']:>8} {r['read_samples_per_s']:>12} "
                f"{r['read_mb_per_s']:>8} {r['p50_ms']:>8} {r['p95_ms']:>8}"
            )
    return 0


def cmd_export(base: str) -> int:
    from .embedder import TorchEmbedder

    _, ckpt_dir = _paths(base)
    embedder = TorchEmbedder.from_pretrained(ckpt_dir)
    sample = embedder.embed("evidence-grounded enterprise agent workflow")
    norm = sum(x * x for x in sample) ** 0.5
    print(f"Exported TorchEmbedder (dim {embedder.dim}); sample embedding L2-norm {norm:.4f}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="discovery-agents ML-infra platform")
    parser.add_argument("command", choices=["curate", "train", "bench", "export"])
    parser.add_argument("--artifacts", default="outputs/ml", help="Artifact base directory.")
    parser.add_argument("--backend", default="numpy", help="Store backend: numpy|zarr|hdf5.")
    parser.add_argument("--executor", default="local", help="Curation executor: local|dask.")
    parser.add_argument("--corpus-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny end-to-end run (curate+train+export)."
    )
    args = parser.parse_args(argv)

    data = DataConfig(
        corpus_size=16 if args.smoke else args.corpus_size,
        store_backend=args.backend,
        executor=args.executor,
        batch_size=4 if args.smoke else 32,
        seq_len=16,
    )
    model = (
        ModelConfig(dim=32, num_layers=2, num_heads=4, max_seq_len=16)
        if args.smoke
        else ModelConfig()
    )
    train = TrainConfig.smoke() if args.smoke else TrainConfig(steps=args.steps)

    if args.command == "curate":
        raise SystemExit(cmd_curate(data, args.artifacts))
    if args.command == "bench":
        raise SystemExit(cmd_bench(data, args.artifacts))
    if args.command == "export":
        raise SystemExit(cmd_export(args.artifacts))
    # train (and --smoke runs the full curate -> train -> export chain)
    if args.smoke:
        cmd_curate(data, args.artifacts)
    cmd_train(data, model, train, args.artifacts)
    if args.smoke:
        cmd_export(args.artifacts)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
