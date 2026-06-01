"""Banking77 loader for the retrieval benchmark.

Offline by default: reads a committed stratified slice so the benchmark and tests run
in CI without a download. ``full=True`` loads the complete split from HuggingFace
``mteb/banking77`` (the ``[benchmark]`` extra) for the headline result.

Banking77 (Casanueva et al., 2020) is CC-BY-4.0; see ``benchmark/data/README.md``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

_HF_NAME = "mteb/banking77"


def _sample_path() -> Path:
    override = os.environ.get("BANKING77_SAMPLE")
    if override:
        return Path(override)
    # src/discovery_agents/mlinfra/datasets.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3] / "benchmark" / "data" / "banking77_sample.csv"


@dataclass
class LabeledSplit:
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class Banking77:
    train: LabeledSplit
    test: LabeledSplit
    label_names: list[str]
    id_to_name: dict[int, str]  # explicit map; label ids may be sparse/non-contiguous

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    def name_for(self, label: int) -> str:
        """Intent name for a label id (do NOT index label_names by id — ids may be sparse)."""
        return self.id_to_name.get(label, str(label))


def load_banking77(full: bool = False) -> Banking77:
    return _load_full() if full else _load_sample()


def _load_sample() -> Banking77:
    path = _sample_path()
    if not path.exists():
        raise FileNotFoundError(f"Banking77 sample not found at {path}")
    splits: dict[str, tuple[list[str], list[int]]] = {"train": ([], []), "test": ([], [])}
    names: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = int(row["label"])
            names[label] = row["label_text"]
            texts, labels = splits[row["split"]]
            texts.append(row["text"])
            labels.append(label)
    return Banking77(
        train=LabeledSplit(*splits["train"]),
        test=LabeledSplit(*splits["test"]),
        label_names=[names[i] for i in sorted(names)],
        id_to_name=dict(names),
    )


def _load_full() -> Banking77:
    from datasets import load_dataset  # lazy: requires the [benchmark] extra

    dataset = load_dataset(_HF_NAME)
    names: dict[int, str] = {}

    def convert(split: str) -> LabeledSplit:
        texts: list[str] = []
        labels: list[int] = []
        for row in dataset[split]:
            label = int(row["label"])
            names[label] = row["label_text"]
            texts.append(row["text"])
            labels.append(label)
        return LabeledSplit(texts, labels)

    train = convert("train")
    test = convert("test")
    return Banking77(
        train=train,
        test=test,
        label_names=[names[i] for i in sorted(names)],
        id_to_name=dict(names),
    )
