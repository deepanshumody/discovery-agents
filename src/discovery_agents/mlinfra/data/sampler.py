"""Label-aware batch sampler for supervised contrastive training.

Each batch draws ``classes_per_batch`` distinct labels and ``samples_per_class``
examples per label, guaranteeing in-batch positives exist for the SupCon loss.
Deterministic given a seed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator


class LabelBatchSampler:
    def __init__(
        self,
        labels: list[int],
        *,
        classes_per_batch: int = 16,
        samples_per_class: int = 4,
        steps: int = 200,
        seed: int = 7,
    ) -> None:
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.by_label[label].append(index)
        self.label_pool = [lbl for lbl, idxs in self.by_label.items() if idxs]
        self.classes_per_batch = min(classes_per_batch, len(self.label_pool))
        self.samples_per_class = samples_per_class
        self.steps = steps
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.steps):
            chosen = self._rng.sample(self.label_pool, self.classes_per_batch)
            batch: list[int] = []
            for label in chosen:
                pool = self.by_label[label]
                if len(pool) >= self.samples_per_class:
                    batch.extend(self._rng.sample(pool, self.samples_per_class))
                else:  # small class: sample with replacement so positives still exist
                    batch.extend(self._rng.choices(pool, k=self.samples_per_class))
            yield batch
