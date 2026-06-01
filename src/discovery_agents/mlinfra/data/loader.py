"""GPU-native DataLoader construction.

On CUDA this pins memory and overlaps host->device copies (``non_blocking``); on CPU
the same code path runs unchanged. Batches are deterministic given a seed.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


def build_dataloader(
    dataset: Dataset[torch.Tensor],
    batch_size: int,
    *,
    shuffle: bool = True,
    device: torch.device | str | None = None,
    num_workers: int = 0,
    seed: int | None = None,
) -> DataLoader[torch.Tensor]:
    # Lazy import avoids a data<->train import cycle at module load.
    from ..train.distributed import get_rank, get_world_size, is_distributed

    pin_memory = device is not None and str(device).startswith("cuda")

    if is_distributed():
        # Each rank gets a disjoint shard, so data-parallel workers don't all train on
        # the same data (call sampler.set_epoch(epoch) per epoch to reshuffle).
        from torch.utils.data.distributed import DistributedSampler

        sampler: DistributedSampler[torch.Tensor] = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            seed=seed or 0,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # sampler handles shuffling; DataLoader shuffle must be False
            collate_fn=collate,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=True,
        )

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=pin_memory,
        num_workers=num_workers,
        generator=generator,
        drop_last=True,
    )


def to_device(batch: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Move a batch to the device, overlapping the copy when memory is pinned."""
    return batch.to(device, non_blocking=True)
