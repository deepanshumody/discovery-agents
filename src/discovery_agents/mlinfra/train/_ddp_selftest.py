"""Importable DDP worker for the test suite.

Lives in the package (not the test module) so ``mp.spawn`` can re-import it by its
qualified name under the default spawn start method. Each rank trains on *different*
data; because DDP all-reduces gradients, all ranks must end with identical parameters.
"""

from __future__ import annotations

import os

import torch

from ..config import ModelConfig
from ..model.encoder import TextEncoder
from ..model.losses import info_nce
from .distributed import cleanup, maybe_ddp, set_seed, setup


def run_ddp_selfcheck(rank: int, world_size: int, out_dir: str, vocab_size: int = 32) -> None:
    setup(rank, world_size, backend="gloo")
    try:
        set_seed(0)  # identical initial parameters on every rank
        model = TextEncoder(
            ModelConfig(dim=16, num_layers=1, num_heads=4, max_seq_len=8, vocab_size=vocab_size)
        )
        ddp = maybe_ddp(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        torch.manual_seed(100 + rank)  # different data per rank
        batch = torch.randint(1, vocab_size, (4, 8))
        ddp.train()
        for _ in range(3):
            z1 = ddp(batch)
            z2 = ddp(batch)
            loss = info_nce(z1, z2, temperature=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        checksum = sum(float(p.detach().sum()) for p in model.parameters())
        with open(os.path.join(out_dir, f"rank{rank}.txt"), "w", encoding="utf-8") as handle:
            handle.write(repr(checksum))
    finally:
        cleanup()
