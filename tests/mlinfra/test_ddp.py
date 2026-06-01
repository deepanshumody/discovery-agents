"""DDP keeps parameters synchronized across ranks (2-process gloo, CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_ddp_keeps_params_in_sync(tmp_path) -> None:
    from discovery_agents.mlinfra.train import spawn
    from discovery_agents.mlinfra.train._ddp_selftest import run_ddp_selfcheck

    spawn(run_ddp_selfcheck, 2, str(tmp_path), 32)

    checksums = [
        float((tmp_path / f"rank{rank}.txt").read_text(encoding="utf-8")) for rank in range(2)
    ]
    # Different data per rank, but DDP averaged gradients -> identical final params.
    assert abs(checksums[0] - checksums[1]) < 1e-4
