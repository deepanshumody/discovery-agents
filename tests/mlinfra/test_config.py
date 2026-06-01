"""Config tests (torch-free; run in every CI job)."""

from __future__ import annotations

from discovery_agents.mlinfra import DataConfig, ModelConfig, TrainConfig


def test_default_configs_are_sane() -> None:
    model = ModelConfig()
    assert model.dim % model.num_heads == 0  # heads must divide the model dim
    assert model.max_seq_len > 0
    data = DataConfig()
    assert data.store_backend in {"numpy", "zarr", "hdf5"}
    assert data.executor in {"local", "dask"}


def test_smoke_config_is_tiny() -> None:
    smoke = TrainConfig.smoke()
    assert smoke.steps <= 5
    assert smoke.checkpoint_every <= smoke.steps or smoke.checkpoint_every == smoke.steps
