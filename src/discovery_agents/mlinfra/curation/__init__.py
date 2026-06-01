"""Distributed corpus curation (Local + Dask) and synthetic data generation."""

from __future__ import annotations

from .base import CorpusCurator, Executor
from .executors import DaskExecutor, LocalExecutor
from .synthetic import generate_corpus

__all__ = ["CorpusCurator", "DaskExecutor", "Executor", "LocalExecutor", "generate_corpus"]
