"""Executors: sequential (Local) and distributed (Dask).

Ray is a documented future backend (see the spec); only Local + Dask are built.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


class LocalExecutor:
    """Sequential, dependency-free executor (the default)."""

    def map(self, fn: Callable[[T], R], items: list[T]) -> list[R]:
        return [fn(item) for item in items]


class DaskExecutor:
    """Dask-bag executor: partitions the work into a task graph.

    Uses the threaded scheduler by default (reliable + fast in CI); point it at a
    ``distributed.Client`` for a real multi-node cluster. Order is preserved.
    """

    def __init__(self, npartitions: int = 4, scheduler: str = "threads") -> None:
        self.npartitions = npartitions
        self.scheduler = scheduler

    def map(self, fn: Callable[[T], R], items: list[T]) -> list[R]:
        import dask.bag as db  # lazy import

        bag = db.from_sequence(items, npartitions=self.npartitions)
        result = bag.map(fn).compute(scheduler=self.scheduler)
        return list(result)
