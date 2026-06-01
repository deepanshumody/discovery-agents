"""Lightweight resource profiler (wall time, RSS, CPU%) via psutil."""

from __future__ import annotations

import os
from time import perf_counter
from types import TracebackType
from typing import Any


class ResourceSampler:
    """Context manager capturing wall time and process memory over a code block."""

    def __init__(self) -> None:
        import psutil  # lazy import

        self._process = psutil.Process(os.getpid())
        self._start = 0.0
        self._rss_start = 0
        self.stats: dict[str, Any] = {}

    def __enter__(self) -> ResourceSampler:
        self._process.cpu_percent(None)  # prime the CPU% measurement
        self._rss_start = self._process.memory_info().rss
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        wall = perf_counter() - self._start
        rss = self._process.memory_info().rss
        self.stats = {
            "wall_s": round(wall, 4),
            "rss_mb": round(rss / 1e6, 2),
            "rss_delta_mb": round((rss - self._rss_start) / 1e6, 2),
            "cpu_percent": round(self._process.cpu_percent(None), 1),
        }
