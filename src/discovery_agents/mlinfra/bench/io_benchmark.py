"""I/O benchmark: compare read throughput/latency across archive backends.

Reports write time, read throughput (samples/s, MB/s) and read-latency percentiles —
the "format evaluation / I/O benchmarking at scale" the role calls for. Timings are
reported, never gated (they vary by machine).
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from ..store import open_store

_SUFFIX = {"zarr": ".zarr", "hdf5": ".h5"}


def benchmark_backend(
    backend: str,
    path: str,
    *,
    rows: int = 2048,
    cols: int = 64,
    batch_size: int = 64,
    num_batches: int = 20,
) -> dict[str, Any]:
    gen = np.random.default_rng(0)
    data = gen.integers(0, 1000, size=(rows, cols)).astype("int64")

    store = open_store(backend, path, mode="w")
    store.create("x", (0, cols), "int64")
    start = perf_counter()
    store.append("x", data)
    store.close()
    write_s = perf_counter() - start

    reader = open_store(backend, path, mode="r")
    latencies_ms: list[float] = []
    total_bytes = 0
    sampler = np.random.default_rng(1)
    for _ in range(num_batches):
        offset = int(sampler.integers(0, max(1, rows - batch_size)))
        t0 = perf_counter()
        chunk = reader.read("x", offset, offset + batch_size)
        latencies_ms.append((perf_counter() - t0) * 1000.0)
        total_bytes += int(np.asarray(chunk).nbytes)
    reader.close()

    read_seconds = sum(latencies_ms) / 1000.0
    samples = num_batches * batch_size
    return {
        "backend": backend,
        "write_s": round(write_s, 4),
        "read_samples_per_s": round(samples / read_seconds, 1) if read_seconds > 0 else 0.0,
        "read_mb_per_s": round(total_bytes / 1e6 / read_seconds, 2) if read_seconds > 0 else 0.0,
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 4),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 4),
    }


def run(backends: list[str], tmp_dir: str, **kwargs: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for backend in backends:
        path = f"{tmp_dir}/bench_{backend}{_SUFFIX.get(backend, '')}"
        try:
            results.append(benchmark_backend(backend, path, **kwargs))
        except Exception as exc:  # a missing optional backend shouldn't abort the sweep
            results.append({"backend": backend, "error": f"{type(exc).__name__}: {exc}"})
    return results
