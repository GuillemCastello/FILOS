"""Execution-only limits and reusable pools, separate from physical inputs."""

from __future__ import annotations

import atexit
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from threading import RLock

from threadpoolctl import threadpool_limits

_POOL_LOCK = RLock()
_PROCESS_POOLS: dict[tuple[str, int], ProcessPoolExecutor] = {}
_THREAD_POOLS: dict[tuple[str, int], ThreadPoolExecutor] = {}
_NATIVE_THREAD_LIMIT = None


def integer_setting(name: str, default: int, *, minimum: int = 1) -> int:
    """Read an integer environment override with an explicit valid range."""
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer; received {raw!r}") from error
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; received {value}")
    return value


def in_execution_worker() -> bool:
    """Prevent recursive process pools in a spawned scientific worker."""
    return multiprocessing.current_process().name != "MainProcess"


def process_spawn_available() -> bool:
    """Allow package tasks in notebooks; avoid reloading placeholder paths.

    Without ``__main__.__file__``, spawn leaves the child's main module empty;
    our workers remain importable from this package. A placeholder filename
    such as ``<stdin>`` instead requests a reload that cannot succeed, so those
    sessions keep serial execution. Spawned workers never create nested pools.
    """
    main_path = getattr(sys.modules.get("__main__"), "__file__", None)
    return (main_path is None or Path(main_path).is_file()) and not in_execution_worker()


def _initialize_process_worker() -> None:
    """Limit native kernels after NumPy/SciPy imports in a spawned worker."""
    global _NATIVE_THREAD_LIMIT
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    _NATIVE_THREAD_LIMIT = threadpool_limits(limits=1)


def process_pool(name: str, workers: int) -> ProcessPoolExecutor:
    """Reuse a named spawn pool; callers preserve result order explicitly."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if in_execution_worker():
        raise RuntimeError("nested scientific process pools are unsupported")
    key = (name, workers)
    with _POOL_LOCK:
        pool = _PROCESS_POOLS.get(key)
        if pool is None:
            pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_process_worker,
            )
            _PROCESS_POOLS[key] = pool
        return pool


def thread_pool(name: str, workers: int) -> ThreadPoolExecutor:
    """Reuse a named thread pool for bounded independent rendering work."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    key = (name, workers)
    with _POOL_LOCK:
        pool = _THREAD_POOLS.get(key)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
            _THREAD_POOLS[key] = pool
        return pool


def discard_process_pool(name: str, workers: int, failed_pool: ProcessPoolExecutor) -> None:
    """Evict a failed pool without removing a concurrent replacement."""
    with _POOL_LOCK:
        key = (name, workers)
        if _PROCESS_POOLS.get(key) is failed_pool:
            del _PROCESS_POOLS[key]
    failed_pool.shutdown(wait=False, cancel_futures=True)


def shutdown_execution_pools() -> None:
    """Release persistent workers after experiments or during interpreter exit."""
    with _POOL_LOCK:
        pools = list(_PROCESS_POOLS.values()) + list(_THREAD_POOLS.values())
        _PROCESS_POOLS.clear()
        _THREAD_POOLS.clear()
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)


atexit.register(shutdown_execution_pools)
