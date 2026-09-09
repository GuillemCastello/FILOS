"""Process-budgeted RAM reuse, with independent lifetimes for preview sessions."""

from __future__ import annotations

import itertools
import os
import sys
import threading
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np


def _object_sizes(value: Any) -> dict[int, int]:
    """Count each Python object and NumPy backing allocation once."""
    sizes: dict[int, int] = {}
    pending = [value]
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in sizes:
            continue
        sizes[identity] = sys.getsizeof(item)
        if isinstance(item, np.ndarray):
            if item.base is not None:
                pending.append(item.base)
        elif isinstance(item, memoryview):
            pending.append(item.obj)
        elif isinstance(item, Mapping):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return sizes


class _Budget:
    """One LRU budget for cache-owned payloads across all browser sessions."""

    def __init__(self) -> None:
        self.entries: OrderedDict[tuple[int, str, str], tuple[Any, dict[int, int]]] = OrderedDict()
        self.references: dict[int, tuple[int, int]] = {}
        self.bytes = 0
        self.lock = threading.RLock()

    def remove(self, key: tuple[int, str, str]) -> None:
        _value, sizes = self.entries.pop(key)
        for identity in sizes:
            count, size = self.references[identity]
            if count == 1:
                self.bytes -= size
                del self.references[identity]
            else:
                self.references[identity] = (count - 1, size)

    def drop_scope(self, scope: int) -> None:
        with self.lock:
            for key in list(self.entries):
                if key[0] == scope:
                    self.remove(key)

    def trim(self) -> None:
        gib = float(os.environ.get("FILAMENT_CACHE_GIB", "16"))
        if not np.isfinite(gib) or gib < 0:
            raise ValueError("FILAMENT_CACHE_GIB must be finite and non-negative")
        limit = int(gib * 2**30)
        while self.bytes > limit and self.entries:
            self.remove(next(iter(self.entries)))


_BUDGET = _Budget()
_SCOPE_IDS = itertools.count()


class StageCache:
    """Session-owned stage keys sharing the process-wide byte budget.

    Eviction releases only cache references. Active previews and worker inputs
    remain alive for as long as their callers need them. Stored values must not
    be mutated after insertion, so both dependency keys and sizes remain valid.
    """

    def __init__(self) -> None:
        self.scope = next(_SCOPE_IDS)
        weakref.finalize(self, _BUDGET.drop_scope, self.scope)

    def get(self, stage: str, key: str) -> Any | None:
        identity = (self.scope, stage, key)
        with _BUDGET.lock:
            _BUDGET.trim()
            entry = _BUDGET.entries.get(identity)
            if entry is None:
                return None
            _BUDGET.entries.move_to_end(identity)
            return entry[0]

    def put(self, stage: str, key: str, value: Any) -> None:
        identity = (self.scope, stage, key)
        sizes = _object_sizes(value)
        with _BUDGET.lock:
            if identity in _BUDGET.entries:
                _BUDGET.remove(identity)
            for object_id, size in sizes.items():
                count, previous_size = _BUDGET.references.get(object_id, (0, size))
                _BUDGET.references[object_id] = (count + 1, previous_size)
                if count == 0:
                    _BUDGET.bytes += size
            _BUDGET.entries[identity] = (value, sizes)
            _BUDGET.trim()

    def clear(self) -> None:
        _BUDGET.drop_scope(self.scope)


def stage_cache(container: dict[str, Any]) -> StageCache:
    """Attach a lifetime-managed RAM cache to the existing public dictionary."""
    with _BUDGET.lock:
        cache = container.get("_ram_cache")
        if not isinstance(cache, StageCache):
            cache = StageCache()
            container["_ram_cache"] = cache
        return cache


def cache_memory_info() -> dict[str, int]:
    """Return cache-owned payload bytes, excluding caller-held active results."""
    with _BUDGET.lock:
        return {"payload_bytes": _BUDGET.bytes, "entries": len(_BUDGET.entries)}
