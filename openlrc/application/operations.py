"""Cross-surface guard for long-running local operations."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActiveOperation:
    kind: str
    label: str


class OperationGuard:
    """Allow one Workflow or Setup operation to own mutable local resources."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: ActiveOperation | None = None

    @property
    def active(self) -> ActiveOperation | None:
        with self._lock:
            return self._active

    @contextmanager
    def acquire(self, kind: str, label: str) -> Iterator[None]:
        with self._lock:
            if self._active is not None:
                raise RuntimeError(f"Cannot start {label}; {self._active.label} is already running.")
            self._active = ActiveOperation(kind, label)
        try:
            yield
        finally:
            with self._lock:
                self._active = None
