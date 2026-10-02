"""Small process-local guard for memory-bound local AI inference."""

from contextlib import contextmanager
from threading import BoundedSemaphore
from typing import Iterator

from app.api.errors import AIConcurrencyLimitError


class AIConcurrencyGuard:
    def __init__(self, *, limit: int, queue_timeout_seconds: float) -> None:
        if limit < 1:
            raise ValueError("AI concurrency limit must be at least 1")
        if queue_timeout_seconds < 0:
            raise ValueError("AI queue timeout must not be negative")
        self._semaphore = BoundedSemaphore(limit)
        self._queue_timeout_seconds = queue_timeout_seconds

    @contextmanager
    def slot(self) -> Iterator[None]:
        acquired = self._semaphore.acquire(timeout=self._queue_timeout_seconds)
        if not acquired:
            raise AIConcurrencyLimitError(
                "local AI capacity is busy; retry the request later"
            )
        try:
            yield
        finally:
            self._semaphore.release()
