# simple circuit breaker implementation
import time
from typing import Callable, Any


class CircuitBreakerOpen(Exception):
    """Raised when the circuit is open and calls are not allowed."""


class CircuitBreaker:
    """Basic circuit breaker for controlling external API calls."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._state = "CLOSED"
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        return self._state

    def call(self, func: Callable[[], Any]) -> Any:
        """Execute a function while enforcing circuit breaker logic."""
        if self._state == "OPEN":
            if (time.time() - self._opened_at) < self.recovery_timeout:
                raise CircuitBreakerOpen("Circuit breaker is open")
            # allow a trial call in half-open state
            self._state = "HALF_OPEN"
        try:
            result = func()
        except Exception:
            self._record_failure()
            raise
        else:
            self._reset()
            return result

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = "OPEN"
            self._opened_at = time.time()

    def _reset(self) -> None:
        self._failures = 0
        self._state = "CLOSED"
