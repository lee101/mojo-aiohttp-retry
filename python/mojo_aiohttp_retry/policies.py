"""Retry policies whose backoff schedules are computed by the Mojo kernels.

Each class mirrors the constructor and ``get_timeout`` contract of the matching
``aiohttp_retry.retry_options`` class, so it can be handed straight to
``aiohttp_retry.RetryClient``. The difference is that the whole schedule is
computed once, at construction time, by one call into the compiled library,
instead of being recomputed one element at a time on the retry path.

``aiohttp_retry`` consumes one timeout per ``get_timeout`` call, in call order,
so ``_schedule`` is a flat array of length ``attempts`` in that same order and
``get_timeout(attempt)`` is ``_schedule[attempt - 1]``. Where the kernel's
natural column index is offset from the call index - the exponential and
jittered schedules, whose upstream exponent is the 1-based attempt number - the
policy computes the extra leading column and drops it.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Iterable

import numpy as np

from . import _lib

_DEFAULT_METHODS = frozenset(
    {"HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST", "CONNECT", "PATCH"}
)


class _BasePolicy:
    """The attribute surface of `aiohttp_retry.RetryOptionsBase`.

    Status, exception and method handling is set up exactly as upstream does;
    only ``get_timeout`` differs, and only in where the number comes from.
    """

    __slots__ = (
        "attempts", "statuses", "exceptions", "methods",
        "retry_all_server_errors", "evaluate_response_callback", "_schedule",
    )

    def __init__(
        self,
        attempts: int = 3,
        statuses: Iterable[int] | None = None,
        exceptions: Iterable[type[Exception]] | None = None,
        methods: Iterable[str] | None = None,
        retry_all_server_errors: bool = True,
        evaluate_response_callback: Callable[..., Any] | None = None,
    ) -> None:
        if attempts < 0:
            raise ValueError("attempts must be non-negative")
        self.attempts = attempts
        self.statuses = set() if statuses is None else statuses
        self.exceptions = set() if exceptions is None else exceptions
        self.methods = (
            _DEFAULT_METHODS if methods is None else {m.upper() for m in methods}
        )
        self.retry_all_server_errors = retry_all_server_errors
        self.evaluate_response_callback = evaluate_response_callback
        self._schedule = np.zeros(0, dtype=np.float64)

    @property
    def schedule(self) -> np.ndarray:
        """The precomputed backoff schedule, in ``get_timeout`` call order."""
        return self._schedule

    def get_timeout(self, attempt: int, response: Any = None) -> float:
        if attempt < 1 or attempt > self.attempts:
            raise IndexError(
                f"attempt {attempt} outside 1..{self.attempts} for {type(self).__name__}"
            )
        return float(self._schedule[attempt - 1])


class ExponentialRetry(_BasePolicy):
    """``min(start_timeout * factor ** attempt, max_timeout)`` per attempt.

    Upstream passes the 1-based attempt number as the exponent, so the leading
    attempt-0 column (which is just ``start_timeout``) is computed and dropped.
    """

    __slots__ = ("start_timeout", "max_timeout", "factor")

    def __init__(
        self,
        attempts: int = 3,
        start_timeout: float = 0.1,
        max_timeout: float = 30.0,
        factor: float = 2.0,
        statuses: Iterable[int] | None = None,
        exceptions: Iterable[type[Exception]] | None = None,
        methods: Iterable[str] | None = None,
        retry_all_server_errors: bool = True,
        evaluate_response_callback: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            attempts=attempts, statuses=statuses, exceptions=exceptions, methods=methods,
            retry_all_server_errors=retry_all_server_errors,
            evaluate_response_callback=evaluate_response_callback,
        )
        self.start_timeout = float(start_timeout)
        self.max_timeout = float(max_timeout)
        self.factor = float(factor)
        self._schedule = _lib.exponential_schedule(
            self.start_timeout, self.factor, self.max_timeout, attempts + 1
        )[0, 1:]


class FibonacciRetry(_BasePolicy):
    """``multiplier * F(call + 2)`` clamped to ``max_timeout``.

    Upstream is stateful: each ``get_timeout`` advances ``prev_step`` and
    ``current_step`` and ignores the attempt index, so the same object called
    twice in a row returns the next Fibonacci term and out-of-order or repeated
    calls are not addressable. Here the whole sequence is computed up front, so
    ``get_timeout(attempt)`` is idempotent and index-addressed.

    The kernel's column ``a`` is already the ``a + 1``-th call, so no leading
    column has to be dropped here.
    """

    __slots__ = ("multiplier", "max_timeout")

    def __init__(
        self,
        attempts: int = 3,
        multiplier: float = 1.0,
        statuses: Iterable[int] | None = None,
        exceptions: Iterable[type[Exception]] | None = None,
        methods: Iterable[str] | None = None,
        max_timeout: float = 3.0,
        retry_all_server_errors: bool = True,
        evaluate_response_callback: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            attempts=attempts, statuses=statuses, exceptions=exceptions, methods=methods,
            retry_all_server_errors=retry_all_server_errors,
            evaluate_response_callback=evaluate_response_callback,
        )
        self.multiplier = float(multiplier)
        self.max_timeout = float(max_timeout)
        self._schedule = _lib.fibonacci_schedule(
            self.multiplier, self.max_timeout, attempts
        )[0]


class JitterRetry(_BasePolicy):
    """Exponential backoff plus ``uniform(0, size) ** factor`` per attempt.

    The jitter draws are taken from ``random.uniform`` at construction time, one
    per schedule column. The exponential term upstream evaluates is
    ``start * factor ** attempt`` with the 1-based attempt number, so column
    ``attempt`` is the one that is kept and the leading attempt-0 column is
    dropped - which consumes exactly ``attempts`` draws for ``attempts``
    timeouts, in the order ``get_timeout`` would have consumed them.

    Mojo's ``pow`` for a non-integer exponent is LLVM's vectorised routine rather
    than libm's scalar one, so the jitter term agrees with upstream to about
    4e-9 relative rather than bit for bit.
    """

    __slots__ = ("start_timeout", "max_timeout", "factor", "random_interval_size")
    def __init__(
        self,
        attempts: int = 3,
        start_timeout: float = 0.1,
        max_timeout: float = 30.0,
        factor: float = 2.0,
        statuses: Iterable[int] | None = None,
        exceptions: Iterable[type[Exception]] | None = None,
        methods: Iterable[str] | None = None,
        random_interval_size: float = 2.0,
        retry_all_server_errors: bool = True,
        evaluate_response_callback: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            attempts=attempts, statuses=statuses, exceptions=exceptions, methods=methods,
            retry_all_server_errors=retry_all_server_errors,
            evaluate_response_callback=evaluate_response_callback,
        )
        self.start_timeout = float(start_timeout)
        self.max_timeout = float(max_timeout)
        self.factor = float(factor)
        self.random_interval_size = float(random_interval_size)
        size = self.random_interval_size
        # draws[a] is the jitter for kernel column a. Column 0 is the dropped
        # attempt-0 slot, so it is filled with a dummy and the real draws land
        # on columns 1..attempts: one per get_timeout call, in call order.
        draws = np.empty(attempts + 1, dtype=np.float64)
        draws[0] = 0.0
        draws[1:] = [random.uniform(0.0, size) for _ in range(attempts)]
        self._schedule = _lib.jitter_schedule(
            self.start_timeout, self.factor, self.max_timeout,
            draws, attempts + 1,
        )[0, 1:]


class RandomRetry(_BasePolicy):
    """``min_timeout + random() * (max_timeout - min_timeout)``, one draw per attempt.

    Upstream ignores the attempt index and draws fresh randomness on every call,
    so the value for a given attempt is not a function of the attempt; the draws
    are made once here, in call order, and indexed thereafter.
    """

    __slots__ = ("min_timeout", "max_timeout")

    def __init__(
        self,
        attempts: int = 3,
        statuses: Iterable[int] | None = None,
        exceptions: Iterable[type[Exception]] | None = None,
        methods: Iterable[str] | None = None,
        min_timeout: float = 0.1,
        max_timeout: float = 3.0,
        retry_all_server_errors: bool = True,
        evaluate_response_callback: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            attempts=attempts, statuses=statuses, exceptions=exceptions, methods=methods,
            retry_all_server_errors=retry_all_server_errors,
            evaluate_response_callback=evaluate_response_callback,
        )
        self.min_timeout = float(min_timeout)
        self.max_timeout = float(max_timeout)
        draws = np.array([random.random() for _ in range(attempts)], dtype=np.float64)
        self._schedule = _lib.linear_schedule(
            self.min_timeout, self.max_timeout, draws
        )[0]
