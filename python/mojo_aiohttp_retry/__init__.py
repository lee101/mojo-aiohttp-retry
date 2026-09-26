"""mojo-aiohttp-retry: the backoff-schedule arithmetic of aiohttp-retry in Mojo.

Installable alongside the real ``aiohttp_retry``, which it is tested against
for parity. The retry policies here are drop-in replacements for
``aiohttp_retry.retry_options``; the rest of the client, the aiohttp plumbing
and the URL rotation are untouched upstream code and are not reimplemented.
"""

from ._lib import (
    exponential_schedule,
    fibonacci_schedule,
    jitter_schedule,
    linear_schedule,
)
from .policies import (
    ExponentialRetry,
    FibonacciRetry,
    JitterRetry,
    RandomRetry,
)

__all__ = [
    "ExponentialRetry",
    "FibonacciRetry",
    "JitterRetry",
    "RandomRetry",
    "exponential_schedule",
    "fibonacci_schedule",
    "jitter_schedule",
    "linear_schedule",
]
__version__ = "0.1.0"
