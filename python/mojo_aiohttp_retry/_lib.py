"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay ``c_int64`` for addresses; ``c_int``
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-aiohttp-retry.so"

_A = ctypes.c_int64
_I = ctypes.c_int64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(f"{_LIB_PATH} not found; run `bash build/build.sh` first")
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.ahr_exponential_schedule.restype = None
    lib.ahr_exponential_schedule.argtypes = [_A, _A, _A, _I, _I, _A]
    lib.ahr_fibonacci_schedule.restype = None
    lib.ahr_fibonacci_schedule.argtypes = [_A, _A, _I, _I, _A]
    lib.ahr_jitter_schedule.restype = None
    lib.ahr_jitter_schedule.argtypes = [_A, _A, _A, _A, _I, _I, _A]
    lib.ahr_linear_schedule.restype = None
    lib.ahr_linear_schedule.argtypes = [_A, _A, _A, _I, _I, _A]
    return lib


lib = _load()


def _v(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


# Bound once: a CDLL attribute lookup costs ~1.6us per call, which is material
# for the single-plan path where the whole call is under 10us.
_exponential = lib.ahr_exponential_schedule
_fibonacci = lib.ahr_fibonacci_schedule
_jitter = lib.ahr_jitter_schedule
_linear = lib.ahr_linear_schedule


def _plan_buffer(a, b, c, n: int) -> np.ndarray:
    """One allocation holding the three plan scalars then the schedule.

    A single retry policy is the common case, and on this box a six-argument
    ctypes call is ~5us, so the surrounding Python allocations have to be kept
    to one to stay in the same league as the pure-Python policy.
    """
    buf = np.empty(3 + n, dtype=np.float64)
    buf[0] = a
    buf[1] = b
    buf[2] = c
    return buf


def _plan_args(starts, factors, max_timeouts):
    starts = _v(starts)
    factors = _v(factors)
    max_timeouts = _v(max_timeouts)
    if starts.ndim != 1 or factors.shape != starts.shape:
        raise ValueError("starts and factors must be 1-D and equal length")
    if max_timeouts.shape != starts.shape:
        raise ValueError("max_timeouts must match starts")
    return starts, factors, max_timeouts


def exponential_schedule(starts, factors, max_timeouts, n_attempts: int) -> np.ndarray:
    """Return the ``(nplans, n_attempts)`` exponential backoff schedule.

    ``result[p, a] == min(starts[p] * factors[p] ** a, max_timeouts[p])``, the
    value ``aiohttp_retry.ExponentialRetry.get_timeout`` produces for
    ``attempt == a``.
    """
    n_attempts = int(n_attempts)
    if n_attempts < 0:
        raise ValueError("n_attempts must be non-negative")
    if np.isscalar(starts):
        buf = _plan_buffer(starts, factors, max_timeouts, n_attempts)
        if n_attempts:
            base = buf.ctypes.data
            _exponential(base, base + 8, base + 16, 1, n_attempts, base + 24)
        return buf[3:].reshape(1, n_attempts)
    starts, factors, max_timeouts = _plan_args(starts, factors, max_timeouts)
    nplans = starts.size
    res = np.empty((nplans, n_attempts), dtype=np.float64)
    if nplans == 0 or n_attempts == 0:
        return res
    lib.ahr_exponential_schedule(
        _addr(starts), _addr(factors), _addr(max_timeouts),
        nplans, n_attempts, _addr(res),
    )
    return res


def fibonacci_schedule(multipliers, max_timeouts, n_attempts: int) -> np.ndarray:
    """Return the ``(nplans, n_attempts)`` Fibonacci backoff schedule.

    ``result[p, a] == min(multipliers[p] * F(a + 2), max_timeouts[p])``, the
    value returned by the ``a``-th ``get_timeout`` call of
    ``FibonacciRetry`` (which is seeded at 1.0/1.0 and advances before
    returning, so its first call yields 2 * multiplier).
    """
    n_attempts = int(n_attempts)
    if n_attempts < 0:
        raise ValueError("n_attempts must be non-negative")
    if np.isscalar(multipliers):
        buf = _plan_buffer(multipliers, max_timeouts, 0.0, n_attempts)
        if n_attempts:
            _fibonacci(buf.ctypes.data, buf.ctypes.data + 8, 1, n_attempts,
                       buf.ctypes.data + 24)
        return buf[3:].reshape(1, n_attempts)
    multipliers = _v(multipliers)
    max_timeouts = _v(max_timeouts)
    if multipliers.ndim != 1 or max_timeouts.shape != multipliers.shape:
        raise ValueError("multipliers and max_timeouts must be 1-D and equal length")
    nplans = multipliers.size
    res = np.empty((nplans, n_attempts), dtype=np.float64)
    if nplans == 0 or n_attempts == 0:
        return res
    lib.ahr_fibonacci_schedule(
        _addr(multipliers), _addr(max_timeouts), nplans, n_attempts, _addr(res)
    )
    return res


def jitter_schedule(starts, factors, max_timeouts, draws, n_attempts: int) -> np.ndarray:
    """Return the ``(nplans, n_attempts)`` jittered backoff schedule.

    ``result[p, a] == min(starts[p] * factors[p] ** a, max_timeouts[p])
    + draws[p, a] ** factors[p]``, matching
    ``aiohttp_retry.JitterRetry.get_timeout``. ``draws`` holds the absolute
    ``uniform(0, random_interval_size)`` values in row-major order; the
    exponential term is clamped, the jitter term is not, exactly as upstream.
    """
    n_attempts = int(n_attempts)
    if n_attempts < 0:
        raise ValueError("n_attempts must be non-negative")
    if np.isscalar(starts):
        flat = np.ascontiguousarray(_v(draws).reshape(-1))
        if flat.size != n_attempts:
            raise ValueError("draws must have shape (1, n_attempts)")
        buf = _plan_buffer(starts, factors, max_timeouts, n_attempts)
        if n_attempts:
            base = buf.ctypes.data
            _jitter(base, base + 8, base + 16, flat.ctypes.data, 1, n_attempts, base + 24)
        return buf[3:].reshape(1, n_attempts)
    starts, factors, max_timeouts = _plan_args(starts, factors, max_timeouts)
    nplans = starts.size
    res = np.empty((nplans, n_attempts), dtype=np.float64)
    if nplans == 0 or n_attempts == 0:
        return res
    flat = np.ascontiguousarray(_v(draws).reshape(-1))
    if flat.size != nplans * n_attempts:
        raise ValueError("draws must have shape (nplans, n_attempts)")
    lib.ahr_jitter_schedule(
        _addr(starts), _addr(factors), _addr(max_timeouts),
        _addr(flat), nplans, n_attempts, _addr(res),
    )
    return res


def linear_schedule(min_timeouts, max_timeouts, uniforms) -> np.ndarray:
    """Return the ``(nplans, n_attempts)`` uniform backoff schedule.

    ``result[p, a] == min_timeouts[p] + uniforms[p, a] * (max_timeouts[p]
    - min_timeouts[p])``, the ``aiohttp_retry.RandomRetry.get_timeout``
    formula. ``uniforms`` must have shape ``(nplans, n_attempts)``.
    """
    if np.isscalar(min_timeouts):
        flat = np.ascontiguousarray(_v(uniforms).reshape(-1))
        buf = _plan_buffer(min_timeouts, max_timeouts, 0.0, flat.size)
        if flat.size:
            base = buf.ctypes.data
            _linear(base, base + 8, flat.ctypes.data, 1, flat.size, base + 24)
        return buf[3:].reshape(1, flat.size)
    min_timeouts = _v(min_timeouts)
    max_timeouts = _v(max_timeouts)
    if min_timeouts.ndim != 1 or max_timeouts.shape != min_timeouts.shape:
        raise ValueError("min_timeouts and max_timeouts must be 1-D and equal length")
    uniforms = _v(uniforms)
    if uniforms.ndim != 2 or uniforms.shape[0] != min_timeouts.size:
        raise ValueError("uniforms must have shape (nplans, n_attempts)")
    nplans = min_timeouts.size
    n_attempts = uniforms.shape[1]
    res = np.empty((nplans, n_attempts), dtype=np.float64)
    if nplans == 0 or n_attempts == 0:
        return res
    flat = np.ascontiguousarray(uniforms.reshape(-1))
    lib.ahr_linear_schedule(
        _addr(min_timeouts), _addr(max_timeouts), _addr(flat),
        nplans, n_attempts, _addr(res),
    )
    return res
