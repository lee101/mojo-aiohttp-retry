"""Correctness-gated benchmark for mojo-aiohttp-retry.

Every case verifies numerical agreement before it is timed, so a regression in
the Mojo kernels shows up as a correctness failure rather than as a suspiciously
good number. Baselines are the fastest fair formulation, not a Python loop that
NumPy would never use:

* batched cases are compared against vectorised NumPy (broadcast ``**`` and
  ``np.minimum``, or an attempt-major loop for the capped Fibonacci schedule,
  which genuinely cannot be expressed as one NumPy expression);
* the policy-construction case is compared against constructing the real
  `aiohttp_retry` policy objects, which is what a fleet of retry plans costs.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import aiohttp_retry  # noqa: E402
import mojo_aiohttp_retry as mar  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _numpy_exponential(starts, factors, maxes, n):
    attempts = np.arange(n, dtype=np.float64)
    return np.minimum(starts[:, None] * factors[:, None] ** attempts[None, :], maxes[:, None])


def bench_exponential(nplans: int = 200_000, n_attempts: int = 32):
    rng = np.random.default_rng(0)
    starts = rng.uniform(0.01, 2.0, nplans)
    factors = rng.uniform(1.0, 3.0, nplans)
    maxes = rng.uniform(0.5, 30.0, nplans)

    expect = _numpy_exponential(starts, factors, maxes, n_attempts)
    got = mar.exponential_schedule(starts, factors, maxes, n_attempts)
    assert np.allclose(got, expect, rtol=1e-12), "exponential schedule mismatch"

    ref = _time(lambda: _numpy_exponential(starts, factors, maxes, n_attempts), 7)
    mine = _time(lambda: mar.exponential_schedule(starts, factors, maxes, n_attempts), 7)
    return f"exponential {nplans}x{n_attempts}", ref, mine


def bench_fibonacci(nplans: int = 200_000, n_attempts: int = 32):
    """The cap makes this a sequential recurrence, so the NumPy baseline loops
    over attempts and stays vectorised over plans - the best fair expression."""
    rng = np.random.default_rng(1)
    mults = rng.uniform(0.1, 3.0, nplans)
    maxes = rng.uniform(1.0, 1e6, nplans)

    def numpy_fibonacci():
        out = np.empty((nplans, n_attempts), dtype=np.float64)
        prev = np.ones(nplans)
        cur = np.ones(nplans)
        for a in range(n_attempts):
            nxt = prev + cur
            prev, cur = cur, nxt
            t = mults * nxt
            out[:, a] = np.minimum(t, maxes)
        return out

    expect = numpy_fibonacci()
    got = mar.fibonacci_schedule(mults, maxes, n_attempts)
    assert np.allclose(got, expect, rtol=0, atol=0), "fibonacci schedule mismatch"

    ref = _time(numpy_fibonacci, 7)
    mine = _time(lambda: mar.fibonacci_schedule(mults, maxes, n_attempts), 7)
    return f"fibonacci {nplans}x{n_attempts}", ref, mine


def bench_jitter(nplans: int = 200_000, n_attempts: int = 32):
    rng = np.random.default_rng(2)
    starts = rng.uniform(0.01, 1.0, nplans)
    factors = rng.uniform(1.0, 3.0, nplans)
    maxes = rng.uniform(0.5, 10.0, nplans)
    draws = rng.uniform(0.0, 2.0, size=(nplans, n_attempts))

    expect = _numpy_exponential(starts, factors, maxes, n_attempts) + draws ** factors[:, None]
    got = mar.jitter_schedule(starts, factors, maxes, draws, n_attempts)
    # Fractional exponent: both sides are vectorised pow, a few ULP apart.
    assert np.allclose(got, expect, rtol=1e-8), "jitter schedule mismatch"

    def numpy_jitter():
        return _numpy_exponential(starts, factors, maxes, n_attempts) + draws ** factors[:, None]

    ref = _time(numpy_jitter, 7)
    mine = _time(lambda: mar.jitter_schedule(starts, factors, maxes, draws, n_attempts), 7)
    return f"jitter {nplans}x{n_attempts}", ref, mine


def bench_policy_construction(nplans: int = 20_000, n_attempts: int = 8):
    """What an application actually pays: one policy object per candidate plan."""
    rng = np.random.default_rng(3)
    starts = rng.uniform(0.01, 2.0, nplans)
    factors = rng.uniform(1.0, 3.0, nplans)
    maxes = rng.uniform(0.5, 30.0, nplans)

    def mine():
        for s, f, m in zip(starts.tolist(), factors.tolist(), maxes.tolist(), strict=True):
            mar.ExponentialRetry(
                attempts=n_attempts, start_timeout=s, factor=f, max_timeout=m
            )

    def theirs():
        for s, f, m in zip(starts.tolist(), factors.tolist(), maxes.tolist(), strict=True):
            aiohttp_retry.ExponentialRetry(
                attempts=n_attempts, start_timeout=s, factor=f, max_timeout=m
            )

    sample = mar.ExponentialRetry(
        attempts=n_attempts, start_timeout=0.1, factor=2.0, max_timeout=30.0
    )
    ref_sample = aiohttp_retry.ExponentialRetry(
        attempts=n_attempts, start_timeout=0.1, factor=2.0, max_timeout=30.0
    )
    for a in range(1, n_attempts + 1):
        np.testing.assert_allclose(
            sample.get_timeout(a), ref_sample.get_timeout(a), rtol=1e-15
        )

    return "policy objects x%d" % nplans, _time(theirs, 7), _time(mine, 7)


def main():
    print(f"{'case':<28}{'reference':>12}{'mojo-aiohttp-retry':>20}{'ratio':>10}")
    print(f"{'':<28}{'':>12}{'':>20}{'(ref/mojo)':>10}")
    print("-" * 70)
    for fn in (bench_exponential, bench_fibonacci, bench_jitter, bench_policy_construction):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        verdict = "faster" if ratio > 1.02 else ("slower" if ratio < 0.98 else "parity")
        print(f"{label:<28}{ref*1e3:>10.2f}ms{got*1e3:>18.2f}ms{ratio:>9.2f}x  {verdict}")


if __name__ == "__main__":
    main()
