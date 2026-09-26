"""Parity against the real aiohttp_retry retry policies.

Every schedule is compared element by element with the corresponding
``aiohttp_retry.retry_options`` class.

Tolerances. Mojo emits its own ``pow`` rather than libm's, so nothing here is
bit-exact. For integer exponents (``factor ** attempt`` with the 1-based attempt
number) the measured worst case over 6.4M samples with exponents up to 31 is
2.7e-15 relative, so those tests assert ``rtol=1e-12``. The jitter term raises a
fractional value to ``factor``, where the vectorised path diverges further: the
measured worst case over 2000 random (base, exponent) pairs is 3.8e-9 relative,
so that test asserts ``rtol=1e-8``.

The clamp is a selection between two already-computed values, so those results
are compared with exact equality.
"""

from unittest import mock

import numpy as np
import pytest

import aiohttp_retry
import mojo_aiohttp_retry as mar

RTOL = 1e-12
JITTER_RTOL = 1e-8


def upstream_exponential(attempts, start, factor, max_timeout):
    return aiohttp_retry.ExponentialRetry(
        attempts=attempts, start_timeout=start, max_timeout=max_timeout, factor=factor
    )


# --------------------------------------------------------------------------
# ExponentialRetry


@pytest.mark.parametrize(
    "start,factor,max_timeout,attempts",
    [
        (0.1, 2.0, 30.0, 8),
        (0.25, 1.5, 4.0, 10),
        (1.0, 3.0, 1e9, 12),
        (0.001, 10.0, 0.5, 6),
        (0.1, 1.0, 30.0, 5),
    ],
)
def test_exponential_matches_upstream(start, factor, max_timeout, attempts):
    theirs = upstream_exponential(attempts, start, factor, max_timeout)
    mine = mar.ExponentialRetry(
        attempts=attempts, start_timeout=start, max_timeout=max_timeout, factor=factor
    )
    for attempt in range(1, attempts + 1):
        np.testing.assert_allclose(
            mine.get_timeout(attempt), theirs.get_timeout(attempt), rtol=RTOL
        )


def test_exponential_clamps_at_max_timeout():
    """Past the clamp point every attempt must return exactly max_timeout."""
    attempts, max_timeout = 12, 3.0
    theirs = upstream_exponential(attempts, 0.1, 2.0, max_timeout)
    mine = mar.ExponentialRetry(
        attempts=attempts, start_timeout=0.1, max_timeout=max_timeout, factor=2.0
    )
    clamped = [a for a in range(1, attempts + 1) if theirs.get_timeout(a) == max_timeout]
    assert clamped, "this configuration should clamp somewhere inside the schedule"
    for attempt in clamped:
        # min() selects one of two stored values, so this one is exact.
        assert mine.get_timeout(attempt) == max_timeout
    for attempt in range(1, clamped[0]):
        assert mine.get_timeout(attempt) < max_timeout


def test_exponential_factor_one_is_flat():
    """factor=1 is the case that catches an exponent that is off by one."""
    mine = mar.ExponentialRetry(attempts=5, start_timeout=0.7, factor=1.0)
    values = [mine.get_timeout(a) for a in range(1, 6)]
    np.testing.assert_allclose(values, [0.7] * 5, rtol=RTOL)


def test_batched_exponential_matches_a_plan_at_a_time():
    """A row-per-plan bug would pass any single-plan test; this catches it."""
    rng = np.random.default_rng(3)
    nplans, n_attempts = 64, 17
    starts = rng.uniform(0.01, 2.0, nplans)
    factors = rng.uniform(1.0, 3.0, nplans)
    maxes = rng.uniform(0.5, 30.0, nplans)

    got = mar.exponential_schedule(starts, factors, maxes, n_attempts + 1)
    assert got.shape == (nplans, n_attempts + 1)
    for p in range(nplans):
        theirs = upstream_exponential(n_attempts, starts[p], factors[p], maxes[p])
        for a in range(1, n_attempts + 1):
            np.testing.assert_allclose(
                got[p, a], theirs.get_timeout(a), rtol=RTOL, err_msg=f"plan {p}"
            )


def test_batched_exponential_handles_empty_and_zero_attempts():
    assert mar.exponential_schedule([], [], [], 5).shape == (0, 5)
    assert mar.exponential_schedule([0.1], [2.0], [30.0], 0).shape == (1, 0)
    with pytest.raises(ValueError):
        mar.exponential_schedule([0.1, 0.2], [2.0], [30.0], 4)
    with pytest.raises(ValueError):
        mar.exponential_schedule([0.1], [2.0], [30.0], -1)


# --------------------------------------------------------------------------
# FibonacciRetry


def test_fibonacci_matches_upstream_sequence():
    """Upstream advances its own state; read it out one call at a time."""
    attempts, multiplier, max_timeout = 10, 1.0, 1e9
    theirs = aiohttp_retry.FibonacciRetry(
        attempts=attempts, multiplier=multiplier, max_timeout=max_timeout
    )
    mine = mar.FibonacciRetry(
        attempts=attempts, multiplier=multiplier, max_timeout=max_timeout
    )
    for attempt in range(1, attempts + 1):
        np.testing.assert_allclose(
            mine.get_timeout(attempt), theirs.get_timeout(attempt), rtol=RTOL
        )


def test_fibonacci_follows_the_closed_form():
    """Call k is multiplier * F(k + 2); a wrong seed shifts the whole series."""
    attempts = 12
    mine = mar.FibonacciRetry(attempts=attempts, multiplier=1.0, max_timeout=1e18)
    fib = [1.0, 1.0]
    for _ in range(attempts + 2):
        fib.append(fib[-1] + fib[-2])
    # fib[j] == F(j + 1) and call k yields F(k + 2), so the value is fib[k + 1].
    for attempt in range(1, attempts + 1):
        np.testing.assert_allclose(
            mine.get_timeout(attempt), fib[attempt + 1], rtol=RTOL
        )


def test_fibonacci_clamps_at_max_timeout():
    attempts = 12
    theirs = aiohttp_retry.FibonacciRetry(
        attempts=attempts, multiplier=2.0, max_timeout=5.0
    )
    mine = mar.FibonacciRetry(attempts=attempts, multiplier=2.0, max_timeout=5.0)
    for attempt in range(1, attempts + 1):
        np.testing.assert_allclose(
            mine.get_timeout(attempt), theirs.get_timeout(attempt), rtol=RTOL
        )
    assert mine.get_timeout(attempts) == 5.0
    assert mine.get_timeout(1) < 5.0


def test_fibonacci_is_index_addressed_where_upstream_is_stateful():
    """Documented divergence: repeated reads do not advance anything."""
    mine = mar.FibonacciRetry(attempts=6, multiplier=1.0, max_timeout=1e9)
    first = mine.get_timeout(3)
    assert mine.get_timeout(3) == first
    assert mine.get_timeout(4) > first

    theirs = aiohttp_retry.FibonacciRetry(attempts=6, multiplier=1.0, max_timeout=1e9)
    theirs.get_timeout(1)
    theirs.get_timeout(1)
    assert theirs.get_timeout(1) > 2.0  # upstream consumed three terms


def test_batched_fibonacci_matches_a_plan_at_a_time():
    rng = np.random.default_rng(5)
    nplans, n_attempts = 40, 20
    mults = rng.uniform(0.1, 3.0, nplans)
    maxes = rng.uniform(1.0, 1e6, nplans)
    got = mar.fibonacci_schedule(mults, maxes, n_attempts)
    assert got.shape == (nplans, n_attempts)
    for p in range(nplans):
        theirs = aiohttp_retry.FibonacciRetry(
            attempts=n_attempts, multiplier=mults[p], max_timeout=maxes[p]
        )
        for a in range(n_attempts):
            np.testing.assert_allclose(
                got[p, a], theirs.get_timeout(a + 1), rtol=RTOL, err_msg=f"plan {p}"
            )


# --------------------------------------------------------------------------
# JitterRetry


def test_jitter_matches_upstream_with_fixed_draws():
    """`random.uniform` is patched so both sides consume the same draws."""
    attempts = 8
    draws = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 1.9, 2.0]
    with mock.patch("random.uniform", side_effect=draws) as m:
        mine = mar.JitterRetry(
            attempts=attempts, start_timeout=0.1, max_timeout=30.0,
            factor=2.0, random_interval_size=2.0,
        )
        assert m.call_count == attempts
        assert all(c.args == (0.0, 2.0) for c in m.call_args_list)

    theirs = aiohttp_retry.JitterRetry(
        attempts=attempts, start_timeout=0.1, max_timeout=30.0,
        factor=2.0, random_interval_size=2.0,
    )
    for attempt in range(1, attempts + 1):
        with mock.patch("random.uniform", side_effect=draws[attempt - 1:]):
            expected = theirs.get_timeout(attempt)
        np.testing.assert_allclose(
            mine.get_timeout(attempt), expected, rtol=JITTER_RTOL,
            err_msg=f"attempt {attempt}",
        )


def test_jitter_term_is_not_clamped():
    """Upstream adds the jitter on top of the clamped exponential value."""
    with mock.patch("random.uniform", side_effect=[1.5] * 6):
        jittered = mar.JitterRetry(
            attempts=6, start_timeout=0.1, max_timeout=0.5, factor=2.0,
            random_interval_size=2.0,
        )
    base = mar.ExponentialRetry(
        attempts=6, start_timeout=0.1, max_timeout=0.5, factor=2.0
    )
    for attempt in range(1, 7):
        assert jittered.get_timeout(attempt) > base.get_timeout(attempt)
    assert jittered.get_timeout(6) > 0.5


def test_batched_jitter_matches_the_formula():
    rng = np.random.default_rng(7)
    nplans, n_attempts = 32, 9
    starts = rng.uniform(0.01, 1.0, nplans)
    factors = rng.uniform(1.0, 4.0, nplans)
    maxes = rng.uniform(0.2, 10.0, nplans)
    draws = rng.uniform(0.0, 2.0, size=(nplans, n_attempts))
    got = mar.jitter_schedule(
        starts, factors, maxes, draws=draws, n_attempts=n_attempts
    )
    expect = (
        np.minimum(
            starts[:, None] * factors[:, None] ** np.arange(n_attempts)[None, :],
            maxes[:, None],
        )
        + draws ** factors[:, None]
    )
    # NumPy's vectorised fractional pow and Mojo's both differ from libm's
    # scalar pow by a few ULP; the two references agree to 4e-9 as measured.
    np.testing.assert_allclose(got, expect, rtol=JITTER_RTOL)
    with pytest.raises(ValueError):
        mar.jitter_schedule(starts, factors, maxes, draws[:, :-1], n_attempts)


# --------------------------------------------------------------------------
# RandomRetry


def test_random_matches_upstream_with_fixed_draws():
    attempts = 6
    draws = [0.0, 0.1, 0.25, 0.5, 0.75, 0.99]
    with mock.patch("random.random", side_effect=draws) as m:
        mine = mar.RandomRetry(attempts=attempts, min_timeout=0.1, max_timeout=3.0)
        assert m.call_count == attempts
    # `RandomRetry` binds `random.random` as a default argument value at class
    # definition time, so patching the module attribute does not reach it; the
    # instance attribute is what get_timeout actually calls.
    theirs = aiohttp_retry.RandomRetry(
        attempts=attempts, min_timeout=0.1, max_timeout=3.0
    )
    for attempt, draw in zip(range(1, attempts + 1), draws, strict=True):
        theirs.random = lambda value=draw: value
        np.testing.assert_allclose(
            mine.get_timeout(attempt), theirs.get_timeout(attempt), rtol=RTOL
        )


def test_random_stays_inside_its_bounds():
    rng = np.random.default_rng(11)
    lo, hi = 0.25, 0.75
    draws = rng.random(64).reshape(1, -1)
    got = mar.linear_schedule([lo], [hi], draws)
    assert got.min() >= lo and got.max() <= hi
    np.testing.assert_allclose(got, lo + draws * (hi - lo), rtol=RTOL)


def test_batched_linear_schedule_is_per_plan():
    rng = np.random.default_rng(13)
    nplans, n_attempts = 20, 5
    los = rng.uniform(0.0, 1.0, nplans)
    his = los + rng.uniform(0.0, 5.0, nplans)
    draws = rng.random((nplans, n_attempts))
    got = mar.linear_schedule(los, his, draws)
    for p in range(nplans):
        for a in range(n_attempts):
            np.testing.assert_allclose(
                got[p, a], los[p] + draws[p, a] * (his[p] - los[p]), rtol=RTOL
            )


# --------------------------------------------------------------------------
# Drop-in surface


@pytest.mark.parametrize(
    "cls", [mar.ExponentialRetry, mar.FibonacciRetry, mar.JitterRetry, mar.RandomRetry]
)
def test_policy_exposes_the_attributes_retry_client_reads(cls):
    """`RetryClient` and `_RequestContext` read exactly these; miss one and it breaks."""
    policy = cls(
        attempts=4, statuses={503}, exceptions={ValueError}, methods={"get", "post"}
    )
    assert policy.attempts == 4
    assert 503 in policy.statuses
    assert ValueError in policy.exceptions
    assert policy.methods == {"GET", "POST"}
    assert policy.retry_all_server_errors is True
    assert policy.evaluate_response_callback is None
    assert callable(policy.get_timeout)


def test_default_methods_match_upstream():
    theirs = aiohttp_retry.ExponentialRetry()
    mine = mar.ExponentialRetry()
    assert mine.methods == theirs.methods
    assert mine.statuses == set(theirs.statuses)
    assert mine.attempts == theirs.attempts


def test_get_timeout_rejects_attempts_outside_the_plan():
    mine = mar.ExponentialRetry(attempts=3)
    with pytest.raises(IndexError):
        mine.get_timeout(0)
    with pytest.raises(IndexError):
        mine.get_timeout(4)


def test_zero_attempt_plan_has_an_empty_schedule():
    mine = mar.ExponentialRetry(attempts=0)
    assert mine.schedule.size == 0
    with pytest.raises(IndexError):
        mine.get_timeout(1)
