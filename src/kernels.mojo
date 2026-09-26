"""Batched retry-schedule arithmetic for aiohttp-retry.

`aiohttp_retry.retry_options` is the only part of aiohttp-retry that does
arithmetic. Four policies decide how long to sleep between attempts:

    ExponentialRetry   min(start_timeout * factor ** attempt, max_timeout)
    FibonacciRetry     min(multiplier * F(attempt + 2), max_timeout)   (stateful)
    JitterRetry        min(start * factor ** attempt, max)
                       + uniform(0, random_interval_size) ** factor
    RandomRetry        min_timeout + random() * (max_timeout - min_timeout)

A single `get_timeout` call is scalar, but evaluating a whole schedule - or the
schedules of a fleet of candidate plans - is a nested loop over
(plan, attempt) doing one `pow` and one clamp per element. That loop is what is
compiled here. Every buffer crosses the C ABI as a 64-bit address and the
pointer is rebuilt inside the body, because `@export` rejects parametric
functions and an inferred pointer origin would make the symbol parametric.
"""

from std.math import pow

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


@export("ahr_exponential_schedule")
def ahr_exponential_schedule(
    starts_addr: Int,
    factors_addr: Int,
    maxes_addr: Int,
    nplans: Int,
    nattempts: Int,
    result_addr: Int,
) abi("C"):
    """Schedule row: min(starts[p] * factors[p] ** a, maxes[p]).

    Mirrors `ExponentialRetry.get_timeout`, which starts counting at
    `current_attempt == 1`, so the caller passes attempts 0..n-1 and adds the
    shift itself.
    """
    var starts = fp(starts_addr)
    var factors = fp(factors_addr)
    var maxes = fp(maxes_addr)
    var res = fp(result_addr)
    for p in range(nplans):
        var s = starts[unsafe_offset=p]
        var f = factors[unsafe_offset=p]
        var m = maxes[unsafe_offset=p]
        var base = p * nattempts
        for a in range(nattempts):
            var t = s * pow(f, Float64(a))
            res[unsafe_offset=base + a] = t if t < m else m


@export("ahr_fibonacci_schedule")
def ahr_fibonacci_schedule(
    multipliers_addr: Int,
    maxes_addr: Int,
    nplans: Int,
    nattempts: Int,
    result_addr: Int,
) abi("C"):
    """Schedule row: min(multipliers[p] * F(a + 2), maxes[p]).

    `FibonacciRetry` keeps `prev_step`/`current_step` as instance state seeded
    at 1.0 and advances them once per `get_timeout` call, so call k returns
    F(k + 2) * multiplier clamped to max_timeout. Recomputing the whole
    sequence from the seed is what makes the schedule batchable at all; the
    stateful upstream object can only ever produce it one element at a time.
    """
    var mults = fp(multipliers_addr)
    var maxes = fp(maxes_addr)
    var res = fp(result_addr)
    for p in range(nplans):
        var mult = mults[unsafe_offset=p]
        var m = maxes[unsafe_offset=p]
        var prev = 1.0
        var cur = 1.0
        var base = p * nattempts
        for a in range(nattempts):
            var nxt = prev + cur
            prev = cur
            cur = nxt
            var t = mult * nxt
            res[unsafe_offset=base + a] = t if t < m else m


@export("ahr_jitter_schedule")
def ahr_jitter_schedule(
    starts_addr: Int,
    factors_addr: Int,
    maxes_addr: Int,
    draws_addr: Int,
    nplans: Int,
    nattempts: Int,
    result_addr: Int,
) abi("C"):
    """Schedule row: min(starts[p] * factors[p] ** a, maxes[p])
                      + pow(draws[p, a], factors[p]).

    The jitter term is not clamped: upstream adds
    `random.uniform(0, random_interval_size) ** factor` on top of the already
    clamped exponential value. `draws` holds the absolute `uniform(0, size)`
    values in row-major (plan, attempt) order, so the kernel stays
    deterministic and parity-testable; the caller owns the RNG.
    """
    var starts = fp(starts_addr)
    var factors = fp(factors_addr)
    var maxes = fp(maxes_addr)
    var draws = fp(draws_addr)
    var res = fp(result_addr)
    for p in range(nplans):
        var s = starts[unsafe_offset=p]
        var f = factors[unsafe_offset=p]
        var m = maxes[unsafe_offset=p]
        var base = p * nattempts
        for a in range(nattempts):
            var base_t = s * pow(f, Float64(a))
            var clamped = base_t if base_t < m else m
            var u = draws[unsafe_offset=base + a]
            res[unsafe_offset=base + a] = clamped + pow(u, f)


@export("ahr_linear_schedule")
def ahr_linear_schedule(
    mins_addr: Int,
    maxes_addr: Int,
    uniforms_addr: Int,
    nplans: Int,
    nattempts: Int,
    result_addr: Int,
) abi("C"):
    """Schedule row: mins[p] + uniforms[p, a] * (maxes[p] - mins[p]).

    `RandomRetry` ignores the attempt index and draws a fresh uniform on every
    call, so the draw axis is (plan, attempt).
    """
    var mins = fp(mins_addr)
    var maxes = fp(maxes_addr)
    var uniforms = fp(uniforms_addr)
    var res = fp(result_addr)
    for p in range(nplans):
        var lo = mins[unsafe_offset=p]
        var span = maxes[unsafe_offset=p] - lo
        var base = p * nattempts
        for a in range(nattempts):
            res[unsafe_offset=base + a] = lo + uniforms[unsafe_offset=base + a] * span
