# mojo-aiohttp-retry

`mojo-aiohttp-retry` is the compute-oriented subset of
[aiohttp-retry](https://github.com/inyutin/aiohttp_retry): the backoff-schedule
arithmetic, compiled into one Mojo shared library.

The Python package is named `mojo_aiohttp_retry`, so it installs alongside the
real `aiohttp_retry` and the tests compare the two directly. The four policy
classes are drop-in replacements for the ones in
`aiohttp_retry.retry_options` and can be handed straight to
`aiohttp_retry.RetryClient`.

```python
import aiohttp_retry
from mojo_aiohttp_retry import ExponentialRetry

retry = aiohttp_retry.RetryClient(
    ExponentialRetry(attempts=6, start_timeout=0.1, factor=2.0, max_timeout=30.0)
)

# or evaluate a whole fleet of candidate plans in one call
import mojo_aiohttp_retry as mar
mar.exponential_schedule([0.1, 0.25], [2.0, 1.5], [30.0, 4.0], 8)
# array([[0.2 , 0.4 , 0.8 , 1.6 , 3.2 , 6.4 , 12.8, 25.6 ],
#        [0.25, 0.375, 0.562, 0.844, 1.266, 1.898, 2.848, 3.0  ]])
```

## Why this is the compute core

`aiohttp_retry` is 652 lines, of which 415 are the aiohttp client plumbing:
`_RequestContext._do_request` loops over attempts, rotates URLs, checks status
codes and sleeps. It does no arithmetic at all.

`aiohttp_retry/retry_options.py` is the only file with floating-point work in
it, and all of it is in `get_timeout`:

| policy | formula |
| --- | --- |
| `ExponentialRetry` | `min(start_timeout * factor ** attempt, max_timeout)` |
| `FibonacciRetry` | `min(multiplier * F(step), max_timeout)`, seeded at 1.0/1.0 |
| `JitterRetry` | `min(start * factor ** attempt, max)` + `uniform(0, size) ** factor` |
| `RandomRetry` | `min_timeout + random() * (max_timeout - min_timeout)` |

A single `get_timeout` is one `pow` and one clamp. Evaluating a *schedule* is a
loop over (plan, attempt) doing exactly that pair, and that is the loop that is
compiled. Two properties make it worth compiling at all:

* the schedule is only ever consumed one element per retry, but it is fully
  determined by the policy's constructor arguments, so it can be computed once;
* `FibonacciRetry` is *stateful* - it advances `prev_step`/`current_step` on
  every call and ignores the attempt index, so upstream cannot produce its own
  sequence out of order or twice. Recomputing the recurrence from the seed is
  what makes a batched schedule addressable at all.

## Covered subset

| area | implemented API |
| --- | --- |
| Policies | `ExponentialRetry`, `FibonacciRetry`, `JitterRetry`, `RandomRetry` |
| Batched schedules | `exponential_schedule`, `fibonacci_schedule`, `jitter_schedule`, `linear_schedule` |
| Kernels | `ahr_exponential_schedule`, `ahr_fibonacci_schedule`, `ahr_jitter_schedule`, `ahr_linear_schedule` |

The policy classes reproduce the full `RetryOptionsBase` attribute surface
(`attempts`, `statuses`, `exceptions`, `methods`, `retry_all_server_errors`,
`evaluate_response_callback`) so the retry client reads them unchanged.

### Not implemented, and why

* **`RetryClient`, `_RequestContext`, `RequestParams`, `_url_to_urls`** -
  coroutine control flow and URL rotation, not compute. Use the real
  `aiohttp_retry`; its policies accept these classes.
* **The `evaluate_response_callback` plumbing** - an `await`ed predicate over an
  `aiohttp.ClientResponse`. Carried through unchanged, never called by the Mojo
  layer.
* **Logging and `Retry-After` handling** - `aiohttp_retry` 2.9.1 does not read
  `Retry-After`; its `get_timeout` is purely a function of the attempt number.
  There is no retry-after arithmetic to port.
* **The deprecated `RetryOptions` factory** - a `DeprecationWarning` wrapper
  around `ExponentialRetry`.

## Behavioural differences from upstream

Two, both deliberate and both covered by tests:

1. **`FibonacciRetry.get_timeout` is index-addressed.** Upstream advances
   internal state on every call, so `get_timeout(3)` twice returns two different
   numbers. Here the schedule is computed once and `get_timeout(attempt)`
   returns `_schedule[attempt - 1]`. The *sequence* is identical; the
   repeat-call behaviour is not, and cannot be.
2. **`RandomRetry` and `JitterRetry` draw their randomness at construction
   time**, one draw per attempt in call order, so a given attempt always yields
   the same timeout. Upstream draws inside `get_timeout`, so the value for an
   attempt is not a function of the attempt.

## Install

```bash
bash build/build.sh          # -> dist/libmojo-aiohttp-retry.so
PYTHONPATH=python python -m pytest tests -q
PYTHONPATH=python python bench/bench.py
```

The repository pins its own Mojo toolchain in `pixi.toml`
(`mojo == 1.2.0.dev2026092605`); use the shared environment rather than
`pixi install`. Set `PYTHONPATH=python` when using the package outside a task.

## Performance

Best-of-seven wall clock, same process, on the shared 36-core build box.
Every case verifies numerical agreement with its reference *before* timing, so
a kernel regression shows up as a correctness failure rather than as a fast
number. Baselines are the fastest fair formulation - vectorised NumPy for the
batched cases, an attempt-major loop for the capped Fibonacci schedule (which
genuinely cannot be one expression), and the real `aiohttp_retry` policy objects
for the object-construction case.

| case | reference | mojo-aiohttp-retry | result |
| --- | ---: | ---: | ---: |
| exponential 200000 plans x 32 attempts | 479.18 ms | 115.65 ms | 4.14x faster |
| fibonacci 200000 plans x 32 attempts | 327.60 ms | 36.47 ms | 8.98x faster |
| jitter 200000 plans x 32 attempts | 849.30 ms | 590.17 ms | 1.44x faster |
| 20000 single policies x 8 attempts | 90.47 ms | 339.61 ms | **0.27x - slower** |

The last row is a real loss and is reported as one. Constructing one policy at a
time is dominated by fixed overheads, not arithmetic: a six-argument ctypes call
costs about 5 us on this box and the NumPy buffer another 2 us, against roughly
5 us for the whole pure-Python `ExponentialRetry.__init__`. There is no
arithmetic left to win, so the wrapper can only lose. The batched API is where
the compilation pays, and it is the API to use when there is more than one plan.

The jitter case is the weakest win because `draw ** factor` with a fractional
exponent is a genuine transcendental in both implementations; Mojo gets it from
LLVM's vectorised `pow` and NumPy from its own, and neither is at libm speed.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-aiohttp-retry.so`.

`python/mojo_aiohttp_retry` owns every array. Each public function normalises
its inputs to contiguous `float64` and makes one call into the kernel.
Buffers cross the C ABI as 64-bit addresses (`ctypes.c_int64`; `c_int`
truncates and segfaults) and are rebuilt inside the kernel as
`Pointer[Float64, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric.

The schedule is indexed by attempt number. `aiohttp_retry` counts attempts from
1, so the exponential and jittered schedules - whose upstream exponent is the
1-based attempt number - are computed with one extra leading column that is then
dropped. The Fibonacci kernel's column 0 is already the first `get_timeout`
call, so nothing is dropped there.

**These kernels are memory- and compute-light per element, so they are left
serial.** A schedule of 200000 x 32 is 6.4M elements at one `pow` each, which
is above the ~2 flops/byte threshold where chunking a thread pool pays, but the
measured single-call throughput is already several times the vectorised NumPy
equivalent, so the serial loop is what ships. No `parallelize` is used: 1.2.0
cannot pass pointers into a parallel body.

## Numerical parity

Mojo emits its own `pow` rather than libm's, so nothing here is bit-exact:

| operation | agreement | asserted |
| --- | --- | --- |
| `factor ** attempt`, integer exponent | 2.7e-15 relative worst case over 6.4M samples, exponents to 31 | `rtol=1e-12` |
| `draw ** factor`, fractional exponent | 3.8e-9 relative worst case over 2000 random pairs | `rtol=1e-8` |
| `min(...)` clamp | exact - it selects one of two computed values | equality |

## Tests

`tests/test_retry_options.py`, 27 tests. They compare element by element
against `aiohttp_retry.ExponentialRetry`, `FibonacciRetry`, `JitterRetry` and
`RandomRetry` instances, including the batched row-per-plan path against a
one-plan-at-a-time loop, the clamp, the Fibonacci seed, the unclamped jitter
term, and the attribute surface `RetryClient` reads. `random.uniform` and
`random.random` are patched so both sides consume the same draws - note that
upstream's `RandomRetry` binds `random.random` as a default argument value at
class-definition time, so the instance attribute has to be replaced, not the
module attribute.

## License

MIT
