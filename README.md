# dircurv

Directional curvature measurement under restricted sampling, with reliability
reporting.

[![tests](https://github.com/workisfun415/directional-curvature/actions/workflows/tests.yml/badge.svg)](https://github.com/workisfun415/directional-curvature/actions)

Estimating a Hessian requires probing in several directions. When the sampling
geometry is restricted — near a boundary, under active constraints, inside a thin
structure, or within an irregular mask — some directions are unavailable and some
spans are short. This package measures the curvature and, more importantly, tells
you how much the geometry has cost you.

It returns three distinct quantities, which must not be conflated:

| quantity | meaning |
|---|---|
| `H_hat` | the recovered Hessian |
| `kappa` | conditioning of the locally available direction geometry |
| `C3_hat` | third-order magnitude, or an explicit reason it is unavailable |

together with the usable aperture, the attainable order of accuracy, and a
verdict.

## What this is not

It does not estimate stiffness, solve an inverse problem, or compete with an
inversion. In the benchmark accompanying the paper, quadratic regression on the
same feasible sample points matched or outperformed the directional estimator in
**all 48 configurations tested**. Read `when_not_to_use()` before relying on any
output; it is a function, not a footnote.

The contribution is the diagnosis, not the estimator.

## Install

```bash
pip install -e ".[dev]"     # from a clone
```

Requires Python 3.9+ and numpy. Examples and the reproduction scripts also need
matplotlib, sympy and scipy.

## Two entry points

**Callable function, analytic geometry** — for derivative-free optimisation,
method development, and reproducing the paper.

```python
import numpy as np
from dircurv import Geometry, measure

f = lambda p: np.exp(p[0] + 0.5*p[1]) * np.cos(p[2])
g = Geometry.cone(dim=3, half_angle_deg=45, max_span=0.3)
print(measure(f, np.full(3, 0.1), g, sigma=1e-6, m=24))
```

**Measured 2D array with a mask** — for images and measured fields. Feasible
directions are derived from the mask by ray-marching.

```python
from dircurv import GridField, reliability_maps
maps = reliability_maps(GridField(array, spacing=0.002, mask=mask),
                        m=16, sigma=1e-4)
```

## Preconditions for measured data

Two are mathematical rather than technical, and violating them produces numbers
that look reasonable and mean nothing.

**Complex fields are processed component-wise.** The expansion is linear in the
field, so it applies to the real and imaginary parts of a complex phasor
separately. It does **not** apply to the amplitude `|u|`: taking the modulus
before differentiation is nonlinear and outside the theory. Complex input raises
`TypeError`.

**The mask must be locally contiguous.** Interpolation is bicubic, which
reproduces cubics exactly and therefore leaves both the quadratic model and the
first-order remainder intact; a lower-order interpolant injects error of the same
size as the quantity being measured. Bicubic needs a complete local 4×4
neighbourhood, so coverage depends on the mask:

| mask | pixels with a complete stencil |
|---|---|
| full grid | 92.7% |
| straight boundary | 89.9% |
| circular | 76.3% |
| irregular notch | 64.4% |
| thin strip | 38.5% |
| 20% scattered dropout | **3.0%** |

Where the stencil would leave the mask the result is `UNUSABLE`. The package
never extrapolates across invalid data: in testing, 165 of 165 such pixels were
refused.

## Attainable order

Second-order accuracy requires antipodal partners to be **sampled**, not merely
feasible. The icosahedral six-axis set and every Fibonacci set contain no
antipodal pair, so in wholly unrestricted geometry they still give first order:

| direction set | sampled antipodes | measured order |
|---|---|---|
| icosahedral, 6 | 0% | 1.01 |
| icosahedral ±, 12 | 100% | 2.00 |
| Fibonacci, 24 | 0% | 1.10 |
| Fibonacci ±, 48 | 100% | 2.00 |

## C3 is never fabricated

If the third-order magnitude cannot be estimated, the package says so and, by
default, recommends quadratic regression instead of substituting a value.
Statuses: `VALID`, `NOISE-LIMITED`, `SPAN-LIMITED`, `UNDER-DETERMINED`,
`USER-SUPPLIED`, `NOT-ATTEMPTED`. Policies via `on_c3_unavailable`: `"defer"`
(default), `"hmax"`, `"require"`.

## Repository layout

```
src/dircurv/      the library: analytic.py, grid2d.py
tests/            25 tests, including the exactness and no-extrapolation gates
examples/         three runnable scripts
paper/            reproduction scripts for the accompanying manuscript
```

## Citing

Archived at https://doi.org/10.5281/zenodo.21793101. See `CITATION.cff`.

## License

MIT.
