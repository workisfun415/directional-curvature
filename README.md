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

**Measured 3D volume with a mask** — for imaging volumes. Same interface, with
`(dz, dy, dx)` spacing and an array indexed `[z, y, x]`.

```python
from dircurv import VolumeField, reliability_volumes, coverage_fraction
field = VolumeField(volume, spacing=(0.002, 0.002, 0.002), mask=mask)
print("coverage:", coverage_fraction(field, step=4))   # check this first
vols = reliability_volumes(field, m=24, sigma=1e-4, step=4)
```

Measurement in 3D is expensive, because each voxel ray-marches every direction.
Start with `step=4`, look at the maps, then refine.

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

In 2D the support is a 4×4 block of 16 pixels:

| mask | pixels with a complete stencil |
|---|---|
| full grid | 92.7% |
| straight boundary | 89.9% |
| circular | 76.3% |
| irregular notch | 64.4% |
| thin strip | 38.5% |
| 20% scattered dropout | **3.0%** |

In 3D it is a 4×4×4 block of 64 voxels, so the requirement is far stricter:

| mask | voxels with a complete stencil |
|---|---|
| full grid | 68.7% |
| half space | 69.2% |
| sphere | 48.5% |
| sphere with a notch | 36.2% |
| slab, 5 voxels thick | 26.0% |
| 10% scattered dropout | **0.1%** |

That last figure is not a bug: 0.9⁶⁴ ≈ 0.001. Scattered per-voxel exclusion is
fatal in 3D. Call `coverage_fraction()` before anything else.

Where the stencil would leave the mask the result is `UNUSABLE`. The package
never extrapolates across invalid data: in testing, 165 of 165 such pixels were
refused. The 3D interface is designed for locally contiguous volumetric data, not
for arbitrary scattered exclusions.

Check in this order, and never reverse it:

```
coverage  ->  rank  ->  aperture and span  ->  C3 status  ->  result
```

If voxels are missing in scattered fashion, filling them is a preprocessing
decision for you to make deliberately — the package will not do it, because
interpolating over gaps alters the derivative information the method is meant to
measure.

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

## How many directions

In 2D the Hessian has three unknowns, so three directions identify it and six in
antipodal pairs give second order. Estimating C3 needs more than the 10 columns
of the degree-3 basis on the circle.

In 3D the Hessian has six unknowns, so six directions identify it and twelve in
pairs give second order.

For the present 3D C3 estimator, at least 20 directional measurements are
required by the dimension of the degree-3 basis on the sphere; `m=24` is adopted
as the default tested configuration, to give an overdetermined fit. This is a
requirement of the C3 estimator, not of the method: **Hessian recovery and
conditioning remain meaningful with fewer directions**, provided the sampled rank
condition is satisfied — at `m=12` the Hessian and κ are returned normally and
only C3 is reported `UNDER-DETERMINED`. What is lost without C3 is control of the
truncation term through the span rule: on the self-test field the relative Hessian
error was 1.6×10⁻² at `m=12` and 5.4×10⁻⁴ at `m=24`.

## C3 is never fabricated

If the third-order magnitude cannot be estimated, the package says so and, by
default, recommends quadratic regression instead of substituting a value.
Statuses: `VALID`, `NOISE-LIMITED`, `SPAN-LIMITED`, `UNDER-DETERMINED`,
`USER-SUPPLIED`, `NOT-ATTEMPTED`. Policies via `on_c3_unavailable`: `"defer"`
(default), `"hmax"`, `"require"`.

## Repository layout

```
src/dircurv/                the library
  analytic.py               callable f, analytic geometry, 2D and 3D
  grid2d.py                 measured 2D array with a mask
  grid3d.py                 measured 3D volume with a mask
tests/                      34 tests, including the exactness and
                            no-extrapolation gates in both dimensions
examples/                   runnable scripts
directional_curvature.py    reproduction script for the manuscript
verify_proofs.py            symbolic verification of the proof steps
```

## Citing

Archived at https://doi.org/10.5281/zenodo.21793101. See `CITATION.cff`.

## License

MIT.
