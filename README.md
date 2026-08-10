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

## Running on a measured volume, without writing code

```bash
pip install -e ".[dev]"          # includes nibabel and scipy for file I/O

# check what is measurable before spending time on a full run
python -m dircurv u.nii.gz --mask brain.nii.gz --coverage-only

# then the analysis; sigma is in the field's own units, or use --sigma-relative
python -m dircurv u.nii.gz --mask brain.nii.gz --sigma-relative 0.01 \
    --out maps/ --step 4
```

Reads `.nii`, `.nii.gz`, `.mat` (v5–v7) and `.npy`/`.npz`. Writes one NIfTI per
map — `verdict`, `kappa`, `rho`, `solid_angle`, `order`, `c3`, `c3_status`,
`span_min`, `span_max` and the six Hessian components — transposed back to the
input axis order so they overlay on the original data.

For complex data, run each part separately, since the operator is linear in the
field:

```bash
python -m dircurv u.mat --component real --out maps_real/
python -m dircurv u.mat --component imag --out maps_imag/
```

Two conventions the loader handles explicitly, because getting either wrong is
silent rather than noisy. **Axis order**: NIfTI and MATLAB store `[x, y, z]`
while the modules index `[z, y, x]`, so axes are reversed by default and the
loaded shape and spacing are printed back for checking. **Units**: none are
converted. Spacing is used as given, so the Hessian is in field-units per
spacing-unit squared, and `--sigma` is absolute rather than relative.

## Which path to use on grid data

**Use `lattice` near boundaries.** Probe points are restricted to grid nodes, so
no interpolation is performed and the support is exactly the nodes each ray
visits. A one-sided ray then needs nothing on the far side of the evaluation
point.

```python
from dircurv import measure_lattice
r = measure_lattice(volume, (iz, iy, ix), spacing=(dz, dy, dx), mask=mask,
                    order=1)      # order=1 gives 26 directions in 3D
print(r)
```

The interpolating paths (`grid2d`, `grid3d`) need a complete 4×4 or 4×4×4
neighbourhood, which is **larger** than a central-difference stencil. Measured on
a masked sphere:

| | applicable voxels | only this method | only central differences |
|---|---|---|---|
| `grid3d` (tricubic) | 15,504 | **0** | 1,873 |
| `lattice` | 22,371 | **4,100** | 0 |
| central differences | 18,271 | — | — |

So the interpolating path has no boundary advantage at all, while `lattice`
reaches 22% more of the volume. Use the interpolating paths only when off-grid
sampling genuinely matters.

**Accuracy against central differences**, MRE-like field, 2 mm voxels, 20 mm
wavelength:

| | σ=10⁻³ | σ=10⁻² |
|---|---|---|
| 3D interior, `lattice` | **0.387** | **0.433** |
| 3D interior, central differences | 0.831 | 0.836 |
| 3D boundary-only voxels, `lattice` | 0.783 | 0.794 |
| 2D interior, `lattice` | 0.426 | — |
| 2D interior, central differences | **0.060** | — |

In three dimensions `lattice` is about twice as accurate as central differences
and reaches voxels they cannot. **In two dimensions it is roughly seven times
worse**: a one-sided stencil is first order, and at ten points per wavelength the
O(h·f‴) term dominates, which the richer 3D direction set recovers from and the
eight short 2D lattice directions do not.

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

## Reliability architecture

No single scalar detects every failure, so each mode has its own detector:

| failure mode | detector | outcome |
|---|---|---|
| too few usable directions | rank | `UNUSABLE` |
| interpolation stencil leaves the mask | mask support | `UNUSABLE` |
| narrow or anisotropic geometry | aperture, span floor | `SPAN-LIMITED` |
| truncation or noise too large | ρ | `CAUTION` / `DEFER` |
| C3 not estimable | pilot status | `DEFER` |
| good geometry and small predicted error | — | `GOOD` |

**Geometry alone never returns GOOD.** The verdict combines the geometry with a
predicted error

    E = kappa ( (1+R)/3 C3 h  +  2 sigma nu / (R(1-R) h^2) ),   rho = E / ||H||_F

Measured behaviour on a smooth 2D field, showing ρ against the true relative
error:

| σ | true error | ρ | verdict |
|---|---|---|---|
| 0 | 0.000 | 0.009 | GOOD |
| 10⁻⁴ | 0.017 | 0.077 | GOOD |
| 10⁻³ | 0.152 | 0.227 | CAUTION |
| 3×10⁻³ | 0.573 | 0.458 | DEFER |
| 10⁻² | 1.951 | 0.799 | DEFER |

ρ understates the true error, because its denominator is itself inflated by
noise, so `rho_defer` is calibrated to 0.4 rather than 1.

**Known limitation.** The layer does not detect severe under-resolution: at five
points per wavelength the relative error is around 0.4 and the verdict is still
GOOD. A guard for this was prototyped and removed — the evidence that motivated
it turned out to be a pathological evaluation point, and at an ordinary point the
recovered Hessian had 2.2% relative error against 2.9% for plain central
differences. `resolution_guard()` remains in the modules as an unvalidated
experimental diagnostic, is not called, and carries no claim of a
resolution-based guarantee. The behaviour is pinned by a test so that anyone
changing it reads the reasoning first.

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
  io.py                     NIfTI, MATLAB and numpy readers and writers
  __main__.py               command line
tests/                      58 tests, including the exactness and
                            no-extrapolation gates in both dimensions
examples/                   runnable scripts
directional_curvature.py    reproduction script for the manuscript
verify_proofs.py            symbolic verification of the proof steps
```

## Citing

Archived at https://doi.org/10.5281/zenodo.21793101. See `CITATION.cff`.

## License

MIT.
