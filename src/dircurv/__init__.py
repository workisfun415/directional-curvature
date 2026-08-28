"""
dircurv -- directional curvature measurement under restricted sampling
=====================================================================

Two entry points, for two situations.

`dircurv.analytic` -- the function is callable and the feasible region is
described analytically (a cone, slab, wedge or free space). Use this for
derivative-free optimisation, for method development, and for reproducing the
results in the accompanying paper.

`dircurv.lattice` -- measured data on a grid, with probe points restricted to
grid nodes so that NO interpolation is used. This is the right choice near a
mask boundary: the support is exactly the nodes each ray visits, so a one-sided
ray needs nothing on the far side. Measured on a masked sphere it reaches about
22 percent more voxels than a central-difference stencil, including 4100 voxels
where central differences cannot be applied at all.

`dircurv.grid2d` and `dircurv.grid3d` -- the same measurement with bicubic or
tricubic interpolation, so directions and spans are free rather than restricted
to the lattice. Use these when sampling off-grid genuinely matters. Note that
the interpolation support is 4x4 or 4x4x4, which is LARGER than a
central-difference stencil, so near a mask boundary these paths give up first
and offer no boundary advantage. Feasible directions are derived from the
mask by ray-marching; interpolation is bicubic and refuses to extrapolate
outside the mask.

Neither estimates stiffness, and neither competes with an inversion. What they
return is curvature together with a reliability assessment: the conditioning of
the locally available geometry, the attainable order of accuracy, and whether the
third-order term can be estimated at all.

Quick start
-----------
    import numpy as np
    from dircurv.analytic import Geometry, measure

    f = lambda p: np.exp(p[0]) * np.cos(p[1])
    g = Geometry.cone(dim=2, half_angle_deg=60, max_span=0.3)
    print(measure(f, [0.1, 0.1], g, sigma=1e-6, m=12))

    from dircurv.grid2d import GridField, reliability_maps
    field = GridField(array, spacing=0.002, mask=mask)
    maps = reliability_maps(field, m=16, sigma=1e-4)

There is also a command line, so a measured volume can be processed without
writing any code:

    python -m dircurv volume.nii.gz --mask brain.nii.gz --sigma-relative 0.01 \
        --out maps/ --coverage-only        # check what is measurable first
    python -m dircurv volume.nii.gz --mask brain.nii.gz --sigma-relative 0.01 \
        --out maps/

Before relying on the output, read `dircurv.analytic.when_not_to_use()`.
"""
__version__ = "0.6.0"

from . import analytic, grid2d, grid3d, lattice, fusion, io   # noqa: F401
from .analytic import Geometry, measure, when_not_to_use, kappa_reference  # noqa: F401
from .grid2d import (GridField, MaskSupportError, measure_pixel,  # noqa: F401
                     reliability_maps)
from .grid3d import (VolumeField, MaskSupportError3D,  # noqa: F401
                     measure_voxel, reliability_volumes, coverage_fraction)
from .lattice import (measure_lattice, feasible_lattice,  # noqa: F401
                      reliability_lattice)
from .fusion import fuse_hessians, sweep_weights  # noqa: F401
from .io import load_field, load_mask, save_maps, describe  # noqa: F401

__all__ = ["analytic", "grid2d", "grid3d", "lattice", "measure_lattice",
           "feasible_lattice", "reliability_lattice", "fusion", "fuse_hessians",
           "sweep_weights", "Geometry", "measure",
           "when_not_to_use", "kappa_reference", "GridField",
           "MaskSupportError", "measure_pixel", "reliability_maps",
           "VolumeField", "MaskSupportError3D", "measure_voxel",
           "reliability_volumes", "coverage_fraction", "io", "load_field",
           "load_mask", "save_maps", "describe", "__version__"]
