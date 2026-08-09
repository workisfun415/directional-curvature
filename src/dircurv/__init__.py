"""
dircurv -- directional curvature measurement under restricted sampling
=====================================================================

Two entry points, for two situations.

`dircurv.analytic` -- the function is callable and the feasible region is
described analytically (a cone, slab, wedge or free space). Use this for
derivative-free optimisation, for method development, and for reproducing the
results in the accompanying paper.

`dircurv.grid2d` and `dircurv.grid3d` -- the data is a measured 2D array or 3D
volume with a validity mask. Use these for images, measured fields and imaging
volumes. Feasible directions are derived from the
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

Before relying on the output, read `dircurv.analytic.when_not_to_use()`.
"""
__version__ = "0.2.0"

from . import analytic, grid2d, grid3d   # noqa: F401
from .analytic import Geometry, measure, when_not_to_use, kappa_reference  # noqa: F401
from .grid2d import (GridField, MaskSupportError, measure_pixel,  # noqa: F401
                     reliability_maps)
from .grid3d import (VolumeField, MaskSupportError3D,  # noqa: F401
                     measure_voxel, reliability_volumes, coverage_fraction)

__all__ = ["analytic", "grid2d", "grid3d", "Geometry", "measure",
           "when_not_to_use", "kappa_reference", "GridField",
           "MaskSupportError", "measure_pixel", "reliability_maps",
           "VolumeField", "MaskSupportError3D", "measure_voxel",
           "reliability_volumes", "coverage_fraction", "__version__"]
