"""Curvature at a point where only a 45-degree cone of directions is usable."""
import numpy as np
from dircurv import Geometry, measure

f = lambda p: np.exp(p[0] + 0.5 * p[1]) * np.cos(p[2]) + 0.2 * p[0] ** 3
print(measure(f, np.full(3, 0.1),
              Geometry.cone(3, half_angle_deg=45, max_span=0.3),
              sigma=1e-6, m=24))
