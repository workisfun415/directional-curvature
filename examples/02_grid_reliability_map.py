"""Reliability maps for a measured 2D field inside an irregular mask."""
import numpy as np
from dircurv import GridField, reliability_maps
from dircurv.grid2d import VERDICT_CODE

n, sp = 61, 0.02
yy, xx = np.mgrid[0:n, 0:n]
X, Y = (xx - n // 2) * sp, (yy - n // 2) * sp
field = np.exp(X) * np.cos(2 * Y) + 0.3 * X ** 3
mask = (np.hypot(X, Y) < 0.4) & ~((np.abs(Y) < 0.1) & (X > 0))

maps = reliability_maps(GridField(field, spacing=sp, mask=mask),
                        m=12, sigma=1e-4, step=3)
names = {v: k for k, v in VERDICT_CODE.items()}
v = maps["verdict"]
for code in sorted(set(v[v > 0].ravel().tolist())):
    print(f"{names[code]:<10} {int((v == code).sum()):>5} pixels")
print("aperture range:", np.nanmin(maps["aperture"]), "to",
      np.nanmax(maps["aperture"]), "deg")
