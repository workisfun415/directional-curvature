"""Reliability volumes for a measured 3D field inside a masked sphere."""
import numpy as np
from dircurv import VolumeField, reliability_volumes, coverage_fraction
from dircurv.grid3d import VERDICT_CODE

n, sp = 25, 0.04
zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
X, Y, Z = (xx - n//2)*sp, (yy - n//2)*sp, (zz - n//2)*sp
vol = np.exp(X + 0.5*Y) * np.cos(2*Z) + 0.2*X**3
mask = np.sqrt(X**2 + Y**2 + Z**2) < 0.32

field = VolumeField(vol, spacing=sp, mask=mask)
print(f"coverage: {100*coverage_fraction(field, step=3):.1f}% of masked voxels")

vols = reliability_volumes(field, m=24, sigma=1e-4, step=4)
names = {v: k for k, v in VERDICT_CODE.items()}
v = vols["verdict"]
for code in sorted(set(v[v > 0].ravel().tolist())):
    print(f"{names[code]:<10} {int((v == code).sum()):>5} voxels")
