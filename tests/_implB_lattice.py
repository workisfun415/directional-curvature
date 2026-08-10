"""Implementation B of the lattice path, written from the definitions.

Deliberately different from src/dircurv/lattice.py:
  * directions enumerated explicitly, no gcd reduction
  * probe formed as a Lagrange divided difference on the three node positions
  * normal equations solved by Cholesky, not lstsq
  * the physical direction built by an explicit index->coordinate map rather
    than by reversing an array

That last difference is the point. Two silent index-order bugs have already
occurred in this work, so B constructs the coordinate mapping from scratch.
"""
import numpy as np
from math import gcd


def dirs_B(ndim, order):
    """All integer vectors with components in [-order, order], minus multiples
    of shorter ones. Enumerated and filtered explicitly."""
    rng = list(range(-order, order + 1))
    if ndim == 2:
        cand = [(a, b) for a in rng for b in rng if (a, b) != (0, 0)]
    else:
        cand = [(a, b, c) for a in rng for b in rng for c in rng
                if (a, b, c) != (0, 0, 0)]
    keep = []
    for v in cand:
        g = 0
        for c in v:
            g = gcd(g, abs(c))
        if g == 1:
            keep.append(v)
    return keep


def phys_dir_B(vec, spacing):
    """Index vector -> physical unit direction, built coordinate by coordinate.

    The array axis k corresponds to physical coordinate (ndim-1-k): axis 0 is
    the slowest index (z in 3D, y in 2D) and the last axis is x. So physical
    component j comes from array axis (ndim-1-j).
    """
    ndim = len(vec)
    phys = np.zeros(ndim)
    length_sq = 0.0
    for j in range(ndim):                       # j = 0 is x, 1 is y, 2 is z
        axis = ndim - 1 - j
        phys[j] = vec[axis] * spacing[axis]
        length_sq += phys[j] ** 2
    L = np.sqrt(length_sq)
    return L, phys / L


def probe_B(A, index, vec, s, R=0.5):
    """Second divided difference on the three node positions, via Lagrange."""
    i0 = np.asarray(index, int)
    v = np.asarray(vec, int)
    pts = [0.0, float(s), float(2 * s)]         # in lattice steps
    ys = [A[tuple(i0)], A[tuple(i0 + s * v)], A[tuple(i0 + 2 * s * v)]]
    dd = 0.0
    for i in range(3):
        den = 1.0
        for j in range(3):
            if j != i:
                den *= (pts[i] - pts[j])
        dd += ys[i] / den
    return 2.0 * dd                             # in units of (lattice step)^-2


def design_B(units):
    U = np.atleast_2d(units)
    n = U.shape[1]
    rows = []
    for u in U:
        r = []
        for i in range(n):
            r.append(u[i] * u[i])
        for i in range(n):
            for j in range(i + 1, n):
                r.append(np.sqrt(2) * u[i] * u[j])
        rows.append(r)
    return np.array(rows)


def unvec_B(h, n):
    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = h[i]
    k = n
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = h[k] / np.sqrt(2)
            k += 1
    return H


def measure_B(A, index, spacing, mask=None, order=2, s=1):
    ndim = A.ndim
    sp = np.atleast_1d(np.asarray(spacing, float))
    if sp.size == 1:
        sp = np.repeat(sp, ndim)
    mask = np.ones_like(A, bool) if mask is None else mask
    i0 = np.asarray(index, int)
    units, q = [], []
    for vec in dirs_B(ndim, order):
        v = np.asarray(vec, int)
        ok = True
        for step in (s, 2 * s):
            p = i0 + step * v
            if not all(0 <= p[k] < A.shape[k] for k in range(ndim)) \
                    or not mask[tuple(p)]:
                ok = False
                break
        if not ok:
            continue
        L, u = phys_dir_B(vec, sp)
        # probe_B is in lattice-step units; convert to physical
        q.append(probe_B(A, index, vec, s) / (L ** 2))
        units.append(u)
    if len(units) < ndim * (ndim + 1) // 2:
        return None, np.inf, len(units)
    D = design_B(np.array(units))
    q = np.array(q)
    G = D.T @ D
    c = np.linalg.solve(G, D.T @ q)
    return unvec_B(c, ndim), float(np.linalg.cond(D)), len(units)
