#!/usr/bin/env python3
"""
lattice.py
==========
Interpolation-free measurement on a grid.

Why this exists. The bicubic and tricubic paths need a complete local
neighbourhood -- 16 pixels in 2D, 64 voxels in 3D -- while a central-difference
stencil needs only 9 or 27. Measured on a masked disc and a masked sphere, that
made the interpolating path *strictly more demanding* than central differences:
there was no pixel or voxel anywhere where it could be applied and central
differences could not. The one-sided advantage the theory predicts was being
cancelled by the interpolation support.

The fix is to stop interpolating. Directions are restricted to lattice vectors
and spans to integer multiples of the voxel size, so every probe point falls
exactly on a grid node. A ray then needs only the nodes it actually visits, and
a one-sided ray needs nothing on the far side of the evaluation point. That is
what restores the boundary behaviour.

Cost of the restriction: directions come from a fixed lattice set rather than
being freely placed, so the conditioning is slightly worse than an optimally
placed set would give. On a grid that freedom was never real.

Note that with R = 1/2 and nodes {0, s, 2s} the probe is the classical one-sided
second difference on equally spaced nodes -- see the divided-difference identity
in the accompanying manuscript. No novelty is claimed for the stencil; what the
framework contributes is the choice of directions and span, and the reporting of
what the local geometry allows.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["lattice_directions", "measure_lattice", "feasible_lattice",
           "LatticeReport", "R_LATTICE"]

R_LATTICE = 0.5          # nodes {0, s, 2s}: the interior node is a grid point


# =====================================================================
# Direction sets on the lattice
# =====================================================================

def _primitive(vecs):
    out, seen = [], set()
    for v in vecs:
        g = 0
        for c in v:
            g = gcd(g, abs(int(c)))
        if g == 0:
            continue
        p = tuple(int(c) // g for c in v)
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def lattice_directions(ndim, order=2) -> List[Tuple[int, ...]]:
    """Primitive integer direction vectors with components in [-order, order].

    order=1 gives 8 directions in 2D and 26 in 3D; order=2 gives 24 and 98.
    Antipodal partners are always both present, so the attainable order is set
    by which of them survive the mask, not by the direction set.
    """
    rng = range(-order, order + 1)
    if ndim == 2:
        cand = [(p, q) for p in rng for q in rng if (p, q) != (0, 0)]
    elif ndim == 3:
        cand = [(p, q, r) for p in rng for q in rng for r in rng
                if (p, q, r) != (0, 0, 0)]
    else:
        raise ValueError("ndim must be 2 or 3")
    return _primitive(cand)


def _physical(vec, spacing):
    """Lattice vector -> physical step length and unit direction.

    The array is indexed [z, y, x] (3D) or [y, x] (2D), and `spacing` follows
    that same order, so a lattice step of `vec` moves by vec * spacing in INDEX
    order. The Hessian, however, is expressed in PHYSICAL order (x, y) or
    (x, y, z), so the components must be reversed before the unit direction is
    formed. Getting this wrong swaps H11 with H22 (or H33) and produces a
    plausible but wrong answer with no error raised -- it cost a factor of ten
    in accuracy before it was caught.
    """
    step_index = np.asarray(vec, float) * np.asarray(spacing, float)
    L = float(np.linalg.norm(step_index))
    step_phys = step_index[::-1]                 # index order -> physical order
    return L, step_phys / L


# =====================================================================
# Feasibility on the lattice
# =====================================================================

def _inside(idx, shape):
    return all(0 <= idx[k] < shape[k] for k in range(len(shape)))


def _node_ok(arr_mask, idx):
    return _inside(idx, arr_mask.shape) and bool(arr_mask[tuple(idx)])


def max_steps(mask, index, vec, s_cap):
    """Largest s such that x0 + s*vec and x0 + 2s*vec are both valid nodes."""
    idx = np.asarray(index, int)
    v = np.asarray(vec, int)
    best = 0
    for s in range(1, s_cap + 1):
        if _node_ok(mask, idx + s * v) and _node_ok(mask, idx + 2 * s * v):
            best = s
        else:
            break
    return best


def feasible_lattice(mask, index, order=2, s_cap=4):
    """Lattice directions usable at this node, with the step count for each."""
    ndim = mask.ndim
    out = []
    for vec in lattice_directions(ndim, order):
        s = max_steps(mask, index, vec, s_cap)
        if s > 0:
            out.append((vec, s))
    return out


# =====================================================================
# Measurement
# =====================================================================

def _design(units):
    U = np.atleast_2d(np.asarray(units, float))
    n = U.shape[1]
    r2 = np.sqrt(2)
    cols = [U[:, i] ** 2 for i in range(n)]
    cols += [r2 * U[:, i] * U[:, j] for i in range(n) for j in range(i + 1, n)]
    return np.column_stack(cols)


def _unvec(h, n):
    r2 = np.sqrt(2)
    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = h[i]
    k = n
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = h[k] / r2
            k += 1
    return H


@dataclass
class LatticeReport:
    index: tuple
    n_directions: int
    kappa: float
    H_hat: Optional[np.ndarray]
    antipodal_sampled: float
    expected_order: int
    span_min: float
    span_max: float
    verdict: str
    flags: list

    def __str__(self):
        L = [f"node                 {self.index}",
             f"lattice directions   {self.n_directions}",
             f"kappa                {self.kappa:.3f}",
             f"antipodal sampled    {self.antipodal_sampled:.0%}",
             f"expected order       {self.expected_order}",
             f"span min/max         {self.span_min:.4g} / {self.span_max:.4g}"]
        if self.H_hat is not None:
            L.append("Hessian:")
            for row in np.atleast_2d(self.H_hat):
                L.append("   " + "  ".join(f"{v: .6g}" for v in row))
        for f in self.flags:
            L.append(f"  ! {f}")
        L.append(f"verdict              {self.verdict}")
        return "\n".join(L)


def measure_lattice(array, index, spacing, mask=None, order=2, s_cap=4,
                    sigma=0.0, prefer_short=True) -> LatticeReport:
    """Measure the Hessian at a grid node using only grid nodes.

    No interpolation is performed anywhere, so the support is exactly the nodes
    visited by each ray.
    """
    A = np.asarray(array, float)
    ndim = A.ndim
    if ndim not in (2, 3):
        raise ValueError("array must be 2D or 3D")
    mask = np.ones_like(A, bool) if mask is None else np.asarray(mask, bool)
    sp = np.atleast_1d(np.asarray(spacing, float))
    if sp.size == 1:
        sp = np.repeat(sp, ndim)
    index = tuple(int(v) for v in index)
    dmin = ndim * (ndim + 1) // 2
    flags = []

    if not mask[index]:
        return LatticeReport(index, 0, np.inf, None, 0.0, 0, 0.0, 0.0,
                             "UNUSABLE", ["node is outside the mask"])

    usable = feasible_lattice(mask, index, order=order, s_cap=s_cap)
    if len(usable) < dmin:
        return LatticeReport(index, len(usable), np.inf, None, 0.0, 0, 0.0, 0.0,
                             "UNUSABLE",
                             [f"only {len(usable)} lattice directions usable; "
                              f"{dmin} are needed for identifiability"])

    R = R_LATTICE
    units, q, spans, vecs = [], [], [], []
    for vec, smax in usable:
        s = 1 if prefer_short else smax
        L, u = _physical(vec, sp)
        h = 2.0 * s * L                       # outer node at 2s lattice steps
        i0 = np.asarray(index, int)
        v = np.asarray(vec, int)
        f0 = A[tuple(i0)]
        fR = A[tuple(i0 + s * v)]
        f1 = A[tuple(i0 + 2 * s * v)]
        q.append(2.0 * ((1 - R) * f0 + R * f1 - fR) / (R * (1 - R) * h * h))
        units.append(u)
        spans.append(h)
        vecs.append(tuple(vec))

    units = np.array(units)
    q = np.array(q)
    D = _design(units)
    kappa = float(np.linalg.cond(D))
    if not np.isfinite(kappa) or kappa > 1e8:
        return LatticeReport(index, len(usable), kappa, None, 0.0, 0,
                             float(min(spans)), float(max(spans)), "UNUSABLE",
                             ["the usable lattice directions do not span the "
                              "space of symmetric matrices"])

    H = _unvec(np.linalg.lstsq(D, q, rcond=None)[0], ndim)

    vs = set(vecs)
    anti = sum(1 for v in vecs if tuple(-np.array(v)) in vs) / len(vecs)
    order_attained = 2 if anti > 0.999 else 1

    if order_attained == 1:
        flags.append("antipodal partners not sampled at this node: first order")
    if kappa > 5:
        flags.append(f"conditioning {kappa:.1f}: the direction set surviving the "
                     "mask is poorly spread")

    verdict = ("GOOD" if (order_attained == 2 and kappa < 3 and not flags)
               else "CAUTION" if kappa < 20 else "LOW")
    return LatticeReport(index, len(usable), kappa, H, anti, order_attained,
                         float(min(spans)), float(max(spans)), verdict, flags)
