"""Implementation A for quadratic regression (Section 7): transformed coordinates.

Parameters w = (f0, g_n, g_t, c, b~, t') with the conical scaling written explicitly:

    f(x0 + h r u) = f0 + h r (g_n u_n + theta g_t.v)
                    + (h r)^2/2 * (c + sqrt2 theta u_n b~.v + theta^2 v^T T' v),

so the design matrix factors as X = X~(theta) D_theta with
D_theta = diag(1, 1, theta I_d, 1, theta I_d, theta^2 I_{d(d+1)/2}).

A regression design is a list of rays (planar point v) with per-ray radii (common radii =
product design, ray-dependent radii = non-product design), plus an optional base point x0.
"""
from __future__ import annotations

import math

import numpy as np

from ..common import SQRT2
from .factorized import dims, psi, z_to_H


class RegressionDesign:
    def __init__(self, V, radii, base_point=True, name=""):
        self.V = np.asarray(V, float)
        self.radii = [tuple(float(r) for r in rs) for rs in radii]
        self.base_point = base_point
        self.name = name

    @property
    def d(self):
        return self.V.shape[1]

    @property
    def n_points(self):
        return sum(len(r) for r in self.radii) + (1 if self.base_point else 0)

    def is_product(self):
        return len(set(self.radii)) == 1

    def points(self, theta, h):
        """Physical sample offsets y = h r u(v, theta) (base point first if present)."""
        P = [np.zeros(self.d + 1)] if self.base_point else []
        for v, rs in zip(self.V, self.radii):
            un = math.sqrt(1 - theta**2 * float(v @ v))
            u = np.r_[theta * v, un]
            P += [h * r * u for r in rs]
        return np.array(P)


def _scaled_rows(des: RegressionDesign, theta, h):
    d = des.d
    rows = []
    if des.base_point:
        rows.append(np.r_[1.0, 0.0, np.zeros(d), 0.0, np.zeros(d), np.zeros(d * (d + 1) // 2)])
    for v, rs in zip(des.V, des.radii):
        un = math.sqrt(1 - theta**2 * float(v @ v))
        for r in rs:
            rows.append(np.r_[1.0, h * r * un, h * r * v, 0.5 * (h * r) ** 2,
                              0.5 * (h * r) ** 2 * SQRT2 * un * v, 0.5 * (h * r) ** 2 * psi(v)])
    return np.array(rows)


def D_theta_reg(d, theta):
    k2 = d * (d + 1) // 2
    return np.diag(np.r_[1.0, 1.0, np.full(d, theta), 1.0, np.full(d, theta), np.full(k2, theta**2)])


def design_matrix(des, theta, h):
    return _scaled_rows(des, theta, h) @ D_theta_reg(des.d, theta)


def _split(w, d):
    """Return (f0, g, z) with z = (c, b~, t') in the ordering used by factorized.z_to_H."""
    k2 = d * (d + 1) // 2
    f0 = w[0]
    g_n, g_t = w[1], w[2:2 + d]
    c = w[2 + d]
    bt = w[3 + d:3 + 2 * d]
    tp = w[3 + 2 * d:3 + 2 * d + k2]
    return f0, np.r_[g_t, g_n], np.r_[c, bt, tp]


def fit_H(des, theta, h, fvals):
    """Least-squares Hessian from function values (fvals may be (N,) or (N, K))."""
    X = design_matrix(des, theta, h)
    W = np.linalg.lstsq(X, fvals, rcond=None)[0]
    n = des.d + 1
    if W.ndim == 1:
        return z_to_H(_split(W, des.d)[2], n)
    return np.stack([z_to_H(_split(W[:, k], des.d)[2], n) for k in range(W.shape[1])])


def cov_physical_frob(des, theta, h, sigma):
    """Exact covariance of the physical-Frobenius Hessian estimate (i.i.d. evaluation noise)."""
    from .factorized import P_matrix
    X = design_matrix(des, theta, h)
    Cw = sigma**2 * np.linalg.inv(X.T @ X)
    d = des.d
    k0 = 2 + d
    sl = slice(k0, k0 + 1 + d + d * (d + 1) // 2)
    P = P_matrix(d + 1)
    return P @ Cw[sl, sl] @ P.T


# ---------------------------------------------------------------- alias matrix (moment formula)
def cubic_sectors(T3, v):
    """P_0..P_3 of Section 7.1 at planar point v for a symmetric 3-tensor in R^{d+1}."""
    d = len(v)
    N = d  # axial index
    P0 = T3[N, N, N]
    P1 = 3 * sum(T3[a, N, N] * v[a] for a in range(d))
    P2 = 3 * sum(T3[a, b, N] * v[a] * v[b] for a in range(d) for b in range(d))
    P3 = sum(T3[a, b, c] * v[a] * v[b] * v[c] for a in range(d) for b in range(d) for c in range(d))
    return np.array([P0, P1, P2, P3])


def alias_limit_moment(des, T3, j):
    """Leading alias coefficients of cubic sector j, from design moments at theta = 0.

    Returns the sector-wise Frobenius norms (axial, mixed [b~], tangential [t'])
    of the limit  lim theta^{k-j} Bias_k / h  predicted by Theorem 7.1, obtained as
    S^{-1} C from the moment (normal-equation) matrices of the theta = 0 scaled design.
    Only the entries with j < k are alias terms; the others are reported for completeness.
    """
    d = des.d
    X0 = _scaled_rows(des, 0.0, 1.0)
    data = []
    if des.base_point:
        data.append(0.0)
    for v, rs in zip(des.V, des.radii):
        Pj = cubic_sectors(T3, v)[j]
        data += [r**3 * Pj / 6 for r in rs]
    data = np.array(data)
    M = X0.T @ X0                     # moment matrix
    b = X0.T @ data                   # cross moments with the omitted cubic sector
    w = np.linalg.solve(M, b)         # = S^{-1} C blockwise (Schur complement of the full system)
    _, _, z = _split(w, d)
    k2 = d * (d + 1) // 2
    return np.array([abs(z[0]), np.linalg.norm(z[1:1 + d]), np.linalg.norm(z[1 + d:1 + d + k2])])
