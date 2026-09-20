"""Implementation B for quadratic regression: physical coordinates only.

Model f(x0 + y) = f0 + g.y + 1/2 y^T H y with H in physical Frobenius coordinates.
No T' shear, no b~ rescaling, no theta scaling: the design matrix is built from the
physical sample offsets y directly.
"""
from __future__ import annotations

import numpy as np

from ..common import frob_to_sym, frob_index


def features(y):
    n = len(y)
    quad = [0.5 * y[i] * y[j] * (1.0 if i == j else 2.0 / np.sqrt(2.0)) for i, j in frob_index(n)]
    return np.r_[1.0, y, quad]


def fit_H(points, fvals):
    """points: (N, n) physical offsets from x0; returns the fitted Hessian(s)."""
    n = points.shape[1]
    X = np.array([features(y) for y in points])
    W = np.linalg.lstsq(X, fvals, rcond=None)[0]
    k = 1 + n
    if W.ndim == 1:
        return frob_to_sym(W[k:], n)
    return np.stack([frob_to_sym(W[k:, i], n) for i in range(W.shape[1])])


def cov_physical_frob(points, sigma):
    n = points.shape[1]
    X = np.array([features(y) for y in points])
    C = sigma**2 * np.linalg.inv(X.T @ X)
    k = 1 + n
    return C[k:, k:]
