"""Implementation B: direct physical Frobenius basis phi(u) = Frobenius vector of u u^T.

No T' = T - cI reparametrisation, no b~ = sqrt(2) b rescaling, no D_theta.
Noise is simulated at the level of individual function evaluations.
"""
from __future__ import annotations

import numpy as np

from ..common import Design, frob_index, frob_to_sym, ray


def phi(u: np.ndarray) -> np.ndarray:
    n = len(u)
    return np.array([u[i] * u[j] * (1.0 if i == j else np.sqrt(2.0)) for i, j in frob_index(n)])


def rays(design: Design, theta: float) -> np.ndarray:
    return np.array([ray(v, theta) for v in design.V])


def gram_physical(design: Design, theta: float) -> np.ndarray:
    Phi = np.array([phi(u) for u in rays(design, theta)])
    return (Phi * design.w[:, None]).T @ Phi


def lambda_F(design: Design, theta: float):
    ev = np.linalg.eigvalsh(gram_physical(design, theta))
    return float(ev[0]), float(ev[-1])


def fit_H(design: Design, theta: float, q: np.ndarray) -> np.ndarray:
    n = design.d + 1
    Phi = np.array([phi(u) for u in rays(design, theta)])
    X = np.linalg.lstsq(Phi, q, rcond=None)[0]
    if X.ndim == 1:
        return frob_to_sym(X, n)
    return np.stack([frob_to_sym(X[:, k], n) for k in range(X.shape[1])])


def curvature_data(f, design: Design, theta: float, stencil, h: float) -> np.ndarray:
    """Noise-free one-sided stencil data q_i = h^-2 sum_j w_j f(j h u_i), x0 = 0."""
    U = rays(design, theta)
    n = U.shape[1]
    f0 = f(np.zeros(n))
    q = []
    for u in U:
        s = stencil.weights[0] * f0
        for j, wj in enumerate(stencil.weights[1:], start=1):
            s += wj * f(j * h * u)
        q.append(s / h**2)
    return np.array(q)


def noisy_curvature_data(q_clean: np.ndarray, stencil, sigma: float, h: float, R: int,
                         n_draws: int, rng: np.random.Generator, shared_base: bool = True) -> np.ndarray:
    """Add evaluation noise. Returns an (m, n_draws) array.

    shared_base=True : f(x0) evaluated R times, averaged, and reused on every ray (Assumption 5.1).
    shared_base=False: historical convention - every ray draws its own f(x0) noise.
    """
    m = q_clean.shape[0]
    r = len(stencil.weights)
    eps = rng.normal(0.0, sigma, size=(m, r - 1, n_draws))
    ray_part = np.einsum("j,mjk->mk", np.asarray(stencil.weights[1:]), eps)
    if shared_base:
        e0 = rng.normal(0.0, sigma, size=(R, n_draws)).mean(axis=0)
        base = stencil.w0 * np.broadcast_to(e0, (m, n_draws))
    else:
        base = stencil.w0 * rng.normal(0.0, sigma, size=(m, n_draws))
    return q_clean[:, None] + (ray_part + base) / h**2
