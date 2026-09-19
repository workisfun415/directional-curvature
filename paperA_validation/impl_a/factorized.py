"""Implementation A: transformed coordinates z = (c, b~, t'), Lemma 3.3.

Everything here is computed from the factorisation A_theta = A~(theta) D_theta,
the averaged Gram matrix G(theta), the shear map P : z -> H, and the Schur
complement S. Physical-Frobenius conditioning is obtained from the generalised
eigenproblem  M z = lambda (P^T P) z, i.e. with the physical norm ||Pz||_F.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh

from ..common import SQRT2, Design, frob_index, frob_to_sym, sym_to_frob


def psi(v: np.ndarray) -> np.ndarray:
    d = len(v)
    return np.array([v[i] * v[j] * (1.0 if i == j else SQRT2) for i, j in frob_index(d)])


def dims(n: int):
    d = n - 1
    return 1, d, d * (d + 1) // 2


def a_tilde(v: np.ndarray, theta: float) -> np.ndarray:
    un = math.sqrt(1.0 - theta**2 * float(v @ v))
    return np.r_[1.0, SQRT2 * un * v, psi(v)]


def D_theta(n: int, theta: float) -> np.ndarray:
    k0, k1, k2 = dims(n)
    return np.diag(np.r_[np.ones(k0), np.full(k1, theta), np.full(k2, theta**2)])


def gram(design: Design, theta: float) -> np.ndarray:
    A = np.array([a_tilde(v, theta) for v in design.V])
    return (A * design.w[:, None]).T @ A


def design_matrix(design: Design, theta: float) -> np.ndarray:
    n = design.d + 1
    A = np.array([a_tilde(v, theta) for v in design.V])
    return A @ D_theta(n, theta)


def P_matrix(n: int) -> np.ndarray:
    """Linear map z -> physical Frobenius vector of H = [[T'+cI, b], [b^T, c]]."""
    d = n - 1
    k0, k1, k2 = dims(n)
    N = k0 + k1 + k2
    P = np.zeros((n * (n + 1) // 2, N))
    for col in range(N):
        z = np.zeros(N)
        z[col] = 1.0
        P[:, col] = sym_to_frob(z_to_H(z, n))
    return P


def z_to_H(z: np.ndarray, n: int) -> np.ndarray:
    d = n - 1
    k0, k1, k2 = dims(n)
    c = z[0]
    b = z[k0:k0 + k1] / SQRT2
    Tp = frob_to_sym(z[k0 + k1:], d) if d > 0 else np.zeros((0, 0))
    H = np.zeros((n, n))
    H[:d, :d] = Tp + c * np.eye(d)
    H[:d, n - 1] = H[n - 1, :d] = b
    H[n - 1, n - 1] = c
    return H


def lambda_F(design: Design, theta: float):
    """(lambda_min^F, lambda_max^F) in the physical Frobenius metric."""
    n = design.d + 1
    D = D_theta(n, theta)
    M = D @ gram(design, theta) @ D
    P = P_matrix(n)
    ev = eigh(M, P.T @ P, eigvals_only=True)
    return float(ev[0]), float(ev[-1])


def schur_S0(design: Design) -> np.ndarray:
    n = design.d + 1
    k0, k1, _ = dims(n)
    G = gram(design, 0.0)
    s = k0 + k1
    G11, G12, G22 = G[:s, :s], G[:s, s:], G[s:, s:]
    return G22 - G12.T @ np.linalg.solve(G11, G12)


def fit_H(design: Design, theta: float, q: np.ndarray) -> np.ndarray:
    """Least-squares estimate in z coordinates, mapped to physical H. q may be (m,) or (m, K)."""
    n = design.d + 1
    A = design_matrix(design, theta)
    Z = np.linalg.lstsq(A, q, rcond=None)[0]
    if Z.ndim == 1:
        return z_to_H(Z, n)
    return np.stack([z_to_H(Z[:, k], n) for k in range(Z.shape[1])])


def sector_variances(design: Design, theta: float, stencil, sigma: float, h: float, R: int = 1):
    """Theorem 5.2 via the exact covariance D^-1 G^-1 D^-1 (ray part) plus the common I_n term.

    Returns expected squared Frobenius error per physical sector (axial, mixed, tangential).
    Requires uniform design weights (a finite design of m rays).
    """
    n = design.d + 1
    d = n - 1
    m = design.m
    D = D_theta(n, theta)
    Dinv = np.linalg.inv(D)
    G = gram(design, theta)
    cov_z = stencil.s2_ray * sigma**2 / (m * h**4) * Dinv @ np.linalg.inv(G) @ Dinv
    P = P_matrix(n)
    cov_f = P @ cov_z @ P.T
    # common-mode: error (w0 eps0bar / h^2) I_n, variance sigma^2 / R
    e = sym_to_frob(np.eye(n))
    cov_f = cov_f + stencil.w0**2 * sigma**2 / (R * h**4) * np.outer(e, e)
    return _sector_traces(cov_f, n)


def _sector_traces(cov_f: np.ndarray, n: int) -> np.ndarray:
    d = n - 1
    idx = frob_index(n)
    ax = [k for k, (i, j) in enumerate(idx) if i == j == n - 1]
    mx = [k for k, (i, j) in enumerate(idx) if i != j and j == n - 1]
    tg = [k for k, (i, j) in enumerate(idx) if i < d and j < d]
    diag = np.diag(cov_f)
    return np.array([diag[ax].sum(), diag[mx].sum(), diag[tg].sum()])


def D_prime(design: Design, stencil) -> np.ndarray:
    """D'_k = s'^2 tr[(G0^{-1})_kk] in z coordinates."""
    n = design.d + 1
    k0, k1, k2 = dims(n)
    Gi = np.linalg.inv(gram(design, 0.0))
    blocks = [slice(0, k0), slice(k0, k0 + k1), slice(k0 + k1, k0 + k1 + k2)]
    return np.array([stencil.s2_ray * np.trace(Gi[b, b]) for b in blocks])
