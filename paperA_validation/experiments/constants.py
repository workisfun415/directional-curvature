"""Conditioning constants: Corollaries 4.4, 4.5 and Proposition 4.6."""
from __future__ import annotations

import itertools
import math

import numpy as np
import sympy as sp
from scipy.optimize import minimize

from .. import common as C
from ..impl_a import factorized as A
from ..impl_b import physical as B


# ---------------------------------------------------------------- exact planar limit
def _sphere_moment(a, d):
    """E[omega^a] for omega uniform on S^{d-1}, exact rational."""
    if any(x % 2 for x in a):
        return sp.Integer(0)
    num = sp.Integer(1)
    for x in a:
        num *= sp.factorial2(x - 1) if x > 0 else 1
    den = sp.Integer(1)
    for k in range(sum(a) // 2):
        den *= (d + 2 * k)
    return num / den


def _ball_moment(a, d):
    s = sum(a)
    return sp.Rational(d, d + s) * _sphere_moment(a, d)


def _monomials(d, deg):
    return [a for a in itertools.product(range(deg + 1), repeat=d) if sum(a) == deg]


def uniform_ball_S0_exact(d: int) -> sp.Matrix:
    """Exact S0 (Schur complement of psi after regressing on (1, v)) for v uniform on B^d."""
    lin = [tuple(int(i == k) for i in range(d)) for k in range(d)]
    feats_l = [(sp.Integer(1), (0,) * d)] + [(sp.Integer(1), e) for e in lin]
    feats_q = []
    for i in range(d):
        for j in range(i, d):
            a = [0] * d
            a[i] += 1
            a[j] += 1
            feats_q.append((sp.Integer(1) if i == j else sp.sqrt(2), tuple(a)))

    def mom(f, g):
        (cf, af), (cg, ag) = f, g
        return cf * cg * _ball_moment(tuple(x + y for x, y in zip(af, ag)), d)

    Mll = sp.Matrix(len(feats_l), len(feats_l), lambda i, j: mom(feats_l[i], feats_l[j]))
    Mlq = sp.Matrix(len(feats_l), len(feats_q), lambda i, j: mom(feats_l[i], feats_q[j]))
    Mqq = sp.Matrix(len(feats_q), len(feats_q), lambda i, j: mom(feats_q[i], feats_q[j]))
    return sp.simplify(Mqq - Mlq.T * Mll.inv() * Mlq)


def uniform_ball_lambda_exact(d: int):
    S0 = uniform_ball_S0_exact(d)
    ev = [sp.nsimplify(sp.simplify(e)) for e in S0.eigenvals().keys()]
    return min(ev, key=lambda e: float(e))


def predicted_uniform(d: int):
    return sp.Rational(4, (d + 2) ** 2 * (d + 4))


def predicted_optimal(d: int):
    return sp.Rational(1, 4) if d == 1 else sp.Rational(2, (d + 2) ** 2)


def kappa_limit_uniform(n: int) -> float:
    return (n + 1) * math.sqrt(n + 3) / 2


def kappa_limit_optimal(n: int) -> float:
    return 2.0 if n == 2 else (n + 1) / math.sqrt(2)


# ---------------------------------------------------------------- finite-theta ladders
def ladder(design_fn, thetas, impl="A"):
    """lambda_min^F/theta^4, lambda_max^F and kappa_F theta^2 along a theta ladder."""
    out = []
    for th in thetas:
        des = design_fn(th)
        lmin, lmax = (A.lambda_F if impl == "A" else B.lambda_F)(des, th)
        out.append({"theta": th, "lmin_over_theta4": lmin / th**4, "lmax": lmax,
                    "kappa_theta2": math.sqrt(lmax / lmin) * th**2})
    return out


def intercept_theta2(rows, key="lmin_over_theta4"):
    """Fit y = a + b theta^2 and return a (the theta -> 0 limit)."""
    th = np.array([r["theta"] for r in rows])
    y = np.array([r[key] for r in rows])
    X = np.c_[np.ones_like(th), th**2]
    return float(np.linalg.lstsq(X, y, rcond=None)[0][0])


def S0_numeric(design: C.Design) -> float:
    return float(np.linalg.eigvalsh(A.schur_S0(design))[0])


# ---------------------------------------------------------------- design search (best found)
def design_search(m: int, theta: float, restarts: int, rng: np.random.Generator):
    best, best_x = -np.inf, None
    for _ in range(restarts):
        x0 = np.r_[rng.uniform(0, 1, m), rng.uniform(0, 2 * np.pi, m)]

        def neg(x):
            r = np.clip(x[:m], 0, 1)
            V = np.c_[r * np.cos(x[m:]), r * np.sin(x[m:])]
            lmin, _ = B.lambda_F(C.Design("s", V, np.full(m, 1 / m)), theta)
            return -lmin / theta**4

        res = minimize(neg, x0, method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-13})
        if -res.fun > best:
            best, best_x = -res.fun, res.x
    r = np.clip(best_x[:m], 0, 1)
    V = np.c_[r * np.cos(best_x[m:]), r * np.sin(best_x[m:])]
    lim = S0_numeric(C.Design("best", V, np.full(m, 1 / m)))
    return {"m": m, "theta": theta, "best_found_lmin_over_theta4": best,
            "best_design_planar_points": V.tolist(), "best_design_S0_limit": lim,
            "label": "numerical best found"}
