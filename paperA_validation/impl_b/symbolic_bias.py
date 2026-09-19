"""Exact symbolic derivation of stencil constants and the Lemma 6.2 sector-bias coefficients.

Works in plain b coordinates (physical H), so coefficients are exact rationals.
The derivation is independent of the formula in Lemma 6.2: it expands
D^nu f[u^nu] in theta and matches the theta^0, theta^1, theta^2 coefficients
against the directional model u^T H u written directly in H.
"""
from __future__ import annotations

import itertools
import math

import sympy as sp


def stencil_constants(weights, p):
    """Exact moment conditions, beta, s_w^2 and s_w'^2 for a one-sided stencil on nodes 0..r-1."""
    w = [sp.Rational(x) for x in weights]
    moments = {l: sum(wj * sp.Integer(j) ** l for j, wj in enumerate(w)) / sp.factorial(l)
               for l in range(p + 3)}
    beta = moments[p + 2]
    ok = all(moments[l] == (1 if l == 2 else 0) for l in range(p + 2))
    s2 = sum(x * x for x in w)
    return {"moment_conditions_hold": bool(ok), "beta": beta, "s2": s2, "s2_ray": s2 - w[0] ** 2}


def _sym_tensor(n, nu):
    """Symbolic fully symmetric tensor: one symbol per sorted multi-index."""
    syms = {}
    for idx in itertools.combinations_with_replacement(range(n), nu):
        syms[idx] = sp.Symbol("f_" + "".join(str(i + 1) for i in idx))
    return syms


def sector_bias_symbolic(n: int, nu: int):
    """Return exact (c, b, T) leading bias per unit beta*h^p, and the Lemma 6.2 prediction."""
    d = n - 1
    th = sp.Symbol("theta", positive=True)
    v = sp.symbols(f"v1:{d + 1}", real=True)
    un = sp.sqrt(1 - th**2 * sum(x**2 for x in v))
    u = [th * x for x in v] + [un]
    F = _sym_tensor(n, nu)
    expr = 0
    for idx, sym in F.items():
        mult = math.factorial(nu)
        for k in set(idx):
            mult //= math.factorial(idx.count(k))
        term = sym * mult
        for i in idx:
            term *= u[i]
        expr += term
    ser = sp.series(expr, th, 0, 3).removeO()
    c0 = sp.expand(ser.coeff(th, 0))
    c1 = sp.expand(ser.coeff(th, 1))
    c2 = sp.expand(ser.coeff(th, 2))
    # model: u^T H u = c u_n^2 + 2 theta u_n b.v + theta^2 v^T T v
    #      = c + 2 theta b.v + theta^2 (v^T T v - c |v|^2) + O(theta^3)
    c = c0.subs({x: 0 for x in v})
    b = [sp.expand(c1).coeff(x, 1).subs({y: 0 for y in v}) / 2 for x in v]
    q2 = sp.expand(c2 + c * sum(x**2 for x in v))  # = v^T T v
    T = sp.zeros(d, d)
    for a in range(d):
        T[a, a] = sp.Poly(q2, *v).coeff_monomial(v[a] ** 2)
        for bb in range(a + 1, d):
            T[a, bb] = T[bb, a] = sp.Poly(q2, *v).coeff_monomial(v[a] * v[bb]) / 2
    # Lemma 6.2 prediction (plain b)
    last = n - 1
    key = lambda *ix: F[tuple(sorted(ix))]
    fN = key(*([last] * nu))
    pred_c = fN
    pred_b = [sp.Rational(nu, 2) * key(a, *([last] * (nu - 1))) for a in range(d)]
    Gm = sp.Matrix(d, d, lambda a, bb: key(a, bb, *([last] * (nu - 2))))
    pred_T = sp.binomial(nu, 2) * Gm - (sp.Rational(nu, 2) - 1) * fN * sp.eye(d)
    return {
        "derived": {"c": c, "b": b, "T": T},
        "lemma": {"c": pred_c, "b": pred_b, "T": pred_T},
        "agree": bool(sp.simplify(c - pred_c) == 0
                      and all(sp.simplify(x - y) == 0 for x, y in zip(b, pred_b))
                      and all(sp.simplify(e) == 0 for e in (T - pred_T))),
        "mixed_tilde_factor": sp.sqrt(2) * sp.Rational(nu, 2),
    }
