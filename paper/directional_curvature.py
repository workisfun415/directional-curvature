#!/usr/bin/env python3
"""
directional_curvature.py
========================
Single-file reproduction package for

    "Geometry, Conditioning, and Parity in One-Sided Directional Curvature
     Recovery, with a Controlled Comparison of Estimators"

Everything in the manuscript is produced by this file: every theorem
verification, every numerical observation, both benchmark tables, and all six
figures. No other module is required beyond numpy and matplotlib.

MAP FROM CODE TO MANUSCRIPT
---------------------------
  verify_hessians()        test-function correctness check (Section: Conventions)
  obs_identifiability()    Observation: identifiability, verifying Theorem 1
  obs_direction_sets()     Observation: rank deficiency of small Fibonacci sets
  thm_cone_singular()      Theorem: cone conditioning  (+ Fig. 1)
  obs_constants()          Observation: constants and their uncertainty (+ Fig. 2)
  obs_parity_scaling()     Observation: parity scaling (+ Fig. 3 left)
  obs_order()              Observation: attainable order (+ Fig. 3 right)
  obs_parity_separation()  Observation: parity separation on caps
  obs_plateau()            Observation: regularisation plateau (+ Fig. 4 left)
  obs_tracking()           Observation: the regularised estimate tracks C3
  obs_pilot_size()         Observation: pilot size
  obs_pilot_usability()    Fig. 4 right
  obs_R_sweep()            Observation: node placement
  obs_dopt()               Observation: D-optimal direction selection
  benchmark_main()         Table 1: controlled comparison at matched budget
  benchmark_msweep()       Table 2: redundant directions at matched budget
  benchmark_sensitivity()  base-point and cone-axis sensitivity

USAGE
-----
  python directional_curvature.py --all          everything (slowest)
  python directional_curvature.py --theorems     theorem verification only
  python directional_curvature.py --benchmark    comparison tables only
  python directional_curvature.py --figures      figures only
  python directional_curvature.py --quick        reduced replicate counts

CONVENTIONS
-----------
Sym_3(R) uses the Frobenius-consistent coordinates
    (H11, H22, H33, sqrt2*H12, sqrt2*H13, sqrt2*H23),
so that the coordinate 2-norm equals ||H||_F.  Caps are sampled on a tensor grid
in (phi, psi); "area-uniform" means cos(phi) equispaced, "polar-uniform" means
phi equispaced.  The default probe uses R = 1/2.  The evaluation point is
x0 = (0.10, -0.05, 0.20) throughout.  Noise is independent N(0, sigma^2) per
evaluation with sigma quoted relative to |f(x0)| + 1.  All seeds are literals.

Author: Ramakrishna Pasupuleti.  ORCID 0009-0008-8418-1430.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field

import numpy as np

X0 = np.array([0.10, -0.05, 0.20])
R_DEFAULT = 0.5
NU = np.sqrt((1 - R_DEFAULT) ** 2 + R_DEFAULT ** 2 + 1.0)
P_ODD_DIM = 10          # dim H_1 + dim H_3 = 3 + 7
QUICK = False


def nrep(n: int) -> int:
    """Replicate count, halved in --quick mode."""
    return max(6, n // 3) if QUICK else n


# =============================================================================
# PART 0.  Test functions with exact Hessians
# =============================================================================

class TestFunction:
    name = "base"

    def f(self, X):
        raise NotImplementedError

    def hess(self, x):
        raise NotImplementedError

    def third_scale(self, x, h=1e-2):
        """Crude axis-wise scale of the third derivative, used as the 'true'
        C3 reference.  Note this is NOT the L2(S^2) norm of the cubic form, so
        estimates differ from it by a fixed normalisation; see the manuscript."""
        d3 = 0.0
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            v = (self.f((x + 2 * e)[None])[0] - 2 * self.f((x + e)[None])[0]
                 + 2 * self.f((x - e)[None])[0] - self.f((x - 2 * e)[None])[0]) / (2 * h ** 3)
            d3 = max(d3, abs(v))
        return max(d3, 1e-12)


class Gaussian(TestFunction):
    name = "gaussian"

    def __init__(self, c=(0.3, -0.2, 0.15), s=0.8):
        self.c = np.asarray(c, float)
        self.s = s

    def f(self, X):
        d = X - self.c
        return np.exp(-np.sum(d * d, axis=-1) / (2 * self.s ** 2))

    def hess(self, x):
        d = x - self.c
        s2 = self.s ** 2
        g = float(np.exp(-d @ d / (2 * s2)))
        return g * (np.outer(d, d) / s2 ** 2 - np.eye(3) / s2)


class ExpSum(TestFunction):
    name = "expsum"

    def f(self, X):
        return np.exp(np.sum(X, axis=-1))

    def hess(self, x):
        return float(np.exp(np.sum(x))) * np.ones((3, 3))


class Poly(TestFunction):
    """f = x^4 + 2 x^2 y^2 + y^3 z + z^4 + x y z"""
    name = "poly"

    def f(self, X):
        x, y, z = X[..., 0], X[..., 1], X[..., 2]
        return x ** 4 + 2 * x ** 2 * y ** 2 + y ** 3 * z + z ** 4 + x * y * z

    def hess(self, p):
        x, y, z = p
        H = np.zeros((3, 3))
        H[0, 0] = 12 * x ** 2 + 4 * y ** 2
        H[1, 1] = 4 * x ** 2 + 6 * y * z
        H[2, 2] = 12 * z ** 2
        H[0, 1] = H[1, 0] = 8 * x * y + z
        H[0, 2] = H[2, 0] = y
        H[1, 2] = H[2, 1] = 3 * y ** 2 + x
        return H


class Trig(TestFunction):
    """f = sin(x + 2y) + cos(2y - z) + sin(x z)"""
    name = "trig"

    def f(self, X):
        x, y, z = X[..., 0], X[..., 1], X[..., 2]
        return np.sin(x + 2 * y) + np.cos(2 * y - z) + np.sin(x * z)

    def hess(self, p):
        x, y, z = p
        s1 = np.sin(x + 2 * y)
        s2, c2 = np.sin(2 * y - z), np.cos(2 * y - z)
        s3, c3 = np.sin(x * z), np.cos(x * z)
        H = np.zeros((3, 3))
        H[0, 0] = -s1 - z ** 2 * s3
        H[1, 1] = -4 * s1 - 4 * c2
        H[2, 2] = -c2 - x ** 2 * s3
        H[0, 1] = H[1, 0] = -2 * s1
        H[0, 2] = H[2, 0] = c3 - x * z * s3
        H[1, 2] = H[2, 1] = 2 * c2
        return H


FUNCTIONS = [Gaussian(), ExpSum(), Poly(), Trig()]


def verify_hessians(tol=1e-4, seed=1):
    """Analytic Hessians against fourth-order central differences."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for fn in FUNCTIONS:
        for _ in range(5):
            x = rng.uniform(-0.4, 0.4, 3)
            h = 1e-3
            Hn = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    ei = np.zeros(3); ei[i] = h
                    ej = np.zeros(3); ej[j] = h
                    Hn[i, j] = (fn.f((x + ei + ej)[None])[0]
                                - fn.f((x + ei - ej)[None])[0]
                                - fn.f((x - ei + ej)[None])[0]
                                + fn.f((x - ei - ej)[None])[0]) / (4 * h * h)
            err = float(np.max(np.abs(Hn - fn.hess(x))))
            worst = max(worst, err)
            assert err < tol, f"{fn.name}: Hessian mismatch {err:.2e}"
    return worst


# =============================================================================
# PART 1.  Geometry, direction sets, design matrix
# =============================================================================

@dataclass
class Geometry:
    """Feasible set as seen from the evaluation point (local coordinates)."""
    kind: str
    theta: float = np.pi / 2
    delta: float = 0.40
    dist: float = 0.05
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def feasible_dir(self, u):
        if self.kind == "cone":
            return float(u @ self.axis) >= np.cos(self.theta)
        return True

    def hmax(self, u):
        """Largest usable span along the unit direction u."""
        if self.kind == "cone":
            return self.delta if self.feasible_dir(u) else 0.0
        if self.kind == "slab":
            uz = float(u @ self.axis)
            return self.delta if uz >= 0 else min(self.delta, self.dist / abs(uz))
        if self.kind == "wedge":
            lim = self.delta
            for n in (np.array([0., 0., 1.]), np.array([0., 1., 0.])):
                un = float(u @ n)
                if un < 0:
                    lim = min(lim, self.dist / abs(un))
            return lim
        raise ValueError(self.kind)

    def contains(self, y):
        r = float(np.linalg.norm(y))
        if r < 1e-14:
            return True
        if r > self.delta + 1e-12:
            return False
        return r <= self.hmax(y / r) + 1e-12

    def sample(self, n, rng):
        out = []
        while len(out) < n:
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            hm = self.hmax(v)
            if hm <= 0:
                continue
            out.append(hm * rng.uniform(0.25, 1.0) * v)
        return np.array(out) if out else np.zeros((0, 3))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def dirs_theorem12():
    return np.array([_unit(v) for v in
                     [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [1, 1, 0], [1, 0, 1], [0, 1, 1]]])


def dirs_icosahedral():
    p = (1 + np.sqrt(5)) / 2
    return np.array([_unit(v) for v in
                     [[0, 1, p], [0, 1, -p], [1, p, 0],
                      [1, -p, 0], [p, 0, 1], [-p, 0, 1]]])


def dirs_fibonacci(m):
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def cap_dirs(theta, npol=40, naz=64, mode="area"):
    """Tensor grid on a polar cap. mode='area' => cos(phi) equispaced."""
    phis = (np.arccos(np.linspace(np.cos(theta), 1.0, npol)) if mode == "area"
            else np.linspace(1e-9, theta, npol))
    D = []
    for a in phis:
        for b in np.linspace(0, 2 * np.pi, naz, endpoint=False):
            D.append([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)])
    return np.array(D)


_CONE_DIR_CACHE = {}


def cone_random_dirs(geom, m, seed=2):
    key = (geom.kind, round(geom.theta, 6), geom.dist, m, seed)
    if key not in _CONE_DIR_CACHE:
        rng = np.random.default_rng(seed)
        D = []
        while len(D) < m:
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            if geom.hmax(v) > 1e-9:
                D.append(v)
        _CONE_DIR_CACHE[key] = np.array(D)
    return _CONE_DIR_CACHE[key]


def safe_rotate(dirs, geom):
    """Compress polar angles into the cone. Identity for theta >= pi/2, where
    the compression would instead EXPAND and distort the set."""
    if geom.kind != "cone" or geom.theta >= np.pi / 2 - 1e-9:
        return dirs
    ax = geom.axis
    out = []
    for u in dirs:
        c = float(u @ ax)
        ang = np.arccos(np.clip(abs(c), -1, 1))
        perp = u - c * ax
        npx = np.linalg.norm(perp)
        newang = geom.theta * (ang / (np.pi / 2))
        out.append(ax.copy() if npx < 1e-12
                   else _unit(np.cos(newang) * ax + np.sin(newang) * perp / npx))
    return np.array(out)


def design_matrix(dirs):
    """Row for u maps (H11,H22,H33,sqrt2 H12,sqrt2 H13,sqrt2 H23) -> u'Hu."""
    s = np.sqrt(2)
    U = np.asarray(dirs, float)
    return np.c_[U[:, 0] ** 2, U[:, 1] ** 2, U[:, 2] ** 2,
                 s * U[:, 0] * U[:, 1], s * U[:, 0] * U[:, 2], s * U[:, 1] * U[:, 2]]


def vec_to_H(h):
    s = np.sqrt(2)
    return np.array([[h[0], h[3] / s, h[4] / s],
                     [h[3] / s, h[1], h[5] / s],
                     [h[4] / s, h[5] / s, h[2]]])


_DOPT_CACHE = {}


def d_optimal_dirs(geom, m, n_cand=400, seed=4):
    """Sequential D-optimal selection inside the feasible region (Fedorov-type
    greedy exchange).  Adopted methodology, not a contribution."""
    key = (geom.kind, round(geom.theta, 6), geom.dist, m, seed,
           tuple(np.round(geom.axis, 6)))
    if key in _DOPT_CACHE:
        return _DOPT_CACHE[key]
    rng = np.random.default_rng(seed)
    C = []
    while len(C) < n_cand:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if geom.hmax(v) > 1e-9:
            C.append(v)
    C = np.array(C)
    A = design_matrix(C)
    chosen, M = [], np.zeros((6, 6))
    for _ in range(m):
        best, bi = -np.inf, None
        for i in range(len(C)):
            sign, ld = np.linalg.slogdet(M + np.outer(A[i], A[i]) + 1e-12 * np.eye(6))
            val = ld if sign > 0 else -np.inf
            if val > best:
                best, bi = val, i
        chosen.append(C[bi])
        M = M + np.outer(A[bi], A[bi])
    _DOPT_CACHE[key] = np.array(chosen)
    return _DOPT_CACHE[key]


# =============================================================================
# PART 2.  Evaluator, probe, spans
# =============================================================================

class Evaluator:
    """Noisy function oracle with caching, so that shared points are shared."""

    def __init__(self, fn, x0, sigma, rng):
        self.fn, self.x0, self.sigma, self.rng = fn, x0, sigma, rng
        self.cache, self.count = {}, 0

    def __call__(self, offset):
        key = tuple(np.round(offset, 12))
        if key in self.cache:
            return self.cache[key]
        val = float(self.fn.f((self.x0 + np.asarray(offset))[None])[0])
        if self.sigma > 0:
            val += self.rng.normal(0.0, self.sigma)
        self.cache[key] = val
        self.count += 1
        return val


def probe(ev, u, h, R=R_DEFAULT):
    """One-sided directional probe; manuscript equation (1).
    q = 2[(1-R)f(x) + R f(x+hu) - f(x+Rhu)] / (R(1-R)h^2) -> u'Hu."""
    f0 = ev(np.zeros(3))
    f1 = ev(h * u)
    fR = ev(R * h * u)
    return 2.0 * ((1 - R) * f0 + R * f1 - fR) / (R * (1 - R) * h * h)


def probe_error_model(h, sigma, c3, R=R_DEFAULT):
    """(bias, noise sd) for one probe; bias = (1+R)/3 * h * |D^3 f|."""
    bias = (1 + R) / 3.0 * h * c3
    amp = 2.0 * np.sqrt((1 - R) ** 2 + R ** 2 + 1.0) / (R * (1 - R) * h * h)
    return bias, amp * sigma


def optimal_span(hmax, sigma, c3, R=R_DEFAULT, fscale=1.0):
    """h* = min(hmax, (2 C_sigma / C_3)^{1/3}), with a round-off noise floor so
    that h* -> 0 is not selected in the noiseless case."""
    sigma = max(sigma, 1e-16 * fscale)
    C3 = (1 + R) / 3.0 * c3
    Cs = 2.0 * np.sqrt((1 - R) ** 2 + R ** 2 + 1.0) / (R * (1 - R)) * sigma
    return float(min(hmax, (2.0 * Cs / max(C3, 1e-14)) ** (1 / 3)))


def probe_sphere(ev, dirs, geom, h_of_u, R=R_DEFAULT):
    U, q = [], []
    for u in dirs:
        hm = geom.hmax(u)
        if hm <= 1e-9:
            continue
        h = min(h_of_u(u), hm)
        if h <= 1e-9:
            continue
        U.append(u)
        q.append(probe(ev, u, h, R))
    return np.array(U), np.array(q)


# =============================================================================
# PART 3.  Estimators (all take the same signature)
# =============================================================================

def quad_basis(Y):
    x, y, z = Y[:, 0], Y[:, 1], Y[:, 2]
    o = np.ones_like(x)
    return np.c_[o, x, y, z, 0.5 * x * x, 0.5 * y * y, 0.5 * z * z,
                 x * y, x * z, y * z]


def quad_solve(Y, vals):
    B = quad_basis(Y)
    c, *_ = np.linalg.lstsq(B, vals, rcond=None)
    return np.array([[c[4], c[7], c[8]],
                     [c[7], c[5], c[9]],
                     [c[8], c[9], c[6]]])


_POISED_CACHE = {}


def poised_set(geom, rng, n_pts=10, n_cand=400):
    """Pivotal selection of a Lambda-poised set inside the feasible region
    (Conn, Scheinberg & Vicente).  Adopted as the fairness standard."""
    key = (geom.kind, round(geom.theta, 6), geom.delta, geom.dist, n_pts,
           tuple(np.round(geom.axis, 6)))
    if key in _POISED_CACHE:
        return _POISED_CACHE[key]
    C = np.vstack([np.zeros(3), geom.sample(n_cand, rng)])
    B = quad_basis(C).copy()
    B = B / np.maximum(np.abs(B).max(axis=0), 1e-12)
    chosen, avail = [], list(range(len(C)))
    for j in range(n_pts):
        piv = max(avail, key=lambda i: abs(B[i, j]))
        if abs(B[piv, j]) < 1e-12:
            break
        chosen.append(piv)
        avail.remove(piv)
        B[piv] = B[piv] / B[piv, j]
        for i in avail:
            B[i] = B[i] - B[i, j] * B[piv]
    Y = C[chosen]
    Lam = (float(np.abs(np.linalg.pinv(quad_basis(Y)) @ quad_basis(C[:200]).T).max())
           if len(Y) >= 10 else np.inf)
    _POISED_CACHE[key] = (Y, Lam)
    return _POISED_CACHE[key]


_REG_CACHE = {}


def m_quad_regression(ev, geom, rng, n, sc, **kw):
    Y, _ = poised_set(geom, rng)
    if len(Y) < 10:
        return None
    key = (geom.kind, round(geom.theta, 6), geom.delta, geom.dist, n)
    if key not in _REG_CACHE:
        extra = geom.sample(max(0, n - len(Y)), rng)
        _REG_CACHE[key] = np.vstack([Y, extra])[:n] if len(extra) else Y[:n]
    Y = _REG_CACHE[key] * sc
    return quad_solve(Y, np.array([ev(y) for y in Y]))


def m_quad_poised(ev, geom, rng, n, sc, **kw):
    Y, _ = poised_set(geom, rng)
    if len(Y) < 10:
        return None
    Y = Y * sc
    return quad_solve(Y, np.array([ev(y) for y in Y]))


def m_mls(ev, geom, rng, n, sc, **kw):
    """Moving least squares: quadratic fit with Gaussian distance weights."""
    Y, _ = poised_set(geom, rng)
    if len(Y) < 10:
        return None
    extra = geom.sample(max(0, n - len(Y)), rng)
    Y = (np.vstack([Y, extra])[:n] if len(extra) else Y[:n]) * sc
    vals = np.array([ev(y) for y in Y])
    rho = max(float(np.linalg.norm(Y, axis=1).max()), 1e-12)
    w = np.exp(-np.sum(Y * Y, axis=1) / (2 * (0.6 * rho) ** 2))
    B = quad_basis(Y) * w[:, None]
    c, *_ = np.linalg.lstsq(B, vals * w, rcond=None)
    return np.array([[c[4], c[7], c[8]], [c[7], c[5], c[9]], [c[8], c[9], c[6]]])


def m_central_fd(ev, geom, rng, n, sc, **kw):
    """19-point O(h^2) stencil; None when any required point is infeasible."""
    h = sc * geom.delta
    E = np.eye(3)
    pts = [np.zeros(3)]
    for i in range(3):
        pts += [h * E[i], -h * E[i]]
    for i, j in itertools.combinations(range(3), 2):
        for si, sj in itertools.product((1, -1), repeat=2):
            pts.append(si * h * E[i] + sj * h * E[j])
    if not all(geom.contains(pt) for pt in pts):
        return None
    f0 = ev(np.zeros(3))
    H = np.zeros((3, 3))
    for i in range(3):
        H[i, i] = (ev(h * E[i]) - 2 * f0 + ev(-h * E[i])) / h ** 2
    for i, j in itertools.combinations(range(3), 2):
        H[i, j] = H[j, i] = (ev(h * E[i] + h * E[j]) - ev(h * E[i] - h * E[j])
                             - ev(-h * E[i] + h * E[j])
                             + ev(-h * E[i] - h * E[j])) / (4 * h * h)
    return H


def _directional_core(ev, geom, dirs, spans, weights=None, R=R_DEFAULT):
    U, q, w = [], [], []
    for k, (u, h) in enumerate(zip(dirs, spans)):
        if h <= 1e-12:
            continue
        U.append(u)
        q.append(probe(ev, u, h, R))
        w.append(1.0 if weights is None else weights[k])
    if len(U) < 6:
        return None
    A = design_matrix(np.array(U))
    q = np.array(q)
    if weights is None:
        hv = np.linalg.lstsq(A, q, rcond=None)[0]
    else:
        W = np.diag(np.array(w))
        hv = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ q, rcond=None)[0]
    return vec_to_H(hv)


def m_onesided_fd(ev, geom, rng, n, sc, R=R_DEFAULT, **kw):
    """Classical one-sided second differences on the {e_i} u {(e_i+e_j)/sqrt2}
    directions; coincides with the directional probe up to the choice of R."""
    D = safe_rotate(dirs_theorem12(), geom)
    return _directional_core(ev, geom, D,
                             [min(sc * geom.delta, geom.hmax(u)) for u in D], R=R)


def m_dir_isotropic(ev, geom, rng, n, sc, dirs=None, R=R_DEFAULT, **kw):
    return _directional_core(ev, geom, dirs,
                             [min(sc * geom.delta, geom.hmax(u)) for u in dirs], R=R)


def m_dir_aniso(ev, geom, rng, n, sc, dirs=None, sigma=0.0, c3=1.0,
                fscale=1.0, R=R_DEFAULT, **kw):
    spans = [optimal_span(geom.hmax(u), sigma, c3, R, fscale) for u in dirs]
    w = []
    for h in spans:
        b, s = probe_error_model(h, sigma, c3, R) if h > 0 else (1.0, 1.0)
        w.append(1.0 / max(b * b + s * s, 1e-30))
    return _directional_core(ev, geom, dirs, spans, weights=w, R=R)


def m_dir_dopt(ev, geom, rng, n, sc, dirs=None, sigma=0.0, c3=1.0,
               fscale=1.0, R=R_DEFAULT, **kw):
    D = d_optimal_dirs(geom, len(dirs))
    spans = [optimal_span(geom.hmax(u), sigma, c3, R, fscale) for u in D]
    w = []
    for h in spans:
        b, s = probe_error_model(h, sigma, c3, R) if h > 0 else (1.0, 1.0)
        w.append(1.0 / max(b * b + s * s, 1e-30))
    return _directional_core(ev, geom, D, spans, weights=w, R=R)


METHODS = [("quad_reg", m_quad_regression, True),
           ("poised", m_quad_poised, True),
           ("MLS", m_mls, True),
           ("centralFD", m_central_fd, True),
           ("onesidedFD", m_onesided_fd, True),
           ("dir_iso", m_dir_isotropic, True),
           ("dir_aniso", m_dir_aniso, False),
           ("dir_Dopt", m_dir_dopt, False)]

SCALES = list(np.logspace(-3.5, 0, 15))


# =============================================================================
# PART 4.  Harmonic parity machinery
# =============================================================================

def real_sh_basis(U, lmax=4):
    """Monomials grouped by total degree.  Spans the same subspaces as the real
    spherical harmonics of matching parity for the degrees used here."""
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    mono = {0: [np.ones_like(x)],
            1: [x, y, z],
            2: [x * x, y * y, z * z, x * y, x * z, y * z],
            3: [x ** 3, y ** 3, z ** 3, x * x * y, x * x * z, y * y * x,
                y * y * z, z * z * x, z * z * y, x * y * z],
            4: [x ** 4, y ** 4, z ** 4, x ** 3 * y, x ** 3 * z, y ** 3 * x,
                y ** 3 * z, z ** 3 * x, z ** 3 * y, x * x * y * y,
                x * x * z * z, y * y * z * z, x * x * y * z, y * y * x * z,
                z * z * x * y]}
    cols, degs = [], []
    for d in range(lmax + 1):
        for m in mono[d]:
            cols.append(m)
            degs.append(d)
    return np.array(cols).T, np.array(degs)


def harmonic_energies(U, q, lmax=4, ridge=0.0):
    """Split q(u) into degree bands.  ridge > 0 gives regularised parity
    inversion, with the parameter measured relative to ||B^T B||_2."""
    B, degs = real_sh_basis(U, lmax)
    if ridge > 0:
        G = B.T @ B
        s1 = float(np.linalg.eigvalsh(G).max())
        c = np.linalg.solve(G + ridge * s1 * np.eye(B.shape[1]), B.T @ q)
    else:
        c, *_ = np.linalg.lstsq(B, q, rcond=None)
    E = {d: float(np.linalg.norm(B[:, degs == d] @ c[degs == d]) / np.sqrt(len(q)))
         for d in range(lmax + 1)}
    return E, c, B, degs


def c3_harmonic(fn, sigma, geom, h0=0.2, m=40, ridge=0.0, lmax=3, seed=3,
                subtract_noise=True):
    """Estimate ||D^3 f|| from the odd harmonic content of the probe field."""
    D = cone_random_dirs(geom, m, seed=7)
    ev = Evaluator(fn, X0, sigma, np.random.default_rng(seed))
    U, q = probe_sphere(ev, D, geom, lambda u: min(h0, geom.hmax(u)))
    if len(U) <= 16:
        return None
    E, *_ = harmonic_energies(U, q, lmax=lmax, ridge=ridge)
    odd = float(np.hypot(E[1], E[3]))
    if subtract_noise and sigma > 0:
        sq = 2 * sigma * NU / (R_DEFAULT * (1 - R_DEFAULT) * h0 ** 2)
        odd = np.sqrt(max(odd ** 2 - P_ODD_DIM * sq ** 2 / len(U), 0.0))
    return max(3.0 * odd / ((1 + R_DEFAULT) * h0), 1e-12)


def c3_direct(fn, sigma, geom, h0=0.3, m=24, seed=3):
    """One-sided third difference on 4 collinear nodes; no parity separation."""
    D = cone_random_dirs(geom, m, seed=seed)
    ev = Evaluator(fn, X0, sigma, np.random.default_rng(seed))
    vals = []
    for u in D:
        h = min(h0, geom.hmax(u))
        s = h / 3.0
        vals.append((ev(3 * s * u) - 3 * ev(2 * s * u)
                     + 3 * ev(s * u) - ev(np.zeros(3))) / s ** 3)
    return float(np.sqrt(np.mean(np.array(vals) ** 2)))


# =============================================================================
# PART 5.  Theorem and observation verification
# =============================================================================

def _null_space(A, tol=1e-10):
    u, s, vt = np.linalg.svd(A)
    r = int((s > tol * max(A.shape) * s.max()).sum()) if s.size else 0
    return vt[r:].T


def obs_identifiability(trials=200, seed=0):
    """Verifies Theorem 'Identifiability floor': H is determined iff n >= 10."""
    print("\n[Theorem: identifiability floor] unresolved Hessian dof vs n")
    print(f"{'n':>4}{'identifiable':>15}{'unresolved dof':>16}")
    rng = np.random.default_rng(seed)
    trials = nrep(trials)
    for n in range(6, 13):
        bad, worst = 0, 0
        for _ in range(trials):
            Y = rng.normal(size=(n, 3)) * 0.3
            Y[0] = 0.0
            ns = _null_space(quad_basis(Y))
            dof = 0 if ns.shape[1] == 0 else int(np.linalg.matrix_rank(ns[4:, :], tol=1e-9))
            worst = max(worst, dof)
            bad += dof > 0
        print(f"{n:>4}{('yes' if bad == 0 else f'no ({bad}/{trials})'):>15}{worst:>16}")
    print("  -> a directional scheme costs 1+2m >= 13 evaluations, above the floor of 10")


def obs_direction_sets():
    """Verifies the rank deficiency of small Fibonacci sets."""
    print("\n[Observation: direction sets at m=6] condition number of the design")
    for nm, D in (("icosahedral", dirs_icosahedral()),
                  ("{e_i} u {(e_i+e_j)/sqrt2}", dirs_theorem12()),
                  ("fibonacci-6", dirs_fibonacci(6))):
        print(f"  {nm:<26} kappa = {np.linalg.cond(design_matrix(D)):.4e}")
    for m in (12, 20, 50):
        print(f"  fibonacci-{m:<16} kappa = {np.linalg.cond(design_matrix(dirs_fibonacci(m))):.4e}")


def thm_cone_singular():
    """Verifies the cone-conditioning theorem and its exact constants."""
    print("\n[Theorem: cone conditioning] singular values, area-uniform measure")
    print("  predicted {1, th/sqrt2, th/sqrt2, ., ., th^2/sqrt24}")
    print(f"{'theta':>7}{'s1':>10}{'s2/th':>10}{'s6/th^2':>10}{'kappa*th^2':>12}")
    for deg in (60, 40, 30, 20, 10, 5, 2):
        th = np.deg2rad(deg)
        D = cap_dirs(th, 60, 96, "area")
        A = design_matrix(D) / np.sqrt(len(D))
        s = np.linalg.svd(A)[1]
        print(f"{deg:>7}{s[0]:>10.4f}{s[1]/th:>10.4f}{s[5]/th**2:>10.4f}"
              f"{(s[0]/s[5])*th**2:>12.4f}")
    print(f"  exact: s2/th -> {1/np.sqrt(2):.5f}, s6/th^2 -> {1/np.sqrt(24):.5f}, "
          f"kappa*th^2 -> {np.sqrt(24):.4f}")
    print("  polar-uniform measure, for comparison:")
    for deg in (10, 2):
        th = np.deg2rad(deg)
        D = cap_dirs(th, 60, 96, "polar")
        A = design_matrix(D) / np.sqrt(len(D))
        s = np.linalg.svd(A)[1]
        print(f"{deg:>7}{s[0]:>10.4f}{s[1]/th:>10.4f}{s[5]/th**2:>10.4f}"
              f"{(s[0]/s[5])*th**2:>12.4f}")
    print(f"  exact: s2/th -> {1/np.sqrt(3):.5f}, s6/th^2 -> {np.sqrt(2/45):.5f}, "
          f"kappa*th^2 -> {np.sqrt(45/2):.4f}")


def obs_constants():
    """Grid refinement of the constant; digits stable under refinement only."""
    print("\n[Observation: constants and their uncertainty] refinement at theta=2 deg")
    th = np.deg2rad(2.0)
    vals = []
    for npol, naz in ((20, 32), (40, 64), (80, 128)):
        D = cap_dirs(th, npol, naz, "area")
        A = design_matrix(D) / np.sqrt(len(D))
        s = np.linalg.svd(A)[1]
        vals.append(s[5] / th ** 2)
        print(f"  {npol:>3}x{naz:<4} s6/th^2 = {vals[-1]:.5f}   "
              f"kappa*th^2 = {(s[0]/s[5])*th**2:.4f}")
    d = np.diff(vals)
    extrap = vals[-1] + d[-1] * (d[-1] / (d[-2] - d[-1])) if len(d) > 1 else vals[-1]
    print(f"  Richardson limit ~ {extrap:.5f}   exact 1/sqrt(24) = {1/np.sqrt(24):.6f}")
    print("  -> constants from a single unrefined grid are unreliable at the 3rd digit")
    return vals


def obs_parity_scaling():
    """Odd-band energy is proportional to h (noiseless) and to h^-2 (noise)."""
    print("\n[Observation: parity scaling] odd-band energy / h  (should be constant)")
    g = Geometry("cone", theta=np.pi, delta=1.0)
    D = dirs_fibonacci(120)
    HS = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
    print(f"{'fn':<10}{'sigma':>8}" + "".join(f"{h:>10}" for h in HS)
          + f"{'slope*':>8}")
    for fn in (Poly(), Trig(), Gaussian()):
        for s_rel in (0.0, 1e-4):
            E = []
            for h in HS:
                ev = Evaluator(fn, X0, s_rel, np.random.default_rng(1))
                U, q = probe_sphere(ev, D, g, lambda u, h=h: h)
                e, *_ = harmonic_energies(U, q, lmax=4)
                E.append(np.hypot(e[1], e[3]))
            E = np.array(E)
            # noiseless: truncation dominates at every h, so fit all points.
            # noisy: truncation still dominates at large h, so the noise
            # exponent is only visible over the smallest spans.
            idx = slice(None) if s_rel == 0 else slice(-3, None)
            slope = np.polyfit(np.log(np.array(HS)[idx]), np.log(E[idx]), 1)[0]
            print(f"{fn.name:<10}{s_rel:>8.0e}"
                  + "".join(f"{v/h:>10.3f}" for v, h in zip(E, HS))
                  + f"{slope:>8.2f}")
    print("  -> slope +1 without noise (third-order term); with noise the")
    print("     small-h slope approaches -2, the crossover moving with sigma")


def obs_order():
    """Second order iff the direction set is antipodally symmetric."""
    print("\n[Proposition: antipodal availability] measured convergence order")
    HS = [0.2, 0.1, 0.05]
    sets = [("full sphere m=120", Geometry("cone", theta=np.pi, delta=1.0),
             lambda: dirs_fibonacci(120), "yes"),
            ("icosa +/- pairs m=12", Geometry("cone", theta=np.pi, delta=1.0),
             lambda: np.vstack([dirs_icosahedral(), -dirs_icosahedral()]), "yes"),
            ("hemisphere", Geometry("cone", theta=np.deg2rad(90), delta=1.0),
             lambda: dirs_fibonacci(120), "no"),
            ("cap 45 deg", Geometry("cone", theta=np.deg2rad(45), delta=1.0),
             lambda: dirs_fibonacci(400), "no"),
            ("cap 20 deg", Geometry("cone", theta=np.deg2rad(20), delta=1.0),
             lambda: dirs_fibonacci(1200), "no"),
            ("slab d=0.3", Geometry("slab", dist=0.3, delta=1.0),
             lambda: dirs_fibonacci(120), "no")]
    print(f"{'fn':<10}{'direction set':<22}{'antipodal':>11}{'order':>8}")
    for fn in (Poly(), Trig(), Gaussian()):
        Ht = fn.hess(X0)
        for name, g, mk, anti in sets:
            E = []
            for h in HS:
                ev = Evaluator(fn, X0, 0.0, np.random.default_rng(1))
                U, q = probe_sphere(ev, mk(), g, lambda u, h=h: h)
                if len(U) < 6:
                    E = None
                    break
                hv = np.linalg.lstsq(design_matrix(U), q, rcond=None)[0]
                E.append(np.linalg.norm(vec_to_H(hv) - Ht, "fro"))
            order = "n/a" if E is None else f"{np.log2(E[0]/E[2])/2:.2f}"
            print(f"{fn.name:<10}{name:<22}{anti:>11}{order:>8}")


def obs_parity_separation():
    """Even/odd spans become indistinguishable on a small cap."""
    print("\n[Observation: parity separation on caps] smallest principal angle")
    print(f"{'theta':>7}{'sin(angle)':>14}")
    for deg in (90, 70, 60, 50, 40, 30, 20):
        th = np.deg2rad(deg)
        D = cap_dirs(th, 25, 48, "polar")
        B, degs = real_sh_basis(D, lmax=4)
        QE, _ = np.linalg.qr(B[:, np.isin(degs, [0, 2, 4])])
        QO, _ = np.linalg.qr(B[:, np.isin(degs, [1, 3])])
        c = float(np.linalg.svd(QE.T @ QO)[1].max())
        print(f"{deg:>7}{np.sqrt(max(1-min(c,1.0)**2,0)):>14.3e}")
    print("  -> decays far faster than the theta^-2 of the conditioning theorem")


def obs_plateau():
    """C3 estimate is insensitive to the ridge parameter over a wide interval."""
    print("\n[Observation: regularisation plateau] C3_hat / C3_true vs lambda_rel")
    LAM = np.logspace(-8, 1, 10)
    print(f"{'theta':>6}{'fn':<7}" + "".join(f"{l:>9.0e}" for l in LAM) + f"{'width':>10}")
    LAMF = np.logspace(-8, 1, 19)
    for deg in (90, 50, 30, 20):
        g = Geometry("cone", theta=np.deg2rad(deg), delta=0.5)
        for fn in (Poly(), Trig()):
            c3t = fn.third_scale(X0)
            fs = abs(float(fn.f(X0[None])[0])) + 1.0
            rf = [(c3_harmonic(fn, 1e-4 * fs, g, ridge=l) or np.nan) / c3t for l in LAMF]
            good = [l for l, v in zip(LAMF, rf) if 0.5 <= v <= 2.0]
            width = (max(good) / min(good)) if good else 0.0
            rc = [(c3_harmonic(fn, 1e-4 * fs, g, ridge=l) or np.nan) / c3t for l in LAM]
            print(f"{deg:>6}{fn.name:<7}" + "".join(f"{v:>9.2f}" for v in rc)
                  + f"{width:>10.0e}")
    print("  -> any lambda_rel in [1e-5, 1e-2] is acceptable; no tuning needed")


def obs_tracking():
    """The regularised estimate tracks C3 rather than merely shrinking."""
    print("\n[Observation: tracking] does C3_hat scale with the true third derivative?")

    class Cubic(TestFunction):
        def __init__(self, A):
            self.A = A
            self.name = f"cubic A={A}"

        def f(self, X):
            return (np.exp(-np.sum(X ** 2, axis=-1) / 1.2)
                    + self.A * (X[..., 0] ** 3 + 0.7 * X[..., 1] ** 3
                                + 0.4 * X[..., 2] ** 3))

    AS = [0.5, 1.0, 3.0, 10.0]
    print(f"{'theta':>7}" + "".join(f"{'A='+str(a):>10}" for a in AS) + f"{'slope':>8}")
    for deg in (90, 50, 30, 20):
        g = Geometry("cone", theta=np.deg2rad(deg), delta=0.5)
        vals = [c3_harmonic(Cubic(a), 2e-4, g, ridge=1e-3) for a in AS]
        slope = np.polyfit(np.log(AS), np.log(vals), 1)[0]
        print(f"{deg:>7}" + "".join(f"{v:>10.2f}" for v in vals) + f"{slope:>8.2f}")
    print("  -> slope 1 is required; shrinkage alone would give 0")


def obs_pilot_size():
    """How few directions suffice, and what happens when the basis is too rich."""
    print("\n[Observation: pilot size] C3_hat / C3_true, degree-3 basis (16 columns)")
    MS = (18, 24, 30, 40, 60)
    g = Geometry("cone", theta=np.pi, delta=0.5)
    print(f"{'fn':<10}{'sigma':>8}" + "".join(f"{'m='+str(m):>10}" for m in MS))
    for fn in (Poly(), Trig(), Gaussian()):
        c3t = fn.third_scale(X0)
        fs = abs(float(fn.f(X0[None])[0])) + 1.0
        for s_rel in (1e-4, 1e-2):
            row = [c3_harmonic(fn, s_rel * fs, g, m=m) for m in MS]
            print(f"{fn.name:<10}{s_rel:>8.0e}"
                  + "".join(f"{(v or np.nan)/c3t:>10.2f}" for v in row))
    bad = c3_harmonic(Poly(), 1e-4, g, m=24, lmax=4)
    print(f"  rank-deficient control (m=24, degree-4 basis of 35 columns): "
          f"C3_hat/C3_true = {(bad or np.nan)/Poly().third_scale(X0):.1f}")


def obs_R_sweep():
    """Node placement: R=1/2 at high noise, R~0.3 at low noise."""
    print("\n[Observation: node placement] directional error vs R")
    RS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    print(f"{'fn':<7}{'geom':<10}{'sigma':>8}" + "".join(f"{'R='+str(r):>10}" for r in RS))
    for fn in (Poly(), Trig()):
        Ht = fn.hess(X0)
        fs = abs(float(fn.f(X0[None])[0])) + 1.0
        c3 = fn.third_scale(X0)
        for gname, g in (("cap45", Geometry("cone", theta=np.deg2rad(45))),
                         ("cap20", Geometry("cone", theta=np.deg2rad(20)))):
            D = safe_rotate(dirs_icosahedral(), g)
            for s_rel in (1e-4, 1e-2):
                sg = s_rel * fs
                row = []
                for R in RS:
                    errs = []
                    for r in range(nrep(20)):
                        ev = Evaluator(fn, X0, sg, np.random.default_rng(17 * r + 3))
                        H = m_dir_aniso(ev, g, None, 13, 1.0, dirs=D, sigma=sg,
                                        c3=c3, fscale=fs, R=R)
                        if H is not None:
                            errs.append(np.linalg.norm(H - Ht, "fro"))
                    row.append(np.mean(errs) if errs else np.nan)
                print(f"{fn.name:<7}{gname:<10}{s_rel:>8.0e}"
                      + "".join(f"{v:>10.3g}" for v in row))


def obs_dopt():
    """D-optimal selection inside the cone versus a compressed fixed set."""
    print("\n[Observation: D-optimal directions] error and conditioning")
    fn = Poly()
    Ht = fn.hess(X0)
    fs = abs(float(fn.f(X0[None])[0])) + 1.0
    c3 = fn.third_scale(X0)
    print(f"{'geom':<8}{'sigma':>8}{'m':>4}{'fixed':>12}{'D-opt':>12}"
          f"{'k fixed':>12}{'k D-opt':>10}")
    for gname, g in (("cap30", Geometry("cone", theta=np.deg2rad(30))),
                     ("cap10", Geometry("cone", theta=np.deg2rad(10)))):
        for s_rel in (1e-4, 1e-2):
            sg = s_rel * fs
            for m in (6, 12):
                fixed = safe_rotate(dirs_icosahedral() if m == 6
                                    else dirs_fibonacci(m), g)
                dopt = d_optimal_dirs(g, m)
                res = {}
                for lab, D in (("fixed", fixed), ("dopt", dopt)):
                    errs = []
                    for r in range(nrep(30)):
                        ev = Evaluator(fn, X0, sg, np.random.default_rng(29 * r + 5))
                        spans = [optimal_span(g.hmax(u), sg, c3, R_DEFAULT, fs) for u in D]
                        H = _directional_core(ev, g, D, spans)
                        if H is not None:
                            errs.append(np.linalg.norm(H - Ht, "fro"))
                    res[lab] = (np.mean(errs) if errs else np.nan,
                                np.linalg.cond(design_matrix(D)))
                print(f"{gname:<8}{s_rel:>8.0e}{m:>4}{res['fixed'][0]:>12.3e}"
                      f"{res['dopt'][0]:>12.3e}{res['fixed'][1]:>12.2f}"
                      f"{res['dopt'][1]:>10.2f}")


# =============================================================================
# PART 6.  Controlled comparison
# =============================================================================

def bench_cell(fn, geom, s_rel, m=6, n_rep=25, seed=0, x0=None, R=R_DEFAULT):
    """One benchmark cell.  Every method receives its own oracle-optimal radius;
    the directional anisotropic variants select their own spans."""
    rng = np.random.default_rng(seed)
    x0 = X0 if x0 is None else x0
    Ht = fn.hess(x0)
    fs = abs(float(fn.f(x0[None])[0])) + 1.0
    sigma = s_rel * fs
    c3 = fn.third_scale(x0)
    dirs = safe_rotate(dirs_icosahedral() if m == 6 else dirs_fibonacci(m), geom)
    n = 1 + 2 * m
    kw = dict(dirs=dirs, sigma=sigma, c3=c3, fscale=fs, R=R)

    def run(fnc, scales):
        best = None
        for sc in scales:
            errs, nev = [], []
            for r in range(n_rep):
                ev = Evaluator(fn, x0, sigma, np.random.default_rng(seed * 7919 + r))
                H = fnc(ev, geom, rng, n, sc, **kw)
                if H is None:
                    continue
                errs.append(np.linalg.norm(H - Ht, "fro"))
                nev.append(ev.count)
            if not errs:
                continue
            a = np.array(errs)
            st = dict(mean=float(a.mean()),
                      se=float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
                      evals=float(np.mean(nev)), scale=sc, feasible=True)
            if best is None or st["mean"] < best["mean"]:
                best = st
        return best or dict(mean=np.nan, se=np.nan, evals=np.nan,
                            scale=np.nan, feasible=False)

    return {name: run(f, SCALES if sweep else [1.0]) for name, f, sweep in METHODS}


def benchmark_main(n_rep=25):
    """Table 1 of the manuscript."""
    print("\n" + "=" * 96)
    print("TABLE 1  controlled comparison, matched budget 13 evaluations,")
    print("         every method at its own oracle-optimal radius; '*' = infeasible")
    print("=" * 96)
    geoms = [("free", Geometry("cone", theta=np.pi)),
             ("cap90", Geometry("cone", theta=np.deg2rad(90))),
             ("cap45", Geometry("cone", theta=np.deg2rad(45))),
             ("cap20", Geometry("cone", theta=np.deg2rad(20))),
             ("slab", Geometry("slab", dist=0.05)),
             ("wedge", Geometry("wedge", dist=0.05))]
    names = [m[0] for m in METHODS]
    for fn in FUNCTIONS:
        print(f"\n--- {fn.name} ---")
        print(f"{'geometry':<9}{'sigma':>8}" + "".join(f"{n:>12}" for n in names))
        for gname, g in geoms:
            for s in (1e-4, 1e-2):
                c = bench_cell(fn, g, s, n_rep=nrep(n_rep),
                               seed=abs(hash((fn.name, gname, s))) % 10 ** 5)
                row = f"{gname:<9}{s:>8.0e}"
                for nm in names:
                    v = c[nm]["mean"]
                    row += f"{'*':>12}" if not np.isfinite(v) else f"{v:>12.3e}"
                print(row)


def benchmark_msweep(n_rep=20):
    """Table 2: redundancy does not close the gap."""
    print("\n" + "=" * 80)
    print("TABLE 2  redundant directions at matched budget")
    print("=" * 80)
    print(f"{'fn':<8}{'geom':<8}{'sigma':>8}{'m':>4}{'evals':>7}"
          f"{'quad_reg':>12}{'dir_aniso':>12}")
    for fn in (Poly(), Trig()):
        for gname, g in (("cap30", Geometry("cone", theta=np.deg2rad(30))),
                         ("cap10", Geometry("cone", theta=np.deg2rad(10))),
                         ("slab", Geometry("slab", dist=0.05))):
            for s in (1e-4, 1e-2):
                for m in (6, 12, 20, 50):
                    c = bench_cell(fn, g, s, m=m, n_rep=nrep(n_rep), seed=11)
                    print(f"{fn.name:<8}{gname:<8}{s:>8.0e}{m:>4}{1+2*m:>7}"
                          f"{c['quad_reg']['mean']:>12.3e}"
                          f"{c['dir_aniso']['mean']:>12.3e}")


def benchmark_sensitivity(n_trials=8, n_rep=20):
    """Does the verdict move with the base point and the cone axis?"""
    print("\n" + "=" * 80)
    print("SENSITIVITY  random base points and cone axes, cap 35 deg, sigma=1e-3")
    print("=" * 80)
    rng = np.random.default_rng(99)
    fn = Poly()
    wins = {"quad": 0, "dir": 0}
    print(f"{'trial':>6}{'axis':>24}{'quad_reg':>12}{'MLS':>12}{'dir_aniso':>12}{'winner':>9}")
    for t in range(n_trials):
        x0 = rng.uniform(-0.35, 0.35, 3)
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        g = Geometry("cone", theta=np.deg2rad(35), axis=ax)
        c = bench_cell(fn, g, 1e-3, n_rep=nrep(n_rep), seed=1000 + t, x0=x0)
        bq = min(c["quad_reg"]["mean"], c["MLS"]["mean"], c["poised"]["mean"])
        bd = c["dir_aniso"]["mean"]
        w = "quad" if bq < bd else "dir"
        wins[w] += 1
        print(f"{t:>6}  ({ax[0]:>5.2f},{ax[1]:>5.2f},{ax[2]:>5.2f})"
              f"{c['quad_reg']['mean']:>12.3e}{c['MLS']['mean']:>12.3e}"
              f"{bd:>12.3e}{w:>9}")
    print(f"  -> quadratic family {wins['quad']}/{n_trials}, "
          f"directional {wins['dir']}/{n_trials}")


# =============================================================================
# PART 7.  Figures
# =============================================================================


def fig_roadmap(outdir="."):
    """Conceptual roadmap: three independent mechanisms and where each result
    sits.  Line art, sans-serif, black and white (Springer artwork guidance)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")

    def box(x, y, w, h, txt, lw=1.0, fs=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc="white", ec="black", lw=lw))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, family="sans-serif")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, lw=0.9, color="black"))

    box(3.3, 4.35, 3.4, 0.62,
        "identifiability floor\n$n\\geq 10$ evaluations (Thm. 2.1)", lw=1.4)

    box(0.15, 2.55, 2.9, 0.95,
        "admissible directions\nconfined to a cap $C_\\theta$")
    box(3.55, 2.55, 2.9, 0.95,
        "one-sided access\nno antipodal partner")
    box(6.95, 2.55, 2.9, 0.95,
        "measurement noise\nlevel $\\sigma$")

    box(0.15, 1.05, 2.9, 0.95,
        "conditioning\n$\\kappa(\\theta)\\asymp\\theta^{-2}$ (Thm. 3.2)")
    box(3.55, 1.05, 2.9, 0.95,
        "first-order ceiling (Prop. 4.4)\nparity split (Prop. 4.2)")
    box(6.95, 1.05, 2.9, 0.95,
        "variance $\\sigma h^{-2}$ (Prop. 5.1)\nspan $h^{*}\\sim\\sigma^{1/3}$")

    box(3.05, 0.05, 3.9, 0.62,
        "recovery error: three independent limits (Cor. 5.4)", lw=1.4)

    for x, xt in ((1.6, 4.15), (5.0, 5.0), (8.4, 5.85)):
        arrow(x, 2.55, x, 2.04)
        arrow(x, 1.03, xt, 0.72)
    arrow(5.0, 4.33, 5.0, 3.54)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig0_roadmap.pdf")
    fig.savefig(f"{outdir}/fig0_roadmap.png", dpi=300)
    plt.close(fig)


def make_figures(outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 160,
                         "axes.grid": True, "grid.alpha": 0.3})

    def save(fig, stem):
        fig.tight_layout()
        fig.savefig(f"{outdir}/{stem}.pdf")
        fig.savefig(f"{outdir}/{stem}.png", dpi=200)
        plt.close(fig)

    fig_roadmap(outdir)

    # Fig 1 -- cone conditioning
    degs = np.array([60, 40, 30, 20, 10, 5, 2, 1])
    S = []
    for d in degs:
        D = cap_dirs(np.deg2rad(d), 60, 96, "area")
        S.append(np.linalg.svd(design_matrix(D) / np.sqrt(len(D)))[1])
    S = np.array(S)
    th = np.deg2rad(degs)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for i, lab in ((0, "$s_1$"), (1, "$s_2,s_3$"), (3, "$s_4,s_5,s_6$")):
        ax[0].loglog(th, S[:, i], "o-", ms=3.5, label=lab)
    ax[0].loglog(th, th / np.sqrt(2), "k--", lw=0.8, label=r"$\theta/\sqrt{2}$")
    ax[0].loglog(th, th ** 2 / np.sqrt(24), "k:", lw=0.8, label=r"$\theta^2/\sqrt{24}$")
    ax[0].set_xlabel(r"$\theta$ (rad)"); ax[0].set_ylabel("singular value")
    ax[0].legend(fontsize=7); ax[0].set_title("(a) singular values")
    ax[1].semilogx(th, (S[:, 0] / S[:, 5]) * th ** 2, "o-", ms=3.5)
    ax[1].axhline(np.sqrt(24), color="k", ls="--", lw=0.8, label=r"$\sqrt{24}=4.899$")
    ax[1].set_xlabel(r"$\theta$ (rad)"); ax[1].set_ylabel(r"$\kappa(\theta)\,\theta^2$")
    ax[1].legend(fontsize=7); ax[1].set_title("(b) constant")
    save(fig, "fig1_cone")

    # Fig 2 -- refinement
    th2 = np.deg2rad(2.0)
    grids = [(20, 32), (40, 64), (80, 128)]
    vals = []
    for npol, naz in grids:
        D = cap_dirs(th2, npol, naz, "area")
        vals.append(np.linalg.svd(design_matrix(D) / np.sqrt(len(D)))[1][5] / th2 ** 2)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.semilogx([g[0] for g in grids], vals, "o-", ms=4, label="computed")
    ax.axhline(1 / np.sqrt(24), color="k", ls="--", lw=0.9,
               label=r"$1/\sqrt{24}=0.20412$")
    ax.set_xlabel("polar grid points"); ax.set_ylabel(r"$s_6/\theta^2$")
    ax.set_title(r"refinement at $\theta=2^\circ$"); ax.legend(fontsize=7)
    save(fig, "fig2_refine")

    # Fig 3 -- parity scaling
    g = Geometry("cone", theta=np.pi, delta=1.0)
    D120 = dirs_fibonacci(120)
    HS = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    for fn, mk in ((Poly(), "o"), (Trig(), "s"), (Gaussian(), "^")):
        E = []
        for h in HS:
            ev = Evaluator(fn, X0, 0.0, np.random.default_rng(1))
            U, q = probe_sphere(ev, D120, g, lambda u, h=h: h)
            e, *_ = harmonic_energies(U, q, lmax=4)
            E.append(np.hypot(e[1], e[3]))
        ax.loglog(HS, E, mk + "-", ms=3.5, label=fn.name)
    E = []
    for h in HS:
        ev = Evaluator(Poly(), X0, 1e-4, np.random.default_rng(1))
        U, q = probe_sphere(ev, D120, g, lambda u, h=h: h)
        e, *_ = harmonic_energies(U, q, lmax=4)
        E.append(np.hypot(e[1], e[3]))
    ax.loglog(HS, E, "v--", ms=3.5, label=r"poly, $\sigma=10^{-4}$")
    ax.loglog(HS, 0.9 * HS, "k--", lw=0.8, label=r"slope $+1$")
    ax.loglog(HS, 2e-5 * HS ** -2.0, "k:", lw=0.8, label=r"slope $-2$")
    ax.set_xlabel("$h$"); ax.set_ylabel("odd-band energy")
    ax.legend(fontsize=6.5); ax.set_title("parity scaling")
    save(fig, "fig3_parity")

    # Fig 4 -- order
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    fn = Poly(); Ht = fn.hess(X0)
    HS4 = np.array([0.2, 0.1, 0.05])
    for name, gg, mk in [("full sphere", Geometry("cone", theta=np.pi, delta=1.0),
                          lambda: dirs_fibonacci(120)),
                         (r"icosa $\pm$ pairs", Geometry("cone", theta=np.pi, delta=1.0),
                          lambda: np.vstack([dirs_icosahedral(), -dirs_icosahedral()])),
                         ("hemisphere", Geometry("cone", theta=np.deg2rad(90), delta=1.0),
                          lambda: dirs_fibonacci(120)),
                         (r"cap $45^\circ$", Geometry("cone", theta=np.deg2rad(45), delta=1.0),
                          lambda: dirs_fibonacci(400))]:
        E = []
        for h in HS4:
            ev = Evaluator(fn, X0, 0.0, np.random.default_rng(1))
            U, q = probe_sphere(ev, mk(), gg, lambda u, h=h: h)
            hv = np.linalg.lstsq(design_matrix(U), q, rcond=None)[0]
            E.append(np.linalg.norm(vec_to_H(hv) - Ht, "fro"))
        ax.loglog(HS4, E, "o-", ms=3.5,
                  label=f"{name} (order {np.log2(E[0]/E[2])/2:.2f})")
    ax.set_xlabel("$h$"); ax.set_ylabel(r"$\|\hat H-H\|_F$")
    ax.legend(fontsize=6.5); ax.set_title("antipodal availability")
    save(fig, "fig4_order")

    # Fig 5 -- plateau
    LAM = np.logspace(-8, 1, 19)
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    fn = Poly(); c3t = fn.third_scale(X0)
    fs = abs(float(fn.f(X0[None])[0])) + 1.0
    for deg, mk in ((90, "o"), (50, "s"), (30, "^"), (20, "v")):
        gg = Geometry("cone", theta=np.deg2rad(deg), delta=0.5)
        r = [(c3_harmonic(fn, 1e-4 * fs, gg, ridge=l) or np.nan) / c3t for l in LAM]
        ax.loglog(LAM, r, mk + "-", ms=3, label=rf"$\theta={deg}^\circ$")
    ax.axhspan(0.5, 2.0, color="k", alpha=0.08)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel(r"$\lambda/\|B^\top B\|_2$"); ax.set_ylabel(r"$\widehat C_3/C_3$")
    ax.legend(fontsize=6.5); ax.set_title("regularisation plateau")
    save(fig, "fig5_plateau")

    # Fig 6 -- pilot usability
    degs6 = [90, 70, 50, 40, 30, 20]
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    a, b, c = [], [], []
    for d in degs6:
        gg = Geometry("cone", theta=np.deg2rad(d), delta=0.5)
        a.append((c3_harmonic(fn, 1e-4 * fs, gg) or np.nan) / c3t)
        b.append((c3_harmonic(fn, 1e-4 * fs, gg, ridge=1e-3) or np.nan) / c3t)
        c.append(c3_direct(fn, 1e-4 * fs, gg) / c3t)
    ax.semilogy(degs6, a, "o-", ms=3.5, label="unregularised")
    ax.semilogy(degs6, b, "s-", ms=3.5, label=r"regularised, $\lambda_{rel}=10^{-3}$")
    ax.semilogy(degs6, c, "^-", ms=3.5, label="direct 3rd difference")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel(r"cap half-angle $\theta$ (deg)")
    ax.set_ylabel(r"$\widehat C_3/C_3$")
    ax.legend(fontsize=6.5); ax.set_title("pilot usability")
    save(fig, "fig6_pilot")
    print("\nfigures written: fig0_roadmap, fig1_cone, fig2_refine, fig3_parity, fig4_order, "
          "fig5_plateau, fig6_pilot  (.pdf and .png)")


# =============================================================================
# Driver
# =============================================================================

def run_theorems():
    print("=" * 96)
    print("THEOREM AND OBSERVATION VERIFICATION")
    print("=" * 96)
    print(f"\nanalytic Hessians verified against 4th-order FD, max error "
          f"{verify_hessians():.2e}")
    obs_identifiability()
    obs_direction_sets()
    thm_cone_singular()
    obs_constants()
    obs_parity_scaling()
    obs_order()
    obs_parity_separation()
    obs_plateau()
    obs_tracking()
    obs_pilot_size()
    obs_R_sweep()
    obs_dopt()


def run_benchmark():
    benchmark_main()
    benchmark_msweep()
    benchmark_sensitivity()


def main():
    global QUICK
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--theorems", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="reduced replicate counts, for a fast smoke test")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    QUICK = args.quick
    if not any((args.all, args.verify, args.theorems, args.benchmark, args.figures)):
        args.all = True
    if args.verify:
        print(f"analytic Hessians verified, max error {verify_hessians():.2e}")
    if args.all or args.theorems:
        run_theorems()
    if args.all or args.benchmark:
        run_benchmark()
    if args.all or args.figures:
        make_figures(args.outdir)


if __name__ == "__main__":
    main()


# =============================================================================
# PART 8.  Additional figures for the manuscript
# =============================================================================

def fig_cone_geometry(outdir="."):
    """Schematic of the admissible cap, with the block decomposition annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, FancyArrowPatch

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for k, (deg, lab) in enumerate(((80, r"wide cap: $\theta=80^\circ$"),
                                    (25, r"narrow cap: $\theta=25^\circ$"))):
        A = ax[k]
        A.set_aspect("equal"); A.axis("off")
        A.set_xlim(-1.35, 1.35); A.set_ylim(-1.15, 1.35)
        t = np.linspace(0, 2 * np.pi, 200)
        A.plot(np.cos(t), np.sin(t), color="0.75", lw=0.8)
        A.add_patch(Wedge((0, 0), 1.0, 90 - deg, 90 + deg, width=0.0,
                          fc="none", ec="black", lw=0.0))
        for a in np.linspace(-np.deg2rad(deg), np.deg2rad(deg), 9):
            A.add_patch(FancyArrowPatch((0, 0),
                                        (np.sin(a), np.cos(a)),
                                        arrowstyle="-|>", mutation_scale=6,
                                        lw=0.8, color="black"))
        tt = np.linspace(np.pi / 2 - np.deg2rad(deg), np.pi / 2 + np.deg2rad(deg), 60)
        A.plot(1.06 * np.cos(tt), 1.06 * np.sin(tt), color="black", lw=1.6)
        A.plot([0], [0], "ko", ms=3.5)
        A.text(0.06, -0.11, "$x$", fontsize=9)
        A.text(0, 1.22, lab, ha="center", fontsize=8.5)
        A.annotate("", xy=(0.0, 1.0), xytext=(0.0, 0.0),
                   arrowprops=dict(arrowstyle="-", ls=":", lw=0.9, color="0.4"))
        A.text(0.03, 0.55, r"$e_3$", fontsize=8, color="0.35")
        kap = 4.899 / np.deg2rad(deg) ** 2
        A.text(0, -1.05, rf"$\kappa\approx{kap:.0f}$", ha="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_cone_geometry.pdf")
    fig.savefig(f"{outdir}/fig_cone_geometry.png", dpi=300)
    plt.close(fig)


def fig_comparison(outdir=".", n_rep=15):
    """Error against cone aperture for the estimator families, matched budget."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degs = [90, 60, 45, 30, 20]
    fn = Poly()
    series = {"quad_reg": [], "MLS": [], "onesidedFD": [], "dir_aniso": []}
    for d in degs:
        g = Geometry("cone", theta=np.deg2rad(d))
        c = bench_cell(fn, g, 1e-4, n_rep=n_rep, seed=4242)
        for k in series:
            series[k].append(c[k]["mean"])
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    style = {"quad_reg": ("o-", "quadratic regression"),
             "MLS": ("s-", "moving least squares"),
             "onesidedFD": ("^--", "one-sided FD"),
             "dir_aniso": ("v--", "directional, anisotropic span")}
    for k, (mk, lab) in style.items():
        ax.semilogy(degs, series[k], mk, ms=4, lw=1.0, label=lab)
    ax.invert_xaxis()
    ax.set_xlabel(r"cap half-angle $\theta$ (deg)")
    ax.set_ylabel(r"mean $\|\widehat H-H\|_F$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_comparison.pdf")
    fig.savefig(f"{outdir}/fig_comparison.png", dpi=300)
    plt.close(fig)
    return degs, series


def fig_thm_illustration(outdir="."):
    """Illustration of the cone-conditioning theorem: the L2 response of the
    three curvature components against the aperture, with the predicted slopes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degs = np.array([80, 60, 45, 30, 20, 10, 5, 2])
    th = np.deg2rad(degs)
    S = []
    for d in degs:
        D = cap_dirs(np.deg2rad(d), 60, 96, "area")
        S.append(np.linalg.svd(design_matrix(D) / np.sqrt(len(D)))[1])
    S = np.array(S)

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.loglog(th, S[:, 0], "o-", ms=4, lw=1.0,
              label=r"normal, $a^{\top}\!Ha$   (slope $0$)")
    ax.loglog(th, S[:, 1], "s-", ms=4, lw=1.0,
              label=r"mixed, $b$   (slope $1$)")
    ax.loglog(th, S[:, 5], "^-", ms=4, lw=1.0,
              label=r"tangential, $T$   (slope $2$)")
    ax.loglog(th, th / np.sqrt(2), "k--", lw=0.7)
    ax.loglog(th, th ** 2 / np.sqrt(24), "k:", lw=0.7)
    ax.set_xlabel(r"aperture $\theta$ (rad)")
    ax.set_ylabel("response to a unit perturbation")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_thm_illustration.pdf")
    fig.savefig(f"{outdir}/fig_thm_illustration.png", dpi=300)
    plt.close(fig)
