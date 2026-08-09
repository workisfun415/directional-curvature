#!/usr/bin/env python3
"""
kr_framework.py
===============
A reusable 2D/3D framework for one-sided directional curvature measurement.

It returns three DISTINCT quantities, which must not be conflated:

    H_hat      the recovered Hessian                     (curvature)
    kappa      conditioning of the direction geometry    (geometry)
    C3_hat     estimated third-order Taylor remainder    (model adequacy)

plus a span recommendation, an attainable-order flag and a reliability verdict.

The framework is deliberately conservative about what it claims. See
`when_not_to_use()` for the documented cases in which a plain quadratic
regression should be preferred; the accompanying study found that it usually
should be.

Reference: R. Pasupuleti, "Geometry, conditioning, and limits of one-sided
directional curvature recovery", https://doi.org/10.5281/zenodo.21793101

Usage
-----
    from kr_framework import Geometry, measure
    g = Geometry.cone(dim=3, half_angle_deg=45, max_span=0.4)
    rep = measure(f, x0, g, sigma=1e-6)
    print(rep)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

__all__ = ["Geometry", "Report", "measure", "when_not_to_use",
           "kappa_reference", "antipodal_sampled_fraction", "R_DEFAULT"]

R_DEFAULT = 0.5          # interior node parameter; 0.5 minimises noise gain


# =====================================================================
# Geometry engine
# =====================================================================

@dataclass
class Geometry:
    """The feasible set as seen from the evaluation point."""
    dim: int
    kind: str = "free"                       # free | cone | slab | wedge | custom
    half_angle: float = np.pi                # radians, for cone
    max_span: float = 1.0
    dist: float = 0.05                       # distance to boundary, slab/wedge
    axis: Optional[np.ndarray] = None
    feasible: Optional[Callable] = None      # custom predicate on a unit vector

    def __post_init__(self):
        if self.axis is None:
            a = np.zeros(self.dim); a[-1] = 1.0
            self.axis = a
        self.axis = np.asarray(self.axis, float)
        self.axis = self.axis / np.linalg.norm(self.axis)

    # -- constructors -------------------------------------------------
    @classmethod
    def free(cls, dim, max_span=1.0):
        return cls(dim=dim, kind="free", max_span=max_span)

    @classmethod
    def cone(cls, dim, half_angle_deg, max_span=1.0, axis=None):
        return cls(dim=dim, kind="cone", half_angle=np.deg2rad(half_angle_deg),
                   max_span=max_span, axis=axis)

    @classmethod
    def slab(cls, dim, dist, max_span=1.0, axis=None):
        return cls(dim=dim, kind="slab", dist=dist, max_span=max_span, axis=axis)

    @classmethod
    def wedge(cls, dim, dist, max_span=1.0):
        return cls(dim=dim, kind="wedge", dist=dist, max_span=max_span)

    # -- queries ------------------------------------------------------
    def hmax(self, u) -> float:
        """Largest usable span along the unit direction u."""
        u = np.asarray(u, float)
        if self.feasible is not None:
            return self.max_span if self.feasible(u) else 0.0
        if self.kind == "free":
            return self.max_span
        if self.kind == "cone":
            return self.max_span if float(u @ self.axis) >= np.cos(self.half_angle) else 0.0
        if self.kind == "slab":
            un = float(u @ self.axis)
            return self.max_span if un >= 0 else min(self.max_span, self.dist / abs(un))
        if self.kind == "wedge":
            lim = self.max_span
            for k in range(self.dim):
                n = np.zeros(self.dim); n[k] = 1.0
                un = float(u @ n)
                if un < 0:
                    lim = min(lim, self.dist / abs(un))
            return lim
        raise ValueError(self.kind)

    def effective_aperture(self, n_probe=2000, seed=0) -> float:
        """Fraction of the sphere that is usable, expressed as the half-angle of
        an equivalent cap. This is the theta that enters kappa ~ C theta^-2."""
        if self.kind == "cone":
            return self.half_angle
        rng = np.random.default_rng(seed)
        V = rng.normal(size=(n_probe, self.dim))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        ok = np.array([self.hmax(v) > 1e-12 for v in V])
        frac = max(ok.mean(), 1e-6)
        if self.dim == 2:
            return frac * np.pi
        return float(np.arccos(np.clip(1 - 2 * frac, -1, 1)))

    def antipodal_feasible_fraction(self, dirs, tol=1e-12) -> float:
        """Fraction of directions whose antipode is PHYSICALLY AVAILABLE. This is
        a property of the geometry and does NOT determine the attainable order."""
        return float(np.mean([self.hmax(u) > tol and self.hmax(-u) > tol
                              for u in dirs]))


def antipodal_sampled_fraction(dirs, tol=1e-8) -> float:
    """Fraction of directions whose antipode is actually PRESENT in the probe
    set. This is the quantity in the antipodal-order proposition: the odd term
    cancels in the least-squares aggregation only between sampled pairs, so a set
    whose antipodes are merely feasible but not probed remains first order."""
    D = np.atleast_2d(np.asarray(dirs, float))
    hit = 0
    for u in D:
        if np.min(np.linalg.norm(D + u, axis=1)) < tol:
            hit += 1
    return hit / len(D)


# =====================================================================
# Direction sets
# =====================================================================

def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def icosahedral_axes():
    p = (1 + np.sqrt(5)) / 2
    return np.array([_unit(v) for v in
                     [[0, 1, p], [0, 1, -p], [1, p, 0],
                      [1, -p, 0], [p, 0, 1], [-p, 0, 1]]])


def fibonacci_sphere(m):
    i = np.arange(m) + 0.5
    ph = np.arccos(1 - 2 * i / m)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)]


def circle_dirs(m):
    a = np.linspace(0, np.pi, m, endpoint=False)
    return np.c_[np.cos(a), np.sin(a)]


def design_matrix(dirs):
    """Rows map the Frobenius-consistent Hessian coordinates to u'Hu."""
    U = np.atleast_2d(np.asarray(dirs, float))
    n = U.shape[1]; s = np.sqrt(2)
    cols = [U[:, i] ** 2 for i in range(n)]
    cols += [s * U[:, i] * U[:, j] for i in range(n) for j in range(i + 1, n)]
    return np.column_stack(cols)


def vec_to_H(h, n):
    H = np.zeros((n, n)); s = np.sqrt(2)
    for i in range(n):
        H[i, i] = h[i]
    k = n
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = h[k] / s
            k += 1
    return H


def select_directions(geom, m=None, seed=4, n_cand=400):
    """Build a direction set INSIDE the feasible region by sequential
    D-optimal selection. Never compress an existing set: a fixed antipodally
    symmetric set loses half its members at a boundary."""
    n = geom.dim
    m = (n * (n + 1) // 2 if m is None else m)
    if geom.kind == "free":
        return icosahedral_axes() if n == 3 and m == 6 else (
            fibonacci_sphere(max(m, 12)) if n == 3 else circle_dirs(m))
    rng = np.random.default_rng(seed)
    C = []
    while len(C) < n_cand:
        v = rng.normal(size=n); v /= np.linalg.norm(v)
        if geom.hmax(v) > 1e-12:
            C.append(v)
    C = np.array(C)
    A = design_matrix(C)
    d = A.shape[1]
    chosen, M = [], np.zeros((d, d))
    for _ in range(m):
        best, bi = -np.inf, None
        for i in range(len(C)):
            sign, ld = np.linalg.slogdet(M + np.outer(A[i], A[i]) + 1e-12 * np.eye(d))
            v = ld if sign > 0 else -np.inf
            if v > best:
                best, bi = v, i
        chosen.append(C[bi]); M += np.outer(A[bi], A[bi])
    return np.array(chosen)


# =====================================================================
# Core measurement
# =====================================================================

def probe(f, x0, u, h, R=R_DEFAULT, cache=None):
    """One-sided directional probe. Three evaluations, all on [x0, x0+h u]."""
    def ev(y):
        if cache is None:
            return float(f(y))
        k = tuple(np.round(y, 12))
        if k not in cache:
            cache[k] = float(f(y))
        return cache[k]
    f0, f1, fR = ev(x0), ev(x0 + h * u), ev(x0 + R * h * u)
    return 2.0 * ((1 - R) * f0 + R * f1 - fR) / (R * (1 - R) * h * h)


def optimal_span(hmax, sigma, c3, R=R_DEFAULT, fscale=1.0):
    """Balance truncation C3 h against noise sigma h^-2; clip at feasibility."""
    sigma = max(sigma, 1e-16 * fscale)
    nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
    C3 = (1 + R) / 3.0 * max(c3, 1e-14)
    Cs = 2.0 * nu / (R * (1 - R)) * sigma
    return float(min(hmax, (2.0 * Cs / C3) ** (1 / 3)))


def _poly_basis(U, lmax=3):
    """Monomials by total degree, for the parity split."""
    n = U.shape[1]
    cols, degs = [np.ones(len(U))], [0]
    for d in range(1, lmax + 1):
        for combo in itertools.combinations_with_replacement(range(n), d):
            col = np.ones(len(U))
            for c in combo:
                col = col * U[:, c]
            cols.append(col); degs.append(d)
    return np.column_stack(cols), np.array(degs)


def estimate_C3(dirs, q, h, R=R_DEFAULT, sigma=0.0, ridge=1e-3):
    """Odd-degree content of the probe field estimates ||D^3 f||, to O(h^2).
    Regularised, because the parity split degenerates on a narrow cap."""
    B, degs = _poly_basis(np.atleast_2d(dirs), 3)
    if len(q) <= B.shape[1]:
        return None                            # rank condition m > dim(basis)
    G = B.T @ B
    s1 = float(np.linalg.eigvalsh(G).max())
    c = np.linalg.solve(G + ridge * s1 * np.eye(B.shape[1]), B.T @ q)
    mask = np.isin(degs, [1, 3])
    odd = float(np.linalg.norm(B[:, mask] @ c[mask]) / np.sqrt(len(q)))
    if sigma > 0:
        nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
        sq = 2 * sigma * nu / (R * (1 - R) * h * h)
        n = np.atleast_2d(dirs).shape[1]
        p = (4 if n == 2 else 10)
        odd = np.sqrt(max(odd ** 2 - p * sq ** 2 / len(q), 0.0))
    return max(3.0 * odd / ((1 + R) * h), 0.0)


# =====================================================================
# Reliability report
# =====================================================================

@dataclass
class Report:
    dim: int
    n_directions: int
    n_evaluations: int
    aperture_deg: float
    kappa: float
    kappa_reference: float
    R: float
    spans: np.ndarray
    H_hat: np.ndarray
    C3_hat: Optional[float]
    C3_status: str
    antipodal_sampled: float
    antipodal_feasible: float
    expected_order: int
    sigma: float
    H_uncertainty: Optional[np.ndarray] = None
    flags: list = field(default_factory=list)
    verdict: str = "UNKNOWN"

    def __str__(self):
        L = [f"dimension            {self.dim}D",
             f"directions           {self.n_directions}",
             f"evaluations          {self.n_evaluations}",
             f"feasible aperture    {self.aperture_deg:.0f} deg",
             f"condition number     {self.kappa:.2f}  "
             f"(free geometry: {self.kappa_reference:.2f})",
             f"node parameter R     {self.R:.2f}",
             f"span   min/max       {self.spans.min():.4g} / {self.spans.max():.4g}",
             f"antipodal sampled    {self.antipodal_sampled:.0%}",
             f"antipodal feasible   {self.antipodal_feasible:.0%}",
             f"expected order       {self.expected_order}",
             f"estimated C3         " +
             ("not estimated" if self.C3_hat is None
              else f"{self.C3_hat:.4g}"),
             f"C3 pilot status      {self.C3_status}",
             f"noise level sigma    {self.sigma:.2g}",
             "Hessian:"]
        for row in np.atleast_2d(self.H_hat):
            L.append("   " + "  ".join(f"{v: .6g}" for v in row))
        if self.H_uncertainty is not None:
            L.append(f"Hessian sd (approx)  {self.H_uncertainty.max():.3g} (max entry)")
        for fl in self.flags:
            L.append(f"  ! {fl}")
        L.append(f"reliability          {self.verdict}")
        return "\n".join(L)


def kappa_reference(dim):
    """Condition number of the unrestricted geometry."""
    return {2: np.sqrt(2), 3: 0.5 * np.sqrt(10)}.get(dim, np.nan)


def measure(f, x0, geom: Geometry, sigma=0.0, m=None, R=R_DEFAULT,
            c3_prior=None, pilot=True, seed=4,
            on_c3_unavailable="defer") -> Report:
    """Measure curvature, conditioning and Taylor-model adequacy at x0."""
    x0 = np.asarray(x0, float)
    pilot_span, span_spread = None, 1.0
    c3_status = "NOT-ATTEMPTED"
    n = geom.dim
    dirs = select_directions(geom, m=m, seed=seed)
    dirs = np.array([u for u in dirs if geom.hmax(u) > 1e-12])
    ndir = len(dirs)
    dmin = n * (n + 1) // 2
    cache = {}
    fscale = abs(float(f(x0))) + 1.0

    # pilot pass at a provisional span, to estimate C3
    c3_estimated = None
    c3 = c3_prior
    if c3_prior is not None:
        c3_status = "USER-SUPPLIED"
    if pilot and c3 is None:
        # The probe expansion carries the cubic term with coefficient
        # (1+R)h/3, so a direction-dependent span would weight D^3 f(u)
        # differently in each observation and the joint fit would be
        # inconsistent. Use ONE pilot span, feasible in every direction.
        hm_all = [geom.hmax(u) for u in dirs]
        h_pilot = 0.5 * float(min(hm_all))
        pilot_span = h_pilot
        span_spread = max(hm_all) / max(min(hm_all), 1e-30)
        if h_pilot <= 1e-12:
            c3_status = "SPAN-LIMITED"
        else:
            q0 = np.array([probe(f, x0, u, h_pilot, R, cache) for u in dirs])
            est = estimate_C3(dirs, q0, h_pilot, R, sigma)
            if est is None:
                c3_status = "UNDER-DETERMINED"
            elif est <= 0:
                # the debiasing of Proposition (noise) truncated to zero
                c3_status = ("SPAN-LIMITED" if span_spread > 3.0
                             else "NOISE-LIMITED")
            else:
                c3, c3_estimated, c3_status = est, est, "VALID"

    # No arbitrary substitution.  If C3 is unknown the span rule cannot be
    # applied, so an explicit fallback policy is used instead.
    c3_known = c3 is not None and c3 > 0
    if not c3_known:
        if on_c3_unavailable == "require":
            raise ValueError(
                f"C3 unavailable ({c3_status}); supply c3_prior or choose "
                "on_c3_unavailable='defer' or 'hmax'")
        c3_for_spans = None

    if c3_known:
        spans = np.array([optimal_span(geom.hmax(u), sigma, c3, R, fscale)
                          for u in dirs])
    else:
        # conservative: the largest feasible span, which minimises the noise
        # contribution at the cost of an unbounded truncation term
        spans = np.array([geom.hmax(u) for u in dirs])
    q = np.array([probe(f, x0, u, h, R, cache) for u, h in zip(dirs, spans)])

    A = design_matrix(dirs)
    kappa = float(np.linalg.cond(A)) if ndir >= dmin else np.inf
    nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
    c3_w = c3 if c3_known else 0.0     # weights fall back to noise only
    w = []
    for h in spans:
        bias = (1 + R) / 3.0 * h * c3_w
        noise = 2 * nu * sigma / (R * (1 - R) * max(h, 1e-14) ** 2)
        w.append(1.0 / max(bias ** 2 + noise ** 2, 1e-30))
    w = np.array(w)

    if ndir >= dmin:
        W = np.diag(w)
        hv = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ q, rcond=None)[0]
        H = vec_to_H(hv, n)
        cov = np.linalg.pinv(A.T @ W @ A)
        Hsd = vec_to_H(np.sqrt(np.abs(np.diag(cov))), n)
    else:
        H, Hsd = np.full((n, n), np.nan), None

    ap_feas = geom.antipodal_feasible_fraction(dirs)
    ap = antipodal_sampled_fraction(dirs)
    order = 2 if ap > 0.999 else 1
    theta = geom.effective_aperture()
    kref = kappa_reference(n)

    flags = []
    if ndir < dmin:
        flags.append(f"only {ndir} feasible directions; {dmin} are required "
                     "for the Hessian to be identifiable")
    if kappa > 10 * kref:
        flags.append(f"conditioning degraded {kappa/kref:.0f}x relative to free "
                     "geometry; the tangential block is least reliable")
    if order == 1:
        msg = ("antipodal partners are not SAMPLED: first order only, and no "
               "direction count or weighting recovers the second")
        if ap_feas > 0.999:
            msg += ("; the antipodes ARE feasible here, so adding them to the "
                    "probe set would restore second order")
        flags.append(msg)
    need = _poly_basis(np.atleast_2d(dirs), 3)[0].shape[1] + 1
    if len(q) < need:
        flags.append(f"C3 not estimated: the parity split needs m >= {need} "
                     f"directions in {n}D, {ndir} were feasible; supply "
                     "c3_prior or increase m")
    if not c3_known and c3_status in ("SPAN-LIMITED", "NOISE-LIMITED",
                                      "UNDER-DETERMINED"):
        flags.append(f"C3 unavailable ({c3_status}); spans fall back to "
                     "h_max and the truncation term is not controlled. A "
                     "well-poised quadratic regression is preferred here")
    if c3_estimated is None and pilot is True and c3_prior is None:
        flags.append("C3 pilot was noise-limited: a single pilot span is needed "
                     "for a consistent fit of the cubic term, and the common "
                     f"span was set by the most restricted direction"
                     + (f" (span spread {span_spread:.0f}x)" if span_spread > 3
                        else ""))
    if sigma > 0 and np.any(spans >= 0.999 * np.array([geom.hmax(u) for u in dirs])):
        flags.append("span clipped by feasibility, so the noise-truncation "
                     "balance could not be reached")
    if geom.kind == "free" and sigma == 0:
        flags.append("unrestricted two-sided geometry: prefer central "
                     "differences or a poised quadratic regression")

    hw = float(np.median(spans)) if len(spans) else 0.0
    normH = float(np.linalg.norm(H, "fro")) if H is not None and np.all(np.isfinite(H)) else 0.0
    if c3_known and normH > 0 and hw > 0 and np.isfinite(kappa):
        nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
        err = float(kappa * ((1 + R) / 3.0 * c3 * hw
                             + 2 * sigma * nu / (R * (1 - R) * hw ** 2)))
        rho = err / normH
    else:
        rho = None
    if rho is not None and rho > 0.1:
        flags.append(f"predicted error is {rho:.2g} times the recovered curvature")

    if ndir < dmin or not np.isfinite(kappa):
        verdict = "UNUSABLE"
    elif not c3_known and on_c3_unavailable == "defer":
        verdict = "DEFER-TO-QUADRATIC-REGRESSION"
    elif rho is None:
        verdict = "CAUTION"
    elif rho > 0.4:
        verdict = "DEFER"
    elif rho > 0.1 or kappa > 5 * kref or order == 1 or len(flags) >= 3:
        verdict = "CAUTION"
    else:
        verdict = "GOOD"

    return Report(dim=n, n_directions=ndir, n_evaluations=len(cache),
                  aperture_deg=float(np.rad2deg(theta)), kappa=kappa,
                  kappa_reference=kref, R=R, spans=spans, H_hat=H,
                  C3_hat=(float(c3_prior) if c3_prior is not None
                          else (None if c3_estimated is None
                                else float(c3_estimated))),
                  C3_status=c3_status,
                  antipodal_sampled=ap, antipodal_feasible=ap_feas,
                  expected_order=order, sigma=sigma,
                  H_uncertainty=Hsd, flags=flags, verdict=verdict)


def when_not_to_use():
    """Documented cases in which this framework should NOT be the primary
    curvature estimator. Derived from the accompanying benchmark, in which
    quadratic regression matched or outperformed directional schemes in all 48
    tested configurations at matched evaluation budget."""
    return [
        "a well-poised quadratic regression can be built inside the feasible "
        "region: it was at least as accurate in every configuration tested",
        "unrestricted two-sided sampling is available: use central differences",
        "maximum Hessian accuracy is the objective rather than diagnosis",
        "the function is not C^3 near x0: the parity expansion does not hold",
        "noise dominates the probe, sigma h^-2 >> C3 h at every feasible span",
        "the feasible direction set is very narrow: kappa ~ theta^-2 makes the "
        "tangential block unrecoverable",
        "the pilot cost of estimating C3 cannot be amortised over many points",
    ]


if __name__ == "__main__":
    print(__doc__.split("Usage")[0])
    print("=" * 66)
    print("REFERENCE CONDITIONING          2D          3D")
    print(f"  unrestricted            {kappa_reference(2):>8.3f}    {kappa_reference(3):>8.3f}")
    print(f"  asymptotic kappa*th^2   {1.5*np.sqrt(5):>8.3f}    {np.sqrt(24):>8.3f}")
    print()

    def f2(p):
        x, y = p
        return np.exp(x) * np.cos(y) + 0.3 * x ** 3

    def f3(p):
        x, y, z = p
        return np.exp(x + 0.5 * y) * np.cos(z) + 0.2 * x ** 3

    for lab, fn, g, sig in (
        ("2D free", f2, Geometry.free(2, max_span=0.3), 0.0),
        ("2D cone 45", f2, Geometry.cone(2, 45, max_span=0.3), 1e-6),
        ("3D free", f3, Geometry.free(3, max_span=0.3), 0.0),
        ("3D hemisphere", f3, Geometry.cone(3, 90, max_span=0.3), 1e-6),
        ("3D cone 20", f3, Geometry.cone(3, 20, max_span=0.3), 1e-6),
        ("3D slab", f3, Geometry.slab(3, dist=0.03, max_span=0.3), 1e-6),
    ):
        x0 = np.zeros(g.dim) + 0.1
        rep = measure(fn, x0, g, sigma=sig, m=12 if g.dim == 3 else 8)
        print("-" * 66)
        print(f"[{lab}]")
        print(rep)
    print("-" * 66)
    print("\nWHEN NOT TO USE THIS FRAMEWORK")
    for s in when_not_to_use():
        print("  -", s)
