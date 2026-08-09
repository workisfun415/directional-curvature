#!/usr/bin/env python3
"""
dircurv2d.py
============
Two-dimensional directional curvature measurement on **measured, gridded data**.

This is the implementable form of the framework: it takes a 2D array of samples,
a pixel spacing and an optional validity mask, and returns per-pixel maps of

    H_hat     the recovered Hessian (three components)
    kappa     conditioning of the locally available direction geometry
    C3_hat    third-order magnitude, when it can be estimated at all
    aperture  the usable angular range at that pixel, in degrees
    order     attainable order, 1 or 2, from SAMPLED antipodal partners
    status    why C3 is or is not available
    verdict   GOOD / MODERATE / LOW / DEFER / UNUSABLE

Nothing here estimates stiffness, and nothing competes with an inversion. The
output is a reliability map: where the local sampling geometry has made the
curvature estimate untrustworthy, and why.

Key differences from the analytic-geometry version:
  * feasible directions come from the MASK, by marching along each ray until it
    leaves the valid region, so irregular boundaries are handled directly;
  * the field is a grid, so probe points are obtained by bilinear interpolation
    and need not land on pixel centres;
  * everything is vectorised over directions for one pixel, and looped over
    pixels, so a 256x256 slice with 16 directions runs in seconds.

Conventions, fixed and not varied:
    C3(x) = [ mean_u |D^3 f(x)[u,u,u]|^2 ]^{1/2}   over the sampled directions
    ||H||  = ||H||_F
    kappa computed in the Frobenius-consistent basis (H11, H22, sqrt2 H12)

Reference: R. Pasupuleti, "Geometry, conditioning, and limits of one-sided
directional curvature recovery".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["GridField", "MaskSupportError", "predicted_error",
           "resolution_guard", "local_directions",
           "measure_pixel", "reliability_maps", "R_DEFAULT",
           "KAPPA_FULL_CIRCLE"]

R_DEFAULT = 0.5
KAPPA_FULL_CIRCLE = np.sqrt(2.0)     # unrestricted 2D reference, exact
P_ODD_2D = 4                         # dim H_1 + dim H_3 on the circle


# =====================================================================
# Gridded field
# =====================================================================

class MaskSupportError(ValueError):
    """Raised when a bicubic stencil would need data outside the valid mask."""


class GridField:
    """A measured 2D field, sampled on a regular grid, with bilinear access.

    Parameters
    ----------
    array : (ny, nx) float
        Sample values. Complex input is not accepted. The probe expansion is
        LINEAR in f, so it may be applied component-wise to the real and
        imaginary parts of a complex phasor. It may NOT be applied to the
        amplitude |u|: taking the modulus before differentiation is a nonlinear
        operation and the parity expansion does not cover it.
    spacing : float or (2,)
        Pixel size in physical units (e.g. mm). Scalar means isotropic.
    mask : (ny, nx) bool, optional
        True where the sample is valid. Defaults to all valid.
    """

    def __init__(self, array, spacing=1.0, mask=None):
        if np.iscomplexobj(array):
            raise TypeError("complex input: apply component-wise to the real "
                            "and imaginary parts; the amplitude is nonlinear "
                            "and outside the theory")
        a = np.asarray(array, float)
        if a.ndim != 2:
            raise ValueError("array must be 2D")
        self.a = a
        self.ny, self.nx = a.shape
        sp = np.atleast_1d(np.asarray(spacing, float))
        self.spacing = np.array([sp[0], sp[-1]]) if sp.size >= 2 else np.array([sp[0], sp[0]])
        self.mask = (np.ones_like(a, bool) if mask is None
                     else np.asarray(mask, bool))
        if self.mask.shape != a.shape:
            raise ValueError("mask shape must match array shape")

    # -- coordinate helpers ------------------------------------------
    def to_index(self, p):
        """Physical offset (dx, dy) -> fractional index offset (dy, dx)."""
        dx, dy = np.asarray(p, float)
        return np.array([dy / self.spacing[0], dx / self.spacing[1]])

    def inside(self, idx) -> bool:
        iy, ix = idx
        return (0.0 <= iy <= self.ny - 1) and (0.0 <= ix <= self.nx - 1)

    def valid(self, idx) -> bool:
        """Mask validity at a fractional index: all four corners must be valid."""
        iy, ix = idx
        if not self.inside(idx):
            return False
        y0, x0 = int(np.floor(iy)), int(np.floor(ix))
        ys = np.arange(y0 - 1, y0 + 3)
        xs = np.arange(x0 - 1, x0 + 3)
        if ys.min() < 0 or xs.min() < 0 or ys.max() > self.ny - 1 \
                or xs.max() > self.nx - 1:
            return False
        return bool(self.mask[np.ix_(ys, xs)].all())

    @staticmethod
    def _crom(t):
        """Catmull-Rom weights for the four samples at -1,0,1,2. This kernel
        reproduces cubic polynomials exactly, so neither the quadratic model nor
        the O(h) cubic term of the probe expansion is corrupted by
        interpolation. Bilinear would not: it reproduces only bilinear
        functions, and its error exceeds the curvature signal being measured."""
        t2, t3 = t * t, t * t * t
        return np.array([-0.5 * t3 + t2 - 0.5 * t,
                         1.5 * t3 - 2.5 * t2 + 1.0,
                         -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                         0.5 * t3 - 0.5 * t2])

    def at(self, idx, strict=False) -> float:
        """Bicubic (Catmull-Rom) sample at a fractional index.

        With strict=True the 4x4 support must lie inside the grid and inside the
        mask, and a MaskSupportError is raised otherwise. Never extrapolate
        across invalid data: silently interpolating over masked tissue would
        fabricate a curvature signal from the mask edge.
        """
        if strict and not self.valid(idx):
            raise MaskSupportError(f"bicubic support at {idx} leaves the mask")
        iy, ix = idx
        y0, x0 = int(np.floor(iy)), int(np.floor(ix))
        fy, fx = iy - y0, ix - x0
        wy, wx = self._crom(fy), self._crom(fx)
        ys = np.clip(np.arange(y0 - 1, y0 + 3), 0, self.ny - 1)
        xs = np.clip(np.arange(x0 - 1, x0 + 3), 0, self.nx - 1)
        blk = self.a[np.ix_(ys, xs)]
        return float(wy @ blk @ wx)

    def sample(self, centre_idx, offset_phys, strict=False) -> float:
        """offset_phys is (dx, dy) in physical units; the index offset is
        (dy, dx)/spacing because the array is indexed [row=y, col=x]."""
        dx, dy = np.asarray(offset_phys, float)
        return self.at(np.asarray(centre_idx, float)
                       + np.array([dy / self.spacing[0], dx / self.spacing[1]]),
                       strict=strict)


# =====================================================================
# Geometry from the mask
# =====================================================================

def hmax_along(field: GridField, centre_idx, u, h_cap, n_steps=48) -> float:
    """Largest span along direction u for which the whole ray stays valid.

    Marches outward and returns the last span at which every sampled point on
    the ray is inside the grid and inside the mask. This is what makes the
    geometry come from the data rather than from an assumed cone.
    """
    lo, hi = 0.0, float(h_cap)
    step = hi / n_steps
    good = 0.0
    t = step
    while t <= hi + 1e-12:
        if not field.valid(np.asarray(centre_idx, float) + field.to_index(t * u)):
            break
        good = t
        t += step
    return good


def local_directions(field: GridField, centre_idx, m=16, h_cap=None,
                     antipodal=True) -> Tuple[np.ndarray, np.ndarray]:
    """Directions usable at this pixel, and the span available along each.

    Directions are placed on the circle in antipodal PAIRS when `antipodal` is
    True. This matters: the odd term cancels only between directions that are
    actually probed, so a set whose antipodes are merely feasible remains first
    order. Pairs are kept only when BOTH members are usable.
    """
    if h_cap is None:
        h_cap = 0.25 * min(field.ny, field.nx) * float(min(field.spacing))
    if antipodal:
        if m % 2:
            m += 1
        ang = np.linspace(0.0, np.pi, m // 2, endpoint=False)
        base = np.c_[np.cos(ang), np.sin(ang)]
        cand = np.vstack([base, -base])
    else:
        ang = np.linspace(0.0, 2 * np.pi, m, endpoint=False)
        cand = np.c_[np.cos(ang), np.sin(ang)]

    hm = np.array([hmax_along(field, centre_idx, u, h_cap) for u in cand])
    if antipodal:
        half = len(base)
        both = np.minimum(hm[:half], hm[half:])          # keep pairs only
        keep = both > 0
        dirs = np.vstack([base[keep], -base[keep]])
        spans = np.concatenate([both[keep], both[keep]])
    else:
        keep = hm > 0
        dirs, spans = cand[keep], hm[keep]
    return dirs, spans


def aperture_deg(field: GridField, centre_idx, h_ref, h_cap=None, m=72) -> float:
    """Angular range usable AT THE SPAN h_ref, in degrees; 360 is unrestricted.

    The span matters. Almost every direction admits *some* nonzero span, so
    counting directions with h_max > 0 reports 360 degrees nearly everywhere and
    is useless. The quantity that enters the conditioning law is the set of
    directions usable at the span actually being probed.
    """
    if h_cap is None:
        h_cap = 0.25 * min(field.ny, field.nx) * float(min(field.spacing))
    ang = np.linspace(0.0, 2 * np.pi, m, endpoint=False)
    U = np.c_[np.cos(ang), np.sin(ang)]
    ok = [hmax_along(field, centre_idx, u, h_cap) >= h_ref - 1e-12 for u in U]
    return 360.0 * float(np.mean(ok))


# =====================================================================
# Core measurement
# =====================================================================

def design_matrix(dirs) -> np.ndarray:
    """Frobenius-consistent: columns for (H11, H22, sqrt2 H12)."""
    U = np.atleast_2d(np.asarray(dirs, float))
    return np.c_[U[:, 0] ** 2, U[:, 1] ** 2, np.sqrt(2) * U[:, 0] * U[:, 1]]


def vec_to_H(h) -> np.ndarray:
    s = np.sqrt(2)
    return np.array([[h[0], h[2] / s], [h[2] / s, h[1]]])


def probe(field: GridField, centre_idx, u, h, R=R_DEFAULT, strict=True) -> float:
    """One-sided directional probe; three samples on [x, x+h u].
    With strict=True (the default) it refuses to interpolate outside the mask."""
    f0 = field.at(np.asarray(centre_idx, float), strict=strict)
    f1 = field.sample(centre_idx, h * np.asarray(u), strict=strict)
    fR = field.sample(centre_idx, R * h * np.asarray(u), strict=strict)
    return 2.0 * ((1 - R) * f0 + R * f1 - fR) / (R * (1 - R) * h * h)


def antipodal_sampled_fraction(dirs, tol=1e-8) -> float:
    D = np.atleast_2d(np.asarray(dirs, float))
    return float(np.mean([np.min(np.linalg.norm(D + u, axis=1)) < tol for u in D]))


def _poly_basis_2d(U, lmax=3):
    """Monomials by total degree on the circle; 1+2+3+4 = 10 columns at lmax=3."""
    x, y = U[:, 0], U[:, 1]
    cols, degs = [np.ones(len(U))], [0]
    for d in range(1, lmax + 1):
        for k in range(d + 1):
            cols.append(x ** (d - k) * y ** k)
            degs.append(d)
    return np.column_stack(cols), np.array(degs)


def estimate_C3(dirs, q, h, sigma=0.0, R=R_DEFAULT, ridge=1e-3):
    """Odd-degree content of the probe field estimates ||D^3 f||.
    Returns (value, status)."""
    B, degs = _poly_basis_2d(np.atleast_2d(dirs), 3)
    if len(q) <= B.shape[1]:
        return None, "UNDER-DETERMINED"
    G = B.T @ B
    s1 = float(np.linalg.eigvalsh(G).max())
    c = np.linalg.solve(G + ridge * s1 * np.eye(B.shape[1]), B.T @ q)
    mask = np.isin(degs, [1, 3])
    odd = float(np.linalg.norm(B[:, mask] @ c[mask]) / np.sqrt(len(q)))
    if sigma > 0:
        nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
        sq = 2 * sigma * nu / (R * (1 - R) * h * h)
        corr = odd ** 2 - P_ODD_2D * sq ** 2 / len(q)
        if corr <= 0:
            return None, "NOISE-LIMITED"
        odd = np.sqrt(corr)
    return max(3.0 * odd / ((1 + R) * h), 0.0), "VALID"


def predicted_error(kappa, c3, h, sigma, R=R_DEFAULT):
    """Predicted error of the recovered Hessian at working span h.

    kappa * ( C3 h  +  2 sigma nu / (R(1-R) h^2) )

    the two terms being truncation and noise. Geometry enters through kappa, so
    the same span costs more where the direction set is poorly conditioned.

    The thresholds are calibrated, not chosen. rho systematically UNDERSTATES the
    true relative error, because its denominator ||H_hat||_F is itself inflated by
    noise. Measured on a smooth 2D field: rho 0.23 against a true relative error
    of 0.15, rho 0.46 against 0.57, rho 0.80 against 1.95, rho 0.99 against 5.88.
    rho_defer is therefore set below 1.
    """
    nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
    trunc = (1 + R) / 3.0 * c3 * h
    noise = 2 * sigma * nu / (R * (1 - R) * max(h, 1e-30) ** 2)
    return float(kappa * (trunc + noise))



def resolution_guard(f_values, H, h, lam_caution=4.0, lam_defer=2.0):
    """EXPERIMENTAL AND UNVALIDATED. Not used by the production path.

    This was written to flag an apparent under-resolution failure that a control
    test later showed to be a pathological choice of evaluation point, not a
    defect: at an ordinary point the recovered Hessian had 2.2 percent relative
    error against the analytic value, against 2.9 percent for plain central
    differences. The heuristic below is therefore retained only as a diagnostic
    for anyone who wants to explore the question, is off by default, and carries
    NO claim of a resolution-based reliability guarantee.

    Conservative probe-resolution warning. NOT a theorem.

    rho cannot certify the under-resolved regime, because C3 is estimated from
    the same probes: when the field varies faster than the probe can follow, the
    odd-harmonic projection under-reports the third derivative and rho looks
    small precisely when it should be large. This guard is an independent, and
    deliberately crude, check on that regime.

    A local variation scale is estimated by treating the field as locally
    oscillatory, |trace(H)| ~ k^2 |f - fbar|, giving lambda ~ 2 pi / k. The
    working span is then compared with that scale. Returns
    (lambda_estimate, status) with status in {OK, UNDER-RESOLVED-CAUTION,
    UNDER-RESOLVED-DEFER}.

    It does not attempt to correct the Hessian, only to say that the estimate
    cannot certify itself here.
    """
    f = np.asarray(f_values, float)
    amp = float(np.max(np.abs(f - f.mean()))) if f.size > 1 else 0.0
    tr = float(abs(np.trace(np.atleast_2d(H))))
    if amp <= 0 or tr <= 0:
        return np.inf, "OK"
    k = np.sqrt(tr / amp)
    lam = 2 * np.pi / k
    if h > lam / lam_defer:
        return lam, "UNDER-RESOLVED-DEFER"
    if h > lam / lam_caution:
        return lam, "UNDER-RESOLVED-CAUTION"
    return lam, "OK"


@dataclass
class PixelReport:
    index: Tuple[int, int]
    n_directions: int
    aperture_deg: float
    kappa: float
    span_min: float
    span_max: float
    H_hat: Optional[np.ndarray]
    C3_hat: Optional[float]
    C3_status: str
    antipodal_sampled: float
    expected_order: int
    error_estimate: Optional[float]
    rho: Optional[float]
    verdict: str
    flags: list

    def __str__(self):
        L = [f"pixel               {self.index}",
             f"directions          {self.n_directions}",
             f"aperture at h_ref   {self.aperture_deg:.0f} of 360 deg",
             f"kappa               {self.kappa:.3f}  "
             f"(full circle: {KAPPA_FULL_CIRCLE:.3f})",
             f"span min/max        {self.span_min:.4g} / {self.span_max:.4g}",
             f"antipodal sampled   {self.antipodal_sampled:.0%}",
             f"expected order      {self.expected_order}",
             f"C3                  " + ("not estimated" if self.C3_hat is None
                                        else f"{self.C3_hat:.4g}"),
             f"C3 status           {self.C3_status}",
             "predicted error     " + ("not available" if self.error_estimate is None
                                       else f"{self.error_estimate:.4g}"),
             "rho = err / |H|_F   " + ("not available" if self.rho is None
                                       else f"{self.rho:.3g}")]
        if self.H_hat is not None:
            L.append("Hessian:")
            for row in self.H_hat:
                L.append("   " + "  ".join(f"{v: .6g}" for v in row))
        for fl in self.flags:
            L.append(f"  ! {fl}")
        L.append(f"verdict             {self.verdict}")
        return "\n".join(L)


def measure_pixel(field: GridField, index, m=16, h_cap=None, sigma=0.0,
                  R=R_DEFAULT, c3_prior=None, on_c3_unavailable="defer",
                  h_floor=1.0, rho_caution=0.1, rho_defer=0.4) -> PixelReport:
    """Measure curvature, conditioning and model adequacy at one pixel."""
    index = (int(index[0]), int(index[1]))
    flags = []
    if not field.mask[index]:
        return PixelReport(index, 0, 0.0, np.inf, 0, 0, None, None,
                           "NOT-ATTEMPTED", 0.0, 0, None, None, "UNUSABLE",
                           ["pixel is outside the mask"])
    if not field.valid(np.asarray(index, float)):
        return PixelReport(index, 0, 0.0, np.inf, 0, 0, None, None,
                           "NOT-ATTEMPTED", 0.0, 0, None, None, "UNUSABLE",
                           ["the interpolation stencil at this pixel already "
                            "reaches outside the mask; no extrapolation is "
                            "performed"])

    dirs, spans = local_directions(field, index, m=m, h_cap=h_cap)
    ndir = len(dirs)
    h_ref = float(np.median(spans)) if ndir else 0.0
    ap = aperture_deg(field, index, h_ref, h_cap=h_cap) if ndir else 0.0
    if ndir < 3:
        return PixelReport(index, ndir, ap, np.inf, 0, 0, None, None,
                           "NOT-ATTEMPTED", 0.0, 0, None, None, "UNUSABLE",
                           [f"only {ndir} usable directions; 3 are required for "
                            "the 2D Hessian to be identifiable"])

    # --- pilot for C3, one COMMON span so the cubic term has one coefficient
    h_pilot = 0.5 * float(spans.min())
    if c3_prior is not None:
        c3, c3_status = float(c3_prior), "USER-SUPPLIED"
    elif h_pilot <= 0:
        c3, c3_status = None, "SPAN-LIMITED"
    else:
        try:
            q0 = np.array([probe(field, index, u, h_pilot, R) for u in dirs])
            c3, c3_status = estimate_C3(dirs, q0, h_pilot, sigma, R)
        except MaskSupportError:
            c3, c3_status = None, "SPAN-LIMITED"
        if c3 is None and c3_status == "NOISE-LIMITED" \
                and spans.max() / max(spans.min(), 1e-30) > 3.0:
            c3_status = "SPAN-LIMITED"

    c3_known = c3 is not None and c3 > 0
    if not c3_known and on_c3_unavailable == "require":
        raise ValueError(f"C3 unavailable ({c3_status}) at {index}")

    # --- working spans
    # A span below the grid resolution cannot be supported by the data, whatever
    # the balance rule says. The floor is tied to the spacing, NOT to |H|, which
    # would be circular since H is what is being estimated.
    h_min = h_floor * float(field.spacing.max())
    if c3_known:
        nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
        fscale = abs(field.at(index)) + 1.0
        sg = max(sigma, 1e-16 * fscale)
        Cs = 2.0 * nu / (R * (1 - R)) * sg
        C3c = (1 + R) / 3.0 * c3
        h_star = (2.0 * Cs / max(C3c, 1e-14)) ** (1 / 3)
        if h_star < h_min:
            # the optimum is not attainable on this grid; say so rather than
            # clipping silently and reporting the result as if it were optimal
            c3_status = "SPAN-LIMITED"
            flags.append(f"the balance span {h_star:.3g} is below the grid floor "
                         f"{h_min:.3g}; the noise-truncation optimum is not "
                         "attainable at this resolution")
        work = np.clip(np.minimum(spans, h_star), h_min, None)
        work = np.minimum(work, spans)
    else:
        work = spans.copy()
        flags.append(f"C3 unavailable ({c3_status}); spans fall back to the "
                     "largest usable and the truncation term is not controlled")

    try:
        q = np.array([probe(field, index, u, h, R) for u, h in zip(dirs, work)])
    except MaskSupportError as e:
        return PixelReport(index, ndir, ap, np.inf, float(work.min()),
                           float(work.max()), None, c3, c3_status,
                           antipodal_sampled_fraction(dirs), 0, "UNUSABLE",
                           flags + [str(e)])
    A = design_matrix(dirs)
    kappa = float(np.linalg.cond(A))
    if not np.isfinite(kappa) or kappa > 1e8:
        return PixelReport(index, ndir, ap, kappa, float(spans.min()),
                           float(spans.max()), None, c3, c3_status,
                           antipodal_sampled_fraction(dirs), 0, None, None,
                           "UNUSABLE",
                           flags + ["direction set is numerically singular: the "
                                    "usable directions do not span Sym_2"])

    if c3_known:
        nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
        w = np.array([1.0 / max(((1 + R) / 3.0 * h * c3) ** 2
                                + (2 * nu * sigma / (R * (1 - R) * h ** 2)) ** 2,
                                1e-30) for h in work])
    else:
        w = np.ones(len(work))
    W = np.diag(w)
    hv = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ q, rcond=None)[0]
    H = vec_to_H(hv)

    apf = antipodal_sampled_fraction(dirs)
    order = 2 if apf > 0.999 else 1

    if kappa > 10 * KAPPA_FULL_CIRCLE:
        flags.append(f"conditioning degraded {kappa/KAPPA_FULL_CIRCLE:.0f}x "
                     "relative to the full circle; the tangential component is "
                     "least reliable")
    if order == 1:
        flags.append("antipodal partners not sampled: first order only")
    if ap < 200:
        flags.append(f"only {ap:.0f} degrees of directions usable")

    # predicted error and its ratio to the recovered curvature
    hw = float(np.median(work))
    normH = float(np.linalg.norm(H, "fro"))
    if c3_known and normH > 0:
        err = predicted_error(kappa, c3, hw, sigma, R)
        rho = err / normH
    else:
        err, rho = None, None

    if rho is not None and rho > rho_caution:
        flags.append(f"predicted error is {rho:.2g} times the recovered "
                     "curvature")

    # geometry alone never returns GOOD
    if not c3_known and on_c3_unavailable == "defer":
        verdict = "DEFER"
    elif rho is None:
        verdict = "CAUTION"
    elif rho > rho_defer:
        verdict = "DEFER"
    elif rho > rho_caution or kappa > 5 * KAPPA_FULL_CIRCLE or order == 1 \
            or len(flags) >= 3:
        verdict = "CAUTION"
    else:
        verdict = "GOOD"

    return PixelReport(index, ndir, ap, kappa, float(work.min()),
                       float(work.max()), H, c3, c3_status, apf, order,
                       err, rho, verdict, flags)


# =====================================================================
# Batch maps
# =====================================================================

VERDICT_CODE = {"UNUSABLE": 0, "DEFER": 1, "CAUTION": 2, "GOOD": 3}
STATUS_CODE = {"NOT-ATTEMPTED": 0, "UNDER-DETERMINED": 1, "SPAN-LIMITED": 2,
               "NOISE-LIMITED": 3, "USER-SUPPLIED": 4, "VALID": 5}


def reliability_maps(field: GridField, m=16, h_cap=None, sigma=0.0,
                     step=1, R=R_DEFAULT, progress=False) -> dict:
    """Per-pixel maps over the whole valid region.

    Returns a dict of 2D arrays: aperture, kappa, order, c3, c3_status,
    verdict, H11, H22, H12, span_min, span_max. Invalid pixels are NaN
    (or 0 in the integer-coded maps).
    """
    ny, nx = field.ny, field.nx
    out = {k: np.full((ny, nx), np.nan) for k in
           ("aperture", "kappa", "c3", "H11", "H22", "H12",
            "span_min", "span_max", "rho")}
    out["order"] = np.zeros((ny, nx), np.int8)
    out["verdict"] = np.zeros((ny, nx), np.int8)
    out["c3_status"] = np.zeros((ny, nx), np.int8)
    total = int(field.mask[::step, ::step].sum())
    done = 0
    for iy in range(0, ny, step):
        for ix in range(0, nx, step):
            if not field.mask[iy, ix]:
                continue
            r = measure_pixel(field, (iy, ix), m=m, h_cap=h_cap,
                              sigma=sigma, R=R)
            out["aperture"][iy, ix] = r.aperture_deg
            out["kappa"][iy, ix] = r.kappa
            out["order"][iy, ix] = r.expected_order
            out["verdict"][iy, ix] = VERDICT_CODE[r.verdict]
            out["c3_status"][iy, ix] = STATUS_CODE[r.C3_status]
            out["span_min"][iy, ix] = r.span_min
            out["span_max"][iy, ix] = r.span_max
            if r.rho is not None:
                out["rho"][iy, ix] = r.rho
            if r.C3_hat is not None:
                out["c3"][iy, ix] = r.C3_hat
            if r.H_hat is not None:
                out["H11"][iy, ix] = r.H_hat[0, 0]
                out["H22"][iy, ix] = r.H_hat[1, 1]
                out["H12"][iy, ix] = r.H_hat[0, 1]
            done += 1
            if progress and done % 500 == 0:
                print(f"  {done}/{total} pixels", flush=True)
    return out


# =====================================================================
# Self-test
# =====================================================================

def _self_test():
    print("=" * 70)
    print("SELF-TEST: exact quadratic on a full grid must be recovered exactly")
    ny = nx = 81
    sp = 0.02
    yy, xx = np.mgrid[0:ny, 0:nx]
    X = (xx - nx // 2) * sp
    Y = (yy - ny // 2) * sp
    Htrue = np.array([[3.0, -1.0], [-1.0, 2.0]])
    Fq = 0.5 * (Htrue[0, 0] * X ** 2 + 2 * Htrue[0, 1] * X * Y
                + Htrue[1, 1] * Y ** 2) + 0.7 * X - 0.4 * Y + 1.3
    # NOTE: array is indexed [y, x]; direction u = (u_x, u_y) maps to an index
    # offset of (u_y, u_x)/spacing, handled by to_index on (dy, dx).
    g = GridField(Fq, spacing=sp)
    r = measure_pixel(g, (ny // 2, nx // 2), m=16)
    print(f"  recovered H =\n{r.H_hat}")
    print(f"  true H      =\n{Htrue}")
    print(f"  max error   = {np.abs(r.H_hat - Htrue).max():.2e}")
    print(f"  kappa       = {r.kappa:.4f}   (full circle {KAPPA_FULL_CIRCLE:.4f})")
    print(f"  order       = {r.expected_order}   antipodal {r.antipodal_sampled:.0%}")

    print("\nSELF-TEST: masked disc with a notch, geometry from the mask")
    R0 = 0.34 * nx * sp
    rad = np.hypot(X, Y)
    mask = rad < R0
    mask &= ~((np.abs(Y) < 0.12) & (X > 0.0))          # a notch cut inward
    F = np.exp(X) * np.cos(2 * Y) + 0.3 * X ** 3
    g2 = GridField(F, spacing=sp, mask=mask)
    cy, cx = ny // 2, nx // 2
    for lab, idx in (("deep interior  ", (cy - 6, cx - 6)),
                     ("near the rim   ", (cy, cx - 24)),
                     ("beside the notch", (cy + 8, cx + 14))):
        if not g2.mask[idx]:
            print(f"  {lab}: masked out"); continue
        r = measure_pixel(g2, idx, m=16, sigma=1e-6)
        print(f"  {lab}: aperture {r.aperture_deg:>5.0f} deg  "
              f"kappa {r.kappa:>7.2f}  order {r.expected_order}  "
              f"dirs {r.n_directions:>2}  C3 {r.C3_status:<16} {r.verdict}")

    print("\nSELF-TEST: full reliability maps on the masked disc (step 4)")
    maps = reliability_maps(g2, m=12, sigma=1e-6, step=4)
    v = maps["verdict"]
    names = {c: n for n, c in VERDICT_CODE.items()}
    tot = int((v > 0).sum())
    print(f"  measured pixels: {tot}")
    for c in sorted(set(v[v > 0].ravel().tolist())):
        print(f"    {names[c]:<10} {int((v == c).sum()):>5} "
              f"({100*int((v==c).sum())/max(tot,1):.0f}%)")
    good = v > VERDICT_CODE["UNUSABLE"]
    ap = maps["aperture"][good]
    print(f"  aperture range over scored pixels: {np.nanmin(ap):.0f} to "
          f"{np.nanmax(ap):.0f} deg")
    kk = maps["kappa"][good & np.isfinite(maps["kappa"])]
    print(f"  kappa over scored pixels: {kk.min():.2f} to {kk.max():.2f}")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()
