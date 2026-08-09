#!/usr/bin/env python3
"""
grid3d.py
=========
Three-dimensional directional curvature measurement on **measured volume data**.

The 3D counterpart of `grid2d`. It takes a 3D array of samples, a voxel spacing
and an optional validity mask, and returns per-voxel

    H_hat     the recovered 3x3 Hessian (six components)
    kappa     conditioning of the locally available direction geometry
    C3_hat    third-order magnitude, or an explicit reason it is unavailable
    aperture  the usable solid angle at that voxel, as a percentage of 4 pi
    order     attainable order, 1 or 2, from SAMPLED antipodal partners
    status    why C3 is or is not available
    verdict   GOOD / MODERATE / LOW / DEFER / UNUSABLE

Two differences from 2D that matter in practice.

The interpolation support is a 4x4x4 block of 64 voxels rather than 4x4 of 16,
so the mask requirement is much stricter and coverage falls faster near
boundaries. This is measured, not assumed: see `coverage_fraction`.

The Hessian has six unknowns rather than three, so at least six directions are
needed, and second-order accuracy needs them in antipodal pairs, hence twelve.

Conventions, fixed and not varied:
    C3(x) = [ mean_u |D^3 f(x)[u,u,u]|^2 ]^{1/2}   over the sampled directions
    ||H||  = ||H||_F
    kappa computed in the Frobenius-consistent basis
           (H11, H22, H33, sqrt2 H12, sqrt2 H13, sqrt2 H23)

Reference: R. Pasupuleti, "Geometry, conditioning, and limits of one-sided
directional curvature recovery".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["VolumeField", "MaskSupportError3D", "predicted_error_3d",
           "resolution_guard", "local_directions_3d",
           "measure_voxel", "reliability_volumes", "coverage_fraction",
           "solid_angle_pct", "R_DEFAULT", "KAPPA_FULL_SPHERE"]

R_DEFAULT = 0.5
KAPPA_FULL_SPHERE = 0.5 * np.sqrt(10)      # icosahedral six-axis reference
P_ODD_3D = 10                              # dim H_1 + dim H_3 on the sphere
_DMIN = 6                                  # dim Sym_3


class MaskSupportError3D(ValueError):
    """Raised when a tricubic stencil would need data outside the valid mask."""


# =====================================================================
# Volume field
# =====================================================================

class VolumeField:
    """A measured 3D field on a regular grid, with tricubic access.

    Parameters
    ----------
    array : (nz, ny, nx) float
        Sample values, indexed [z, y, x]. Complex input is refused: the probe
        expansion is linear in the field, so it applies component-wise to the
        real and imaginary parts of a complex phasor, but NOT to the amplitude
        |u|, which is a nonlinear transformation the expansion does not cover.
    spacing : float or (3,)
        Voxel size in physical units, as (dz, dy, dx). Scalar means isotropic.
    mask : (nz, ny, nx) bool, optional
        True where the sample is valid.
    """

    def __init__(self, array, spacing=1.0, mask=None):
        if np.iscomplexobj(array):
            raise TypeError("complex input: apply component-wise to the real "
                            "and imaginary parts; the amplitude is nonlinear "
                            "and outside the theory")
        a = np.asarray(array, float)
        if a.ndim != 3:
            raise ValueError("array must be 3D, indexed [z, y, x]")
        self.a = a
        self.nz, self.ny, self.nx = a.shape
        sp = np.atleast_1d(np.asarray(spacing, float))
        self.spacing = np.array([sp[0]] * 3) if sp.size == 1 else np.asarray(sp, float)
        if self.spacing.size != 3:
            raise ValueError("spacing must be scalar or length 3, as (dz, dy, dx)")
        self.mask = (np.ones_like(a, bool) if mask is None
                     else np.asarray(mask, bool))
        if self.mask.shape != a.shape:
            raise ValueError("mask shape must match array shape")

    # -- coordinates --------------------------------------------------
    def to_index(self, offset_phys):
        """Physical offset (dx, dy, dz) -> fractional index offset (dz, dy, dx)."""
        dx, dy, dz = np.asarray(offset_phys, float)
        return np.array([dz, dy, dx]) / self.spacing

    @staticmethod
    def _crom(t):
        """Catmull-Rom weights for samples at -1, 0, 1, 2. Reproduces cubics
        exactly, so neither the quadratic model nor the O(h) cubic term of the
        probe expansion is corrupted by interpolation. Trilinear would not:
        it reproduces only trilinear functions, and its error is of the same
        size as the curvature being measured."""
        t2, t3 = t * t, t * t * t
        return np.array([-0.5 * t3 + t2 - 0.5 * t,
                         1.5 * t3 - 2.5 * t2 + 1.0,
                         -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                         0.5 * t3 - 0.5 * t2])

    def _support(self, idx):
        i0 = np.floor(idx).astype(int)
        rng = [np.arange(i0[k] - 1, i0[k] + 3) for k in range(3)]
        return i0, rng

    def valid(self, idx) -> bool:
        """Is the whole 4x4x4 tricubic support inside the grid and the mask?"""
        i0, rng = self._support(np.asarray(idx, float))
        for k, n in enumerate((self.nz, self.ny, self.nx)):
            if rng[k].min() < 0 or rng[k].max() > n - 1:
                return False
        return bool(self.mask[np.ix_(*rng)].all())

    def at(self, idx, strict=False) -> float:
        """Tricubic (Catmull-Rom) sample at a fractional index.

        With strict=True the 4x4x4 support must lie inside the grid and inside
        the mask, and MaskSupportError3D is raised otherwise. Never extrapolate
        across invalid data: interpolating over masked tissue would fabricate a
        curvature signal from the mask edge.
        """
        idx = np.asarray(idx, float)
        if strict and not self.valid(idx):
            raise MaskSupportError3D(f"tricubic support at {idx} leaves the mask")
        i0, rng = self._support(idx)
        f = idx - i0
        w = [self._crom(f[k]) for k in range(3)]
        rng = [np.clip(r, 0, n - 1)
               for r, n in zip(rng, (self.nz, self.ny, self.nx))]
        blk = self.a[np.ix_(*rng)]
        return float(np.einsum("i,j,k,ijk->", w[0], w[1], w[2], blk))

    def sample(self, centre_idx, offset_phys, strict=False) -> float:
        return self.at(np.asarray(centre_idx, float) + self.to_index(offset_phys),
                       strict=strict)


# =====================================================================
# Geometry from the mask
# =====================================================================

def hmax_along_3d(field: VolumeField, centre_idx, u, h_cap, n_steps=32) -> float:
    """Largest span along u for which the whole ray keeps a valid support."""
    step = float(h_cap) / n_steps
    good, t = 0.0, step
    c = np.asarray(centre_idx, float)
    while t <= h_cap + 1e-12:
        if not field.valid(c + field.to_index(t * np.asarray(u))):
            break
        good = t
        t += step
    return good


def _icosahedral_axes():
    p = (1 + np.sqrt(5)) / 2
    V = np.array([[0, 1, p], [0, 1, -p], [1, p, 0],
                  [1, -p, 0], [p, 0, 1], [-p, 0, 1]], float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def _fibonacci_hemisphere(m):
    i = np.arange(m) + 0.5
    ph = np.arccos(1 - i / m)             # upper hemisphere only
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)]


def local_directions_3d(field: VolumeField, centre_idx, m=24, h_cap=None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Directions usable at this voxel, in antipodal pairs, with their spans.

    Pairs are the point: the odd term of the probe expansion cancels only
    between directions that are actually probed, so a set whose antipodes are
    merely feasible remains first order. A pair is kept only when both members
    are usable, which is why the count can fall below the six that identifiability
    requires even where the mask looks generous.
    """
    if h_cap is None:
        h_cap = 0.2 * min(field.nz, field.ny, field.nx) * float(field.spacing.min())
    half = max(3, m // 2)
    base = _icosahedral_axes() if half == 6 else _fibonacci_hemisphere(half)
    hp = np.array([hmax_along_3d(field, centre_idx, u, h_cap) for u in base])
    hm = np.array([hmax_along_3d(field, centre_idx, -u, h_cap) for u in base])
    both = np.minimum(hp, hm)
    keep = both > 0
    dirs = np.vstack([base[keep], -base[keep]])
    spans = np.concatenate([both[keep], both[keep]])
    return dirs, spans


def solid_angle_pct(field: VolumeField, centre_idx, h_ref, h_cap=None, m=64
                    ) -> float:
    """Percentage of the full solid angle usable AT THE SPAN h_ref.

    The span matters: almost every direction admits some nonzero span, so
    counting directions with h_max > 0 reports 100 percent nearly everywhere and
    says nothing. The set that enters the conditioning law is the one usable at
    the span actually probed.
    """
    if h_cap is None:
        h_cap = 0.2 * min(field.nz, field.ny, field.nx) * float(field.spacing.min())
    U = _fibonacci_hemisphere(m // 2)
    U = np.vstack([U, -U])
    ok = [hmax_along_3d(field, centre_idx, u, h_cap) >= h_ref - 1e-12 for u in U]
    return 100.0 * float(np.mean(ok))


def coverage_fraction(field: VolumeField, step=1) -> float:
    """Fraction of masked-in voxels whose 4x4x4 support is complete.

    Report this before trusting anything else: tricubic needs 64 valid
    neighbours, so a mask with scattered exclusions can leave almost no voxel
    measurable.
    """
    tot = ok = 0
    for iz in range(0, field.nz, step):
        for iy in range(0, field.ny, step):
            for ix in range(0, field.nx, step):
                if not field.mask[iz, iy, ix]:
                    continue
                tot += 1
                if field.valid(np.array([iz, iy, ix], float)):
                    ok += 1
    return ok / max(tot, 1)


# =====================================================================
# Core measurement
# =====================================================================

def design_matrix_3d(dirs) -> np.ndarray:
    """Frobenius-consistent columns for (H11,H22,H33,sqrt2 H12,sqrt2 H13,sqrt2 H23).
    Note the index order: u = (ux, uy, uz) with x the fastest array axis."""
    U = np.atleast_2d(np.asarray(dirs, float))
    s = np.sqrt(2)
    return np.c_[U[:, 0] ** 2, U[:, 1] ** 2, U[:, 2] ** 2,
                 s * U[:, 0] * U[:, 1], s * U[:, 0] * U[:, 2],
                 s * U[:, 1] * U[:, 2]]


def vec_to_H3(h) -> np.ndarray:
    s = np.sqrt(2)
    return np.array([[h[0], h[3] / s, h[4] / s],
                     [h[3] / s, h[1], h[5] / s],
                     [h[4] / s, h[5] / s, h[2]]])


def probe3(field: VolumeField, centre_idx, u, h, R=R_DEFAULT, strict=True) -> float:
    """One-sided directional probe; three samples on [x, x + h u]."""
    f0 = field.at(np.asarray(centre_idx, float), strict=strict)
    f1 = field.sample(centre_idx, h * np.asarray(u), strict=strict)
    fR = field.sample(centre_idx, R * h * np.asarray(u), strict=strict)
    return 2.0 * ((1 - R) * f0 + R * f1 - fR) / (R * (1 - R) * h * h)


def antipodal_sampled_fraction_3d(dirs, tol=1e-8) -> float:
    D = np.atleast_2d(np.asarray(dirs, float))
    return float(np.mean([np.min(np.linalg.norm(D + u, axis=1)) < tol for u in D]))


def _poly_basis_3d(U, lmax=3):
    """Monomials by total degree; 1 + 3 + 6 + 10 = 20 columns at lmax = 3."""
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    cols, degs = [np.ones(len(U))], [0]
    for d in range(1, lmax + 1):
        for i in range(d + 1):
            for j in range(d - i + 1):
                k = d - i - j
                cols.append(x ** i * y ** j * z ** k)
                degs.append(d)
    return np.column_stack(cols), np.array(degs)


def estimate_C3_3d(dirs, q, h, sigma=0.0, R=R_DEFAULT, ridge=1e-3):
    """Odd-degree content of the probe field estimates ||D^3 f||.
    Returns (value, status). Needs more than 20 directions to be determined."""
    B, degs = _poly_basis_3d(np.atleast_2d(dirs), 3)
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
        corr = odd ** 2 - P_ODD_3D * sq ** 2 / len(q)
        if corr <= 0:
            return None, "NOISE-LIMITED"
        odd = np.sqrt(corr)
    return max(3.0 * odd / ((1 + R) * h), 0.0), "VALID"


def predicted_error_3d(kappa, c3, h, sigma, R=R_DEFAULT):
    """kappa * ( (1+R)/3 C3 h  +  2 sigma nu / (R(1-R) h^2) ): truncation plus
    noise, amplified by the conditioning of the direction geometry.

    The thresholds are calibrated, not chosen. rho systematically UNDERSTATES the
    true relative error, because its denominator ||H_hat||_F is itself inflated by
    noise. Measured on a smooth 2D field: rho 0.23 against a true relative error
    of 0.15, rho 0.46 against 0.57, rho 0.80 against 1.95, rho 0.99 against 5.88.
    rho_defer is therefore set below 1.
    """
    nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
    return float(kappa * ((1 + R) / 3.0 * c3 * h
                          + 2 * sigma * nu / (R * (1 - R) * max(h, 1e-30) ** 2)))



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
class VoxelReport:
    index: Tuple[int, int, int]
    n_directions: int
    solid_angle_pct: float
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
        L = [f"voxel                {self.index}",
             f"directions           {self.n_directions}",
             f"solid angle at h_ref {self.solid_angle_pct:.0f}% of 4 pi",
             f"kappa                {self.kappa:.3f}  "
             f"(full sphere: {KAPPA_FULL_SPHERE:.3f})",
             f"span min/max         {self.span_min:.4g} / {self.span_max:.4g}",
             f"antipodal sampled    {self.antipodal_sampled:.0%}",
             f"expected order       {self.expected_order}",
             "C3                   " + ("not estimated" if self.C3_hat is None
                                        else f"{self.C3_hat:.4g}"),
             f"C3 status            {self.C3_status}",
             "predicted error      " + ("not available" if self.error_estimate is None
                                        else f"{self.error_estimate:.4g}"),
             "rho = err / |H|_F    " + ("not available" if self.rho is None
                                        else f"{self.rho:.3g}")]
        if self.H_hat is not None:
            L.append("Hessian:")
            for row in self.H_hat:
                L.append("   " + "  ".join(f"{v: .6g}" for v in row))
        for fl in self.flags:
            L.append(f"  ! {fl}")
        L.append(f"verdict              {self.verdict}")
        return "\n".join(L)


def _unusable(index, ndir, sa, msg, c3=None, status="NOT-ATTEMPTED", flags=None):
    return VoxelReport(index, ndir, sa, np.inf, 0.0, 0.0, None, c3, status,
                       0.0, 0, None, None, "UNUSABLE", (flags or []) + [msg])


def measure_voxel(field: VolumeField, index, m=24, h_cap=None, sigma=0.0,
                  R=R_DEFAULT, c3_prior=None, on_c3_unavailable="defer",
                  h_floor=1.0, rho_caution=0.1, rho_defer=0.4) -> VoxelReport:
    """Measure curvature, conditioning and model adequacy at one voxel."""
    index = tuple(int(v) for v in index)
    flags = []
    if not field.mask[index]:
        return _unusable(index, 0, 0.0, "voxel is outside the mask")
    if not field.valid(np.asarray(index, float)):
        return _unusable(index, 0, 0.0,
                         "the tricubic stencil at this voxel already reaches "
                         "outside the mask; no extrapolation is performed")

    dirs, spans = local_directions_3d(field, index, m=m, h_cap=h_cap)
    ndir = len(dirs)
    if ndir < _DMIN:
        return _unusable(index, ndir, 0.0,
                         f"only {ndir} usable directions in antipodal pairs; "
                         f"{_DMIN} are required for the 3D Hessian to be "
                         "identifiable")
    h_ref = float(np.median(spans))
    sa = solid_angle_pct(field, index, h_ref, h_cap=h_cap)

    # --- pilot for C3, one COMMON span so the cubic term has one coefficient
    h_pilot = 0.5 * float(spans.min())
    if c3_prior is not None:
        c3, c3_status = float(c3_prior), "USER-SUPPLIED"
    elif h_pilot <= 0:
        c3, c3_status = None, "SPAN-LIMITED"
    else:
        try:
            q0 = np.array([probe3(field, index, u, h_pilot, R) for u in dirs])
            c3, c3_status = estimate_C3_3d(dirs, q0, h_pilot, sigma, R)
        except MaskSupportError3D:
            c3, c3_status = None, "SPAN-LIMITED"
        if c3 is None and c3_status == "NOISE-LIMITED" \
                and spans.max() / max(spans.min(), 1e-30) > 3.0:
            c3_status = "SPAN-LIMITED"

    c3_known = c3 is not None and c3 > 0
    if not c3_known and on_c3_unavailable == "require":
        raise ValueError(f"C3 unavailable ({c3_status}) at {index}")

    # A span below the voxel resolution cannot be supported by the data. The
    # floor is tied to the spacing, NOT to |H|, which would be circular.
    h_min = h_floor * float(field.spacing.max())
    if c3_known:
        nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
        fscale = abs(field.at(np.asarray(index, float))) + 1.0
        sg = max(sigma, 1e-16 * fscale)
        h_star = (2.0 * (2.0 * nu / (R * (1 - R)) * sg)
                  / max((1 + R) / 3.0 * c3, 1e-14)) ** (1 / 3)
        if h_star < h_min:
            c3_status = "SPAN-LIMITED"
            flags.append(f"the balance span {h_star:.3g} is below the voxel floor "
                         f"{h_min:.3g}; the noise-truncation optimum is not "
                         "attainable at this resolution")
        work = np.minimum(np.clip(np.minimum(spans, h_star), h_min, None), spans)
    else:
        work = spans.copy()
        flags.append(f"C3 unavailable ({c3_status}); spans fall back to the "
                     "largest usable and the truncation term is not controlled")

    try:
        q = np.array([probe3(field, index, u, h, R) for u, h in zip(dirs, work)])
    except MaskSupportError3D as e:
        return _unusable(index, ndir, sa, str(e), c3, c3_status, flags)

    A = design_matrix_3d(dirs)
    kappa = float(np.linalg.cond(A))
    if not np.isfinite(kappa) or kappa > 1e8:
        return _unusable(index, ndir, sa,
                         "direction set is numerically singular: the usable "
                         "directions do not span Sym_3", c3, c3_status, flags)

    if c3_known:
        nu = np.sqrt((1 - R) ** 2 + R ** 2 + 1.0)
        w = np.array([1.0 / max(((1 + R) / 3.0 * h * c3) ** 2
                                + (2 * nu * sigma / (R * (1 - R) * h ** 2)) ** 2,
                                1e-30) for h in work])
    else:
        w = np.ones(len(work))
    W = np.diag(w)
    H = vec_to_H3(np.linalg.lstsq(A.T @ W @ A, A.T @ W @ q, rcond=None)[0])

    apf = antipodal_sampled_fraction_3d(dirs)
    order = 2 if apf > 0.999 else 1

    if kappa > 10 * KAPPA_FULL_SPHERE:
        flags.append(f"conditioning degraded {kappa/KAPPA_FULL_SPHERE:.0f}x "
                     "relative to the full sphere; the tangential block is "
                     "least reliable")
    if order == 1:
        flags.append("antipodal partners not sampled: first order only")
    if sa < 60:
        flags.append(f"only {sa:.0f}% of the solid angle usable at this span")

    hw = float(np.median(work))
    normH = float(np.linalg.norm(H, "fro"))
    if c3_known and normH > 0:
        err = predicted_error_3d(kappa, c3, hw, sigma, R)
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
    elif rho > rho_caution or kappa > 5 * KAPPA_FULL_SPHERE or order == 1 \
            or len(flags) >= 3:
        verdict = "CAUTION"
    else:
        verdict = "GOOD"

    return VoxelReport(index, ndir, sa, kappa, float(work.min()),
                       float(work.max()), H, c3, c3_status, apf, order,
                       err, rho, verdict, flags)


# =====================================================================
# Batch volumes
# =====================================================================

VERDICT_CODE = {"UNUSABLE": 0, "DEFER": 1, "CAUTION": 2, "GOOD": 3}
STATUS_CODE = {"NOT-ATTEMPTED": 0, "UNDER-DETERMINED": 1, "SPAN-LIMITED": 2,
               "NOISE-LIMITED": 3, "USER-SUPPLIED": 4, "VALID": 5}


def reliability_volumes(field: VolumeField, m=24, h_cap=None, sigma=0.0,
                        step=2, R=R_DEFAULT, progress=False) -> dict:
    """Per-voxel volumes over the valid region.

    `step` subsamples; 3D measurement is expensive because each voxel
    ray-marches every direction. Start with step=4 on a real volume, inspect,
    then refine.
    """
    shp = (field.nz, field.ny, field.nx)
    out = {k: np.full(shp, np.nan) for k in
           ("solid_angle", "kappa", "c3", "H11", "H22", "H33",
            "H12", "H13", "H23", "span_min", "span_max", "rho")}
    for k in ("order", "verdict", "c3_status"):
        out[k] = np.zeros(shp, np.int8)
    done = 0
    for iz in range(0, field.nz, step):
        for iy in range(0, field.ny, step):
            for ix in range(0, field.nx, step):
                if not field.mask[iz, iy, ix]:
                    continue
                r = measure_voxel(field, (iz, iy, ix), m=m, h_cap=h_cap,
                                  sigma=sigma, R=R)
                out["solid_angle"][iz, iy, ix] = r.solid_angle_pct
                out["kappa"][iz, iy, ix] = r.kappa
                out["order"][iz, iy, ix] = r.expected_order
                out["verdict"][iz, iy, ix] = VERDICT_CODE[r.verdict]
                out["c3_status"][iz, iy, ix] = STATUS_CODE[r.C3_status]
                out["span_min"][iz, iy, ix] = r.span_min
                out["span_max"][iz, iy, ix] = r.span_max
                if r.rho is not None:
                    out["rho"][iz, iy, ix] = r.rho
                if r.C3_hat is not None:
                    out["c3"][iz, iy, ix] = r.C3_hat
                if r.H_hat is not None:
                    H = r.H_hat
                    for key, v in (("H11", H[0, 0]), ("H22", H[1, 1]),
                                   ("H33", H[2, 2]), ("H12", H[0, 1]),
                                   ("H13", H[0, 2]), ("H23", H[1, 2])):
                        out[key][iz, iy, ix] = v
                done += 1
                if progress and done % 200 == 0:
                    print(f"  {done} voxels", flush=True)
    return out


# =====================================================================
# Self-test
# =====================================================================

def _self_test():
    print("=" * 72)
    print("3D SELF-TEST")
    n, sp = 33, 0.03
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    X = (xx - n // 2) * sp
    Y = (yy - n // 2) * sp
    Z = (zz - n // 2) * sp

    print("\n1. exact quadratic must be recovered to machine precision")
    Ht = np.array([[3.0, -1.0, 0.5], [-1.0, 2.0, 0.2], [0.5, 0.2, 1.5]])
    Fq = 0.5 * (Ht[0, 0] * X ** 2 + Ht[1, 1] * Y ** 2 + Ht[2, 2] * Z ** 2
                + 2 * Ht[0, 1] * X * Y + 2 * Ht[0, 2] * X * Z
                + 2 * Ht[1, 2] * Y * Z) + 0.7 * X - 0.4 * Y + 0.2 * Z + 1.3
    v = VolumeField(Fq, spacing=sp)
    r = measure_voxel(v, (n // 2, n // 2, n // 2), m=12)
    print(f"   max |H_hat - H| = {np.abs(r.H_hat - Ht).max():.2e}")
    print(f"   kappa = {r.kappa:.4f}  (full sphere {KAPPA_FULL_SPHERE:.4f})"
          f"   order {r.expected_order}   antipodal {r.antipodal_sampled:.0%}")

    print("\n2. tricubic support: coverage against the mask")
    rad = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    masks = {
        "full grid": np.ones_like(X, bool),
        "half space": X > -0.05,
        "sphere": rad < 0.30,
        "sphere with notch": (rad < 0.30) & ~((np.abs(Y) < 0.08) & (X > 0)),
        "slab, 5 voxels": np.abs(Z) < 2.5 * sp,
        "10% dropout": np.random.default_rng(1).random(X.shape) > 0.10,
    }
    print(f"   {'mask':<20}{'coverage':>10}")
    for nm, mk in masks.items():
        cf = coverage_fraction(VolumeField(Fq, spacing=sp, mask=mk), step=2)
        print(f"   {nm:<20}{100*cf:>9.1f}%")

    print("\n3. no extrapolation: edge voxels must be refused")
    mk = rad < 0.24
    v2 = VolumeField(Fq, spacing=sp, mask=mk)
    edge = [(iz, iy, ix) for iz in range(n) for iy in range(n) for ix in range(n)
            if mk[iz, iy, ix] and not v2.valid(np.array([iz, iy, ix], float))]
    sample = edge[::max(1, len(edge) // 12)]
    refused = sum(measure_voxel(v2, e, m=12).verdict == "UNUSABLE"
                  for e in sample)
    print(f"   edge voxels found {len(edge)}, sampled {len(sample)}, "
          f"refused {refused}")

    print("\n4. a real field inside a masked sphere")
    F = np.exp(X + 0.5 * Y) * np.cos(2 * Z) + 0.2 * X ** 3
    def Hf(x, y, z):
        e = np.exp(x + 0.5 * y); c, s = np.cos(2 * z), np.sin(2 * z)
        return np.array([[e*c + 1.2*x, 0.5*e*c, -2*e*s],
                         [0.5*e*c, 0.25*e*c, -1.0*e*s],
                         [-2*e*s, -1.0*e*s, -4*e*c]])
    v3 = VolumeField(F, spacing=sp, mask=(rad < 0.30))
    c = n // 2
    print(f"   {'voxel':<16}{'solid%':>8}{'relH':>10}{'kappa':>8}"
          f"{'C3 status':>17}{'ord':>4}{'verdict':>10}")
    for lab, idx in (("centre", (c, c, c)),
                     ("off centre", (c, c, c + 5)),
                     ("near the rim", (c, c, c + 8))):
        rr = measure_voxel(v3, idx, m=12, sigma=1e-6)
        if rr.H_hat is None:
            print(f"   {lab:<16}{rr.solid_angle_pct:>8.0f}{'n/a':>10}"
                  f"{'inf':>8}{rr.C3_status:>17}{rr.expected_order:>4}"
                  f"{rr.verdict:>10}")
        else:
            Ht2 = Hf(X[idx], Y[idx], Z[idx])
            rel = np.linalg.norm(rr.H_hat - Ht2, "fro") / np.linalg.norm(Ht2, "fro")
            print(f"   {lab:<16}{rr.solid_angle_pct:>8.0f}{rel:>10.2e}"
                  f"{rr.kappa:>8.2f}{rr.C3_status:>17}{rr.expected_order:>4}"
                  f"{rr.verdict:>10}")
    print("=" * 72)


if __name__ == "__main__":
    _self_test()
