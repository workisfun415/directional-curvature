"""Final regression suite for the reliability layer.

Each test pins a required behaviour, not merely that the code runs. Two
corrections are locked in here:

  span floor   the working span is tied to the grid, so it cannot collapse to
               the round-off optimum computed against the wrong scale; when the
               balance span is unattainable the status is SPAN-LIMITED rather
               than a silent clip reported as optimal.

  error-aware  the verdict combines geometry with the predicted error
  verdict      rho = kappa (C3 h + sigma nu / h^2) / ||H||_F. Geometry alone
               never returns GOOD.

A resolution guard was prototyped and REMOVED: the failure that motivated it was
a pathological evaluation point, not a defect. `resolution_guard` remains in the
modules as an unvalidated experimental diagnostic and is not called. The
consequence is a documented limitation, pinned by
`test_underresolution_is_a_known_limitation` below.
"""
import numpy as np
import pytest
from dircurv import GridField, measure_pixel, VolumeField, measure_voxel
from dircurv.analytic import (Geometry, icosahedral_axes, design_matrix,
                              vec_to_H, probe, antipodal_sampled_fraction)

N, SP = 81, 0.02
_yg, _xg = np.mgrid[0:N, 0:N]
X, Y = (_xg - N // 2) * SP, (_yg - N // 2) * SP
C = (N // 2, N // 2)
SMOOTH = np.exp(X) * np.cos(2 * Y) + 0.3 * X ** 3


def _noisy(F, sg, seed=5):
    if sg == 0:
        return F
    rng = np.random.default_rng(seed)
    return F + rng.normal(0, sg * (np.abs(F).max() + 1), F.shape)


# ---------------------------------------------------------------- exactness
def test_exact_quadratic_is_good():
    H = np.array([[3.0, -1.0], [-1.0, 2.0]])
    F = 0.5 * (H[0, 0] * X ** 2 + 2 * H[0, 1] * X * Y + H[1, 1] * Y ** 2) + 0.7 * X
    r = measure_pixel(GridField(F, spacing=SP), C, m=16)
    assert np.abs(r.H_hat - H).max() < 1e-10
    assert r.rho < 1e-8
    assert r.verdict == "GOOD"


# ------------------------------------------------------------- noise ladder
@pytest.mark.parametrize("sigma,expected", [(1e-6, "GOOD"), (1e-4, "GOOD"),
                                            (1e-3, "CAUTION"), (1e-2, "DEFER"),
                                            (1e-1, "DEFER")])
def test_verdict_degrades_with_noise(sigma, expected):
    r = measure_pixel(GridField(_noisy(SMOOTH, sigma), spacing=SP), C,
                      m=16, sigma=sigma)
    assert r.verdict == expected


def test_rho_increases_monotonically_with_noise():
    rhos = []
    for sg in (1e-6, 1e-5, 1e-4, 1e-3):
        r = measure_pixel(GridField(_noisy(SMOOTH, sg), spacing=SP), C,
                          m=16, sigma=sg)
        rhos.append(r.rho)
    assert all(b > a for a, b in zip(rhos, rhos[1:])), rhos


# --------------------------------------------------------------- smoothness
def test_kink_downgrades_from_good_to_caution():
    """As a kink approaches, C3 rises and the verdict must react."""
    far = 0.30
    near = 0.04
    out = {}
    for d in (far, near):
        F = np.exp(X) * np.cos(Y) + 2.0 * np.where(X > d, (X - d) ** 2, 0.0)
        out[d] = measure_pixel(GridField(F, spacing=SP), C, m=16, sigma=1e-6)
    assert out[far].verdict == "GOOD"
    assert out[near].verdict == "CAUTION"
    assert out[near].C3_hat > 3 * out[far].C3_hat


# ------------------------------------------------------------ realistic MRE
def test_realistic_mre_is_never_good():
    """50 Hz, 2 mm voxels, 20 um amplitude. This is the acceptance test: on data
    of the kind a collaborator would actually send, the module must not report
    GOOD."""
    m, sp = 41, 0.002
    zz, yy, xx = np.mgrid[0:m, 0:m, 0:m]
    Z, Yv, Xv = (zz - m // 2) * sp, (yy - m // 2) * sp, (xx - m // 2) * sp
    k, amp = 2 * np.pi / 0.020, 20e-6
    U = amp * np.sin(k * Z) * np.exp(-(Xv ** 2 + Yv ** 2) / (2 * 0.02 ** 2))
    brain = np.sqrt(Xv ** 2 + Yv ** 2 + Z ** 2) < 0.032
    for sg in (1e-3, 1e-2, 1e-1):
        rng = np.random.default_rng(2)
        v = VolumeField(U + rng.normal(0, sg * amp, U.shape), spacing=sp,
                        mask=brain)
        r = measure_voxel(v, (m // 2, m // 2, m // 2), m=24, sigma=sg * amp)
        assert r.verdict != "GOOD", (sg, r.verdict, r.rho)


# --------------------------------------------------------------- span floor
def test_unattainable_balance_span_is_span_limited():
    """With no noise the balance span falls below the grid floor, and that must
    be reported rather than clipped silently."""
    r = measure_pixel(GridField(SMOOTH, spacing=SP), C, m=16, sigma=0.0)
    assert r.C3_status == "SPAN-LIMITED"
    assert r.span_min >= SP - 1e-12


def test_span_never_collapses_below_the_grid():
    for sg in (0.0, 1e-8, 1e-6):
        r = measure_pixel(GridField(_noisy(SMOOTH, sg), spacing=SP), C,
                          m=16, sigma=sg)
        assert r.span_min >= SP - 1e-12, (sg, r.span_min)


# --------------------------------------------------------- geometry failures
def test_insufficient_rank_is_unusable():
    mask = np.abs(Y) < 0.5 * SP           # one-pixel strip cannot span Sym_2
    r = measure_pixel(GridField(SMOOTH, spacing=SP, mask=mask), C, m=16)
    assert r.verdict == "UNUSABLE"
    assert r.H_hat is None


def test_missing_stencil_is_unusable():
    mask = np.hypot(X, Y) < 0.20
    g = GridField(SMOOTH, spacing=SP, mask=mask)
    edge = [(i, j) for i in range(N) for j in range(N)
            if mask[i, j] and not g.valid(np.array([i, j], float))]
    assert len(edge) > 10
    for idx in edge[::max(1, len(edge) // 6)]:
        assert measure_pixel(g, idx, m=12).verdict == "UNUSABLE"


# ------------------------------------------------------------------- order
def test_antipodal_gives_second_order_and_bare_set_first():
    def f(p):
        x, y, z = p
        return np.exp(x + 0.5 * y) * np.cos(z) + 0.2 * x ** 3 + x * y * z

    def H(p):
        x, y, z = p
        e = np.exp(x + 0.5 * y); c, s = np.cos(z), np.sin(z)
        return np.array([[e*c + 1.2*x, 0.5*e*c + z, -e*s + y],
                         [0.5*e*c + z, 0.25*e*c, -0.5*e*s + x],
                         [-e*s + y, -0.5*e*s + x, -e*c]])

    def order(dirs, h0=0.1):
        x0 = np.full(3, 0.1)
        errs = []
        for h in (h0, h0 / 2):
            q = np.array([probe(f, x0, u, h) for u in dirs])
            Hh = vec_to_H(np.linalg.lstsq(design_matrix(dirs), q, rcond=None)[0], 3)
            errs.append(np.linalg.norm(Hh - H(x0), "fro"))
        return np.log2(errs[0] / errs[1])

    ico = icosahedral_axes()
    both = np.vstack([ico, -ico])
    assert antipodal_sampled_fraction(ico) == 0.0
    assert antipodal_sampled_fraction(both) == 1.0
    assert 0.7 < order(ico) < 1.4
    assert 1.8 < order(both) < 2.2


# ------------------------------------------------- oscillatory control field
def test_oscillatory_error_grows_as_resolution_falls():
    """Control for the removed guard: at a fixed, non-degenerate point the error
    behaves as it should. The pathological point used earlier does not."""
    i, j = N // 2 + 3, N // 2 + 2
    errs = []
    for ppw in (40, 20, 10, 5):
        k = 2 * np.pi / (ppw * SP)
        F = np.sin(k * X) * np.cos(k * Y)
        x0, y0 = X[i, j], Y[i, j]
        Ht = np.array([[-k*k*np.sin(k*x0)*np.cos(k*y0), -k*k*np.cos(k*x0)*np.sin(k*y0)],
                       [-k*k*np.cos(k*x0)*np.sin(k*y0), -k*k*np.sin(k*x0)*np.cos(k*y0)]])
        r = measure_pixel(GridField(F, spacing=SP), (i, j), m=16)
        errs.append(np.linalg.norm(r.H_hat - Ht, "fro") / np.linalg.norm(Ht, "fro"))
    assert errs[0] < 0.02                       # 40 points per wavelength
    assert all(b > a for a, b in zip(errs, errs[1:])), errs


def test_underresolution_is_a_known_limitation():
    """Pinned deliberately: the reliability layer does NOT detect severe
    under-resolution. At 5 points per wavelength the error is large and the
    verdict is still GOOD. A guard for this was prototyped and removed because
    the evidence that motivated it was a pathological evaluation point. Anyone
    changing this behaviour should read the note at the top of this file."""
    i, j = N // 2 + 3, N // 2 + 2
    k = 2 * np.pi / (5 * SP)
    F = np.sin(k * X) * np.cos(k * Y)
    r = measure_pixel(GridField(F, spacing=SP), (i, j), m=16)
    assert r.verdict == "GOOD"                  # documented, not endorsed
