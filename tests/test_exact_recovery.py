"""Exactness gates. These must pass to machine precision; if they do not,
nothing downstream is meaningful."""
import numpy as np
import pytest
from dircurv.analytic import Geometry, measure, design_matrix, vec_to_H, probe
from dircurv.grid2d import GridField, measure_pixel


def test_analytic_quadratic_2d():
    H = np.array([[3.0, -1.0], [-1.0, 2.0]])
    f = lambda p: 0.5 * p @ H @ p + np.array([0.7, -0.4]) @ p + 1.3
    r = measure(f, [0.1, 0.1], Geometry.free(2, 0.3), sigma=0.0, m=12)
    # At sigma=0 the span rule sits at the round-off optimum h ~ eps^(1/3),
    # and the probe divides by h^2, so the floor is eps/h^2 ~ 1e-6. This is the
    # correct behaviour, not a defect: a smaller h would be worse.
    assert np.abs(r.H_hat - H).max() < 1e-5


def test_analytic_quadratic_3d():
    H = np.array([[3.0, -1.0, 0.5], [-1.0, 2.0, 0.2], [0.5, 0.2, 1.5]])
    f = lambda p: 0.5 * p @ H @ p + np.array([0.7, -0.4, 0.1]) @ p
    r = measure(f, [0.1, 0.1, 0.1], Geometry.free(3, 0.3), sigma=0.0, m=24)
    assert np.abs(r.H_hat - H).max() < 1e-5


def test_grid_quadratic_bicubic_is_exact():
    """Bicubic reproduces cubics, so a quadratic must come back exactly.
    A lower-order interpolant would inject error of the same size as the signal
    the method measures; this test is what caught that."""
    n, sp = 81, 0.02
    yy, xx = np.mgrid[0:n, 0:n]
    X, Y = (xx - n // 2) * sp, (yy - n // 2) * sp
    H = np.array([[3.0, -1.0], [-1.0, 2.0]])
    F = 0.5 * (H[0, 0] * X**2 + 2 * H[0, 1] * X * Y + H[1, 1] * Y**2) \
        + 0.7 * X - 0.4 * Y + 1.3
    r = measure_pixel(GridField(F, spacing=sp), (n // 2, n // 2), m=16)
    assert np.abs(r.H_hat - H).max() < 1e-10


def test_probe_annihilates_affine():
    f = lambda p: 2.0 + 3.0 * p[0] - 1.5 * p[1]
    for h in (0.3, 0.1, 0.01):
        assert abs(probe(f, np.zeros(2), np.array([0.6, 0.8]), h)) < 1e-8


@pytest.mark.parametrize("dim,m", [(2, 12), (3, 24)])
def test_design_roundtrip(dim, m):
    rng = np.random.default_rng(0)
    U = rng.normal(size=(m, dim))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    H = rng.normal(size=(dim, dim)); H = H + H.T
    q = np.array([u @ H @ u for u in U])
    hv = np.linalg.lstsq(design_matrix(U), q, rcond=None)[0]
    assert np.abs(vec_to_H(hv, dim) - H).max() < 1e-10
