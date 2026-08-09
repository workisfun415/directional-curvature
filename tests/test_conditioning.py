"""Conditioning of the restricted-direction geometry."""
import numpy as np
from dircurv.analytic import (design_matrix, icosahedral_axes, fibonacci_sphere,
                              kappa_reference)


def _cap(theta, npol=24, naz=32):
    ph = np.arccos(np.linspace(np.cos(theta), 1.0, npol))
    return np.array([[np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]
                     for a in ph
                     for b in np.linspace(0, 2 * np.pi, naz, endpoint=False)])


def _arc(theta, m=2001):
    ph = np.linspace(-theta, theta, m)
    return np.c_[np.sin(ph), np.cos(ph)]


def test_icosahedral_is_the_optimal_six():
    """Known from the diffusion-tensor literature: kappa = sqrt(10)/2."""
    k = np.linalg.cond(design_matrix(icosahedral_axes()))
    assert abs(k - 0.5 * np.sqrt(10)) < 1e-6
    assert abs(k - kappa_reference(3)) < 1e-6


def test_fibonacci_six_is_rank_deficient():
    """At m=6 the Fibonacci spiral gives near-antipodal pairs, which are
    duplicate measurements because u'Hu is even in u."""
    assert np.linalg.cond(design_matrix(fibonacci_sphere(6))) > 1e10
    for m in (12, 20, 50):
        assert np.linalg.cond(design_matrix(fibonacci_sphere(m))) < 2.0


def test_kappa_grows_as_theta_to_the_minus_two():
    """The exponent, not the constant: kappa * theta^2 must stay bounded."""
    for build in (_arc, _cap):
        vals = []
        for deg in (30, 15, 7.5):
            th = np.deg2rad(deg)
            D = build(th)
            A = design_matrix(D) / np.sqrt(len(D))
            s = np.linalg.svd(A)[1]
            vals.append((s[0] / s[-1]) * th ** 2)
        assert max(vals) / min(vals) < 1.5, vals


def test_full_geometry_reference_values():
    assert abs(np.linalg.cond(design_matrix(_arc(np.pi))) - np.sqrt(2)) < 1e-3
    assert abs(kappa_reference(2) - np.sqrt(2)) < 1e-12
