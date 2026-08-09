"""Attainable order is set by SAMPLED antipodal partners, not by feasible ones."""
import numpy as np
from dircurv.analytic import (Geometry, icosahedral_axes, fibonacci_sphere,
                              antipodal_sampled_fraction, design_matrix,
                              vec_to_H, probe)


def _f(p):
    x, y, z = p
    return np.exp(x + 0.5 * y) * np.cos(z) + 0.2 * x ** 3 + x * y * z


def _H(p):
    x, y, z = p
    e = np.exp(x + 0.5 * y); c, s = np.cos(z), np.sin(z)
    return np.array([[e*c + 1.2*x, 0.5*e*c + z, -e*s + y],
                     [0.5*e*c + z, 0.25*e*c, -0.5*e*s + x],
                     [-e*s + y, -0.5*e*s + x, -e*c]])


def _order(dirs, h0=0.1):
    x0 = np.full(3, 0.1)
    errs = []
    for h in (h0, h0 / 2):
        q = np.array([probe(_f, x0, u, h) for u in dirs])
        Hh = vec_to_H(np.linalg.lstsq(design_matrix(dirs), q, rcond=None)[0], 3)
        errs.append(np.linalg.norm(Hh - _H(x0), "fro"))
    return np.log2(errs[0] / errs[1])


def test_sampled_fraction_detects_the_difference():
    ico = icosahedral_axes()
    assert antipodal_sampled_fraction(ico) == 0.0
    assert antipodal_sampled_fraction(np.vstack([ico, -ico])) == 1.0
    fib = fibonacci_sphere(24)
    assert antipodal_sampled_fraction(fib) == 0.0
    assert antipodal_sampled_fraction(np.vstack([fib, -fib])) == 1.0


def test_feasible_is_not_sufficient():
    """In free space every antipode is feasible, yet the icosahedral set gives
    first order because none is probed."""
    g = Geometry.free(3, 0.3)
    ico = icosahedral_axes()
    assert g.antipodal_feasible_fraction(ico) == 1.0
    assert antipodal_sampled_fraction(ico) == 0.0


def test_order_one_without_sampled_antipodes():
    assert 0.7 < _order(icosahedral_axes()) < 1.4
    assert 0.7 < _order(fibonacci_sphere(24)) < 1.4


def test_order_two_with_sampled_antipodes():
    ico = icosahedral_axes()
    assert 1.8 < _order(np.vstack([ico, -ico])) < 2.2
    fib = fibonacci_sphere(24)
    assert 1.8 < _order(np.vstack([fib, -fib])) < 2.2
