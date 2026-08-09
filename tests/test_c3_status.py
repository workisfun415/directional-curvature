"""C3 must never be fabricated: it is either estimated or explicitly unavailable."""
import numpy as np
import pytest
from dircurv.analytic import Geometry, measure


def _f(p):
    x, y, z = p
    return np.exp(x + 0.5 * y) * np.cos(z) + 0.2 * x ** 3 + x * y * z


def test_valid_in_free_geometry():
    r = measure(_f, np.full(3, 0.1), Geometry.free(3, 0.3), sigma=1e-6, m=24)
    assert r.C3_status == "VALID"
    assert r.C3_hat is not None and r.C3_hat > 0


def test_unavailable_in_a_slab_and_deferred():
    r = measure(_f, np.full(3, 0.1), Geometry.slab(3, 0.03, 0.3),
                sigma=1e-6, m=24)
    assert r.C3_status in ("SPAN-LIMITED", "NOISE-LIMITED")
    assert r.C3_hat is None
    assert r.verdict == "DEFER-TO-QUADRATIC-REGRESSION"


def test_user_supplied_prior_is_labelled():
    r = measure(_f, np.full(3, 0.1), Geometry.slab(3, 0.03, 0.3),
                sigma=1e-6, m=24, c3_prior=2.0)
    assert r.C3_status == "USER-SUPPLIED" and r.C3_hat == 2.0


def test_require_policy_raises():
    with pytest.raises(ValueError):
        measure(_f, np.full(3, 0.1), Geometry.slab(3, 0.03, 0.3),
                sigma=1e-6, m=24, on_c3_unavailable="require")


def test_under_determined_when_too_few_directions():
    r = measure(_f, np.full(3, 0.1), Geometry.free(3, 0.3), sigma=0.0, m=6)
    assert r.C3_status == "UNDER-DETERMINED"
